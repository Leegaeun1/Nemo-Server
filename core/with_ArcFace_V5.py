import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import math

# ==========================================
# AdaFace 및 InsightFace 모듈 로드
# ==========================================
try:
    from net import build_model  # AdaFace 아키텍처
except ImportError:
    print("❌ AdaFace 'net.py'가 필요합니다.")
    exit()

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align  # 핵심: 얼굴 정렬(Alignment) 도구
except ImportError:
    print("❌ insightface 라이브러리가 없습니다. (pip install insightface)")
    exit()

# ==========================================
# 1. 딥러닝 모델 초기화 (서버 GPU 세팅)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 실행 디바이스: {device} (서버 연산 모드)")

try:
    face_detector = YOLO('models/yolov11n-face.pt').to(device)
except:
    face_detector = YOLO('yolov8n-face.pt').to(device)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=False, min_detection_confidence=0.3)

# 등록 단계에서 원본 사진의 랜드마크를 정밀하게 찾기 위한 일회성 FaceMesh (정적 이미지용)
static_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)

face_aligner = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_aligner.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

adaface_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("🔄 AdaFace 모델 로드 중...")
adaface_model = build_model('ir_50').to(device)
checkpoint = torch.load('adaface_ir50_ms1mv2.ckpt', map_location=device)
state_dict = checkpoint.get('state_dict', checkpoint)
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
adaface_model.load_state_dict(state_dict, strict=False)
adaface_model.eval()
print("✅ AdaFace 모델 로드 성공 (하이브리드 모드)")

def is_same_person(embed1, embed2, threshold=0.4):
    e1 = embed1.flatten() / (np.linalg.norm(embed1) + 1e-8)
    e2 = embed2.flatten() / (np.linalg.norm(embed2) + 1e-8)
    distance = np.dot(e1, e2)
    return distance > threshold, distance

# 얼굴 위에 안경 PNG 동적 합성
def overlay_glasses(img, glasses_bgra, face_landmarks):
    h, w = img.shape[:2]
    # 눈 좌표 추출 (MediaPipe 기준 33: 좌측 눈 꼬리, 263: 우측 눈 꼬리)
    lx, ly = int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h)
    rx, ry = int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h)
    
    # 눈 사이 거리 및 기울기 각도 계산
    dx, dy = rx - lx, ry - ly
    eye_dist = math.sqrt(dx**2 + dy**2)
    angle = math.degrees(math.atan2(dy, dx))
    
    # 안경 크기 조절 (눈 사이 거리의 약 2.2배 크기가 평균적인 안경 너비)
    scale = (eye_dist * 2.2) / glasses_bgra.shape[1]
    g_w, g_h = int(glasses_bgra.shape[1] * scale), int(glasses_bgra.shape[0] * scale)
    if g_w == 0 or g_h == 0: return img
    resized_glasses = cv2.resize(glasses_bgra, (g_w, g_h))
    
    # 안경 회전 (고개가 기울어졌을 경우 보정)
    center = (g_w // 2, g_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_glasses = cv2.warpAffine(resized_glasses, M, (g_w, g_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    # 합성 시작점(x1, y1) 계산 (안경이 눈 정중앙보다 살짝 위에 오도록 Y축 0.45 보정)
    cx, cy = (lx + rx) // 2, (ly + ry) // 2
    x1 = cx - g_w // 2
    y1 = cy - int(g_h * 0.45)
    x2, y2 = x1 + g_w, y1 + g_h
    
    # 이미지 경계선을 벗어나는 얼굴은 원본 그대로 반환
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return img 
        
    # 알파 블렌딩 (투명도 합성)
    result = img.copy()
    alpha_s = rotated_glasses[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        result[y1:y2, x1:x2, c] = (alpha_s * rotated_glasses[:, :, c] + alpha_l * result[y1:y2, x1:x2, c])
        
    return result

# ==========================================
# 2. 인물 등록 (가상 템플릿 증강)
# ==========================================
print("\n🔄 인물 등록 및 가상 안경 템플릿 생성 시작...")
known_embeddings = []

# 가상 안경 파일 로드
glasses_paths = glob.glob("glasses/*.png")
glasses_list = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in glasses_paths]
glasses_list = [g for g in glasses_list if g is not None and g.shape[2] == 4]
print(f"👓 준비된 가상 안경: {len(glasses_list)}개")

img_paths = glob.glob("Test_person/rei3.*")

for img_path in img_paths:
    img = cv2.imread(img_path) 
    if img is None: continue
    
    # [Step 1] 원본 맨얼굴 등록
    faces = face_aligner.get(img)
    if not faces:
        print(f"❌ {os.path.basename(img_path)}에서 얼굴을 찾을 수 없습니다.")
        continue
        
    target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    aligned_bgr = face_align.norm_crop(img, landmark=target_face.kps, image_size=112)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    
    input_tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = adaface_model(input_tensor)[0].cpu().numpy()
    known_embeddings.append(embedding)
    print(f"✅ {os.path.basename(img_path)} 원본 등록 완료")

    # [Step 2] 가상 안경 합성 템플릿 등록
    if len(glasses_list) > 0:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_results = static_face_mesh.process(img_rgb)
        
        if mp_results.multi_face_landmarks:
            landmarks = mp_results.multi_face_landmarks[0]
            successful_augs = 0
            
            for glasses_bgra in glasses_list:
                augmented_img = overlay_glasses(img, glasses_bgra, landmarks)
                
                # 합성된 얼굴을 InsightFace로 다시 정렬 및 크롭
                aug_faces = face_aligner.get(augmented_img)
                if aug_faces:
                    target_aug_face = sorted(aug_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    aligned_aug_bgr = face_align.norm_crop(augmented_img, landmark=target_aug_face.kps, image_size=112)
                    aligned_aug_rgb = cv2.cvtColor(aligned_aug_bgr, cv2.COLOR_BGR2RGB)
                    
                    input_tensor = adaface_transform(Image.fromarray(aligned_aug_rgb)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        aug_embedding = adaface_model(input_tensor)[0].cpu().numpy()
                    known_embeddings.append(aug_embedding)
                    successful_augs += 1
                    
            print(f"✅ {os.path.basename(img_path)} 가상 안경 템플릿 {successful_augs}개 추가 등록 완료")

# ==========================================
# 3. 영상 설정 및 신원 확인 메인 루프 
# ==========================================
video_path = os.path.abspath("Test_video/test_video6.mp4")
cap = cv2.VideoCapture(video_path)

out = cv2.VideoWriter('Test_video/output_hybrid_rei4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                      cap.get(cv2.CAP_PROP_FPS) or 30, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

checked_identities = {}
SIMILARITY_THRESHOLD = 0.35 # 가상 템플릿이 생겼으므로 0.38~0.42 정도로 엄격하게 유지해도 인식이 잘 됩니다.
print("\n🔍 [1/2 단계] 영상 전체를 스캔하며 신원을 파악 중입니다...")

global_identities = {} 
identity_votes = {}    
VOTE_REQUIREMENT = 3

frame_tracking_data = [] 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = face_detector.track(frame, persist=True, conf=0.3, imgsz=640, device=device, verbose=False)
    
    current_boxes = []
    current_ids = []
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        current_boxes = results[0].boxes.xyxy.cpu().numpy()
        current_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, f_id in zip(current_boxes, current_ids):
            x1, y1, x2, y2 = map(int, box)
            
            if global_identities.get(f_id) == True:
                continue
                
            try:
                img_h, img_w = frame.shape[:2]
                pad_w, pad_h = int((x2 - x1) * 0.3), int((y2 - y1) * 0.3)
                rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                rx2, ry2 = min(img_w, x2 + pad_w), min(img_h, y2 + pad_h)
                
                face_crop = frame[ry1:ry2, rx1:rx2]
                
                if face_crop.size > 0:
                    faces_in_crop = face_aligner.get(face_crop)
                    
                    if faces_in_crop:
                        target_face = sorted(faces_in_crop, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                        
                        aligned_bgr = face_align.norm_crop(face_crop, landmark=target_face.kps, image_size=112)
                        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                        
                        input_tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            target_embed = adaface_model(input_tensor)[0].cpu().numpy()
                        
                        best_sim = -1.0
                        best_idx = -1  # 몇 번째 정답지랑 가장 비슷한지 추적!
                        
                        for i, known_emb in enumerate(known_embeddings):
                            same, sim = is_same_person(target_embed, known_emb, SIMILARITY_THRESHOLD)
                            if sim > best_sim: 
                                best_sim = sim
                                best_idx = i  # 최고 점수를 갱신할 때마다 인덱스 저장
                        
                        if best_sim > SIMILARITY_THRESHOLD:
                            identity_votes[f_id] = identity_votes.get(f_id, 0) + 1
                            
                            # 0번이면 원본 맨얼굴, 1번 이상이면 합성된 가상 안경!
                            match_type = "원본 맨얼굴" if best_idx == 0 else f"가상 안경 #{best_idx}"
                            print(f"👍 ID {f_id} 득표! (매칭: {match_type}, 유사도: {best_sim:.3f})")
                            
                            if identity_votes[f_id] >= VOTE_REQUIREMENT:
                                global_identities[f_id] = True
                                print(f"🟢 ID {f_id} 최종 인식 성공 -> 영구 저장 확정!")
                        else:
                            if f_id not in global_identities:
                                global_identities[f_id] = False 
            except Exception as e:
                pass
                
    frame_tracking_data.append((current_boxes, current_ids))

print("✅ 분석 완료! 렌더링을 시작합니다.")

# ==========================================
# [Pass 2] 최종 렌더링 및 비디오 저장 단계
# ==========================================
print("\n🎬 [2/2 단계] 최종 영상을 렌더링하여 저장합니다...")

cap.release()
time.sleep(0.5) 

cap = cv2.VideoCapture(video_path)
frame_idx = 0

if not cap.isOpened():
    print(f"❌ 2단계 영상 재열기 실패! 경로를 확인하세요: {video_path}")

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame_idx >= len(frame_tracking_data): 
        break

    boxes, ids = frame_tracking_data[frame_idx]
    
    for box, f_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        
        if global_identities.get(f_id) == True:
            cv2.rectangle(frame, (max(0,x1), max(0,y1)), (min(frame.shape[1],x2), min(frame.shape[0],y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"KNOWN:{f_id}", (max(0,x1), max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            img_h, img_w = frame.shape[:2]
            rx1, ry1 = max(0, x1), max(0, y1)
            rx2, ry2 = min(img_w, x2), min(img_h, y2)
            roi = frame[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                try:
                    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    mesh_results = face_mesh.process(rgb_roi)
                    if mesh_results.multi_face_landmarks:
                        h, w, _ = roi.shape
                        all_points = [(int(lm.x * w), int(lm.y * h)) for lm in mesh_results.multi_face_landmarks[0].landmark]
                        hull = cv2.convexHull(np.array(all_points))
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, hull, 255)
                        roi_blur = cv2.GaussianBlur(roi, (121, 121), 40)
                        roi = np.where(mask[:, :, None] == 255, roi_blur, roi)
                        frame[ry1:ry2, rx1:rx2] = roi
                    else:
                        raise Exception("MediaPipe fallback")
                except:
                    center_x, center_y = (rx1 + rx2) // 2, (ry1 + ry2) // 2
                    radius = int(max(rx2 - rx1, ry2 - ry1) * 0.5)
                    c_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.circle(c_mask, (roi.shape[1]//2, roi.shape[0]//2), radius, 255, -1)
                    k_size = max(1, min(roi.shape[:2]) // 2 * 2 - 1)
                    f_blur = cv2.GaussianBlur(roi, (k_size, k_size), 50)
                    frame[ry1:ry2, rx1:rx2] = np.where(c_mask[:,:,None] == 255, f_blur, roi)

    out.write(frame)
    frame_idx += 1
    
    if frame_idx % 30 == 0:
        print(f"렌더링 진행 중... ({frame_idx} 프레임 완료)")

print("✅ 모든 처리가 완료되었습니다!")
cap.release()
out.release()
cv2.destroyAllWindows()