import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image

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

# 1-1. YOLO (트래킹 및 빠른 검출)
try:
    face_detector = YOLO('models/yolov11n-face.pt').to(device)
except:
    face_detector = YOLO('yolov8n-face.pt').to(device)

# 1-2. MediaPipe (모자이크 렌더링)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=False, min_detection_confidence=0.3)

# 1-3. InsightFace (오직 '얼굴 정렬 및 랜드마크 추출' 용도로만 가볍게 로드)
# allowed_modules=['detection'] 옵션으로 무거운 인식 모듈은 메모리에 안 올립니다.
face_aligner = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_aligner.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

# 1-4. AdaFace (가림 극복 특징 추출)
adaface_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]) # Resize(112,112)는 face_align이 해주므로 생략 가능!

print("🔄 AdaFace 모델 로드 중...")
adaface_model = build_model('ir_50').to(device)
checkpoint = torch.load('adaface_ir50_ms1mv2.ckpt', map_location=device)
state_dict = checkpoint.get('state_dict', checkpoint)
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
adaface_model.load_state_dict(state_dict, strict=False)
adaface_model.eval()
print("✅ AdaFace 모델 로드 성공 (하이브리드 모드)")

# 완벽한 코사인 유사도 함수 (L2 정규화 강제 적용)
def is_same_person(embed1, embed2, threshold=0.4):
    e1 = embed1.flatten() / (np.linalg.norm(embed1) + 1e-8)
    e2 = embed2.flatten() / (np.linalg.norm(embed2) + 1e-8)
    distance = np.dot(e1, e2)
    return distance > threshold, distance

# ==========================================
# 2. 인물 등록 (Test_person 폴더)
# ==========================================
print("\n🔄 인물 등록 및 하이브리드 특징 추출 시작...")
known_embeddings = []
img_paths = glob.glob("Test_person/hosi.*")

for img_path in img_paths:
    img = cv2.imread(img_path) 
    if img is None: continue
    
    # 1. InsightFace로 얼굴 랜드마크(kps) 찾기
    faces = face_aligner.get(img)
    if not faces:
        print(f"❌ {os.path.basename(img_path)}에서 얼굴을 찾을 수 없습니다.")
        continue
        
    target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    
    # 2. [핵심] 5개의 랜드마크 점을 기준으로 얼굴을 반듯하게 회전 & 112x112 크롭!
    aligned_bgr = face_align.norm_crop(img, landmark=target_face.kps, image_size=112)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. 반듯해진 얼굴을 AdaFace에 넣어 특징 추출
    input_tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = adaface_model(input_tensor)[0].cpu().numpy()
        
    known_embeddings.append(embedding)
    print(f"✅ {os.path.basename(img_path)} 등록 완료 (InsightFace 정렬 + AdaFace 추출)")

# ==========================================
# 3. 영상 설정 및 신원 확인 메인 루프
# ==========================================
cap = cv2.VideoCapture("Test_video/test_video2.mp4")
out = cv2.VideoWriter('Test_video/output_hybrid1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                      cap.get(cv2.CAP_PROP_FPS) or 30, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

checked_identities = {}
SIMILARITY_THRESHOLD = 0.25 # AdaFace 기준 최적 유사도

print("\n🎬 하이브리드 영상 처리 시작...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. YOLO로 빠른 추적 (ID 할당)
    results = face_detector.track(frame, persist=True, conf=0.3, imgsz=640, device=device, verbose=False)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, f_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            
            # 신원 확인 로직 (ID 캐싱 적용)
            if f_id not in checked_identities:
                img_h, img_w = frame.shape[:2]
                
                # YOLO 박스보다 넉넉하게 잘라서 InsightFace에게 넘김 (랜드마크 탐지율 상승)
                pad_w, pad_h = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
                rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                rx2, ry2 = min(img_w, x2 + pad_w), min(img_h, y2 + pad_h)
                face_crop = frame[ry1:ry2, rx1:rx2]
                
                if face_crop.size > 0:
                    try:
                        # 1. 크롭된 영역에서 랜드마크 추출
                        faces_in_crop = face_aligner.get(face_crop)
                        
                        if faces_in_crop:
                            target_face = sorted(faces_in_crop, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                            
                            # 2. 얼굴 정면 정렬 (112x112)
                            aligned_bgr = face_align.norm_crop(face_crop, landmark=target_face.kps, image_size=112)
                            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                            
                            # 3. AdaFace로 특징 추출
                            input_tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
                            with torch.no_grad():
                                target_embed = adaface_model(input_tensor)[0].cpu().numpy()
                            
                            # 4. 비교
                            best_sim = -1.0
                            for known_emb in known_embeddings:
                                same, sim = is_same_person(target_embed, known_emb, SIMILARITY_THRESHOLD)
                                if sim > best_sim: best_sim = sim
                            
                            if best_sim > SIMILARITY_THRESHOLD:
                                checked_identities[f_id] = True 
                                print(f"🟢 ID {f_id} 인식 성공! (유사도: {best_sim:.3f})")
                            else:
                                checked_identities[f_id] = False 
                                print(f"🔴 ID {f_id} 모르는 사람 (유사도: {best_sim:.3f})")
                        else:
                            continue # 얼굴 특징점 못 찾으면 다음 프레임에서 재시도
                    except Exception as e:
                        print(f"⚠️ ID {f_id} 처리 에러: {e}")
                        continue
                else:
                    continue

            # ==========================================
            # 블러 처리 로직 (MediaPipe)
            # ==========================================
            if checked_identities.get(f_id, False):
                cv2.rectangle(frame, (max(0,x1), max(0,y1)), (min(frame.shape[1],x2), min(frame.shape[0],y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"KNOWN:{f_id}", (max(0,x1), max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                continue 

            rx1, ry1 = max(0, x1), max(0, y1)
            rx2, ry2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            roi = frame[ry1:ry2, rx1:rx2]
            blurred_done = False
            
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
                        blurred_done = True
                except: pass
            
            if not blurred_done: # Fallback
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = int(max(x2 - x1, y2 - y1) * 0.5)
                sub_x1, sub_y1 = max(0, center_x - radius), max(0, center_y - radius)
                sub_x2, sub_y2 = min(frame.shape[1], center_x + radius), min(frame.shape[0], center_y + radius)
                face_sub_img = frame[sub_y1:sub_y2, sub_x1:sub_x2]
                
                if face_sub_img.size > 0:
                    c_mask = np.zeros(face_sub_img.shape[:2], dtype=np.uint8)
                    cv2.circle(c_mask, (center_x - sub_x1, center_y - sub_y1), radius, 255, -1)
                    k_size = max(1, min(face_sub_img.shape[:2]) // 2 * 2 - 1)
                    if k_size >= 1:
                        f_blur = cv2.GaussianBlur(face_sub_img, (k_size, k_size), 50)
                        frame[sub_y1:sub_y2, sub_x1:sub_x2] = np.where(c_mask[:,:,None] == 255, f_blur, face_sub_img)
    
    out.write(frame)
    cv2.imshow('Hybrid Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

print("✅ 하이브리드 서버 연산 파이프라인 처리가 완료되었습니다!")
cap.release()
out.release()
cv2.destroyAllWindows()