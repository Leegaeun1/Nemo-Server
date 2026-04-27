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
# "부위만 보여도 전체 얼굴로 고정" 로직 적용
# ==========================================
class HeatmapEngine:
    def __init__(self, frame_width, frame_height, grid_scale=0.1):
        self.grid_w = int(frame_width * grid_scale)
        self.grid_h = int(frame_height * grid_scale)
        self.scale = grid_scale
        self.channels = {}       
        self.last_sizes = {}     
        self.last_centers = {} 
        self.base_decay = 0.96 # 쓰레기 잔상을 빨리 지우기 위해 감쇠율 살짝 낮춤

    def get_visual_heatmap(self):
        combined_heat = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for f_id, heatmap in self.channels.items():
            combined_heat = np.maximum(combined_heat, heatmap) 
        
        normalized_heat = np.clip(combined_heat / 10.0 * 255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized_heat, cv2.COLORMAP_JET)
        
        gray = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(colored_heatmap, colored_heatmap, mask=mask)

    def update_and_recover(self, boxes, ids, confs):
        for f_id in list(self.channels.keys()):
            self.channels[f_id] *= self.base_decay

        final_boxes = []
        final_ids = []
        seen_ids = set()

        # 1. "히트맵 안에 얼굴 부위가 있다면!" -> ID 매칭 로직
        for box, f_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            hcx, hcy = int(cx * self.scale), int(cy * self.scale)
            
            matched_id = f_id
            
            # 잡힌 부위가 기존 히트맵 영역 안인지 확인
            if f_id not in self.channels: 
                best_id = None
                best_heat = 0.5 
                for old_id, heatmap in self.channels.items():
                    if 0 <= hcy < self.grid_h and 0 <= hcx < self.grid_w:
                        if heatmap[hcy, hcx] > best_heat:
                            best_heat = heatmap[hcy, hcx]
                            best_id = old_id
                if best_id is not None:
                    matched_id = best_id 
            
            # 중복 탐지 방지
            if matched_id in seen_ids:
                continue
            seen_ids.add(matched_id)

            # 온전한 얼굴인가, 아니면 눈/코/머리카락 일부인가?
            is_full_face = True
            
            if conf < 0.45:
                is_full_face = False
            elif matched_id in self.last_sizes:
                prev_w, prev_h = self.last_sizes[matched_id]
                # 크기가 20% 이상 확 줄어들었다면 (팔에 가려져서 눈만 잡힘)
                if w < prev_w * 0.8 or h < prev_h * 0.8:
                    is_full_face = False

            # 닻(Anchor) 업데이트 안전 로직 추가
            if matched_id in self.last_sizes:
                # 1. 이미 아는 사람인 경우
                if is_full_face:
                    # 온전한 얼굴이면 정상적으로 중심점과 크기(닻)를 부드럽게 업데이트!
                    prev_w, prev_h = self.last_sizes[matched_id]
                    pcx, pcy = self.last_centers[matched_id]
                    self.last_sizes[matched_id] = (int(prev_w * 0.7 + w * 0.3), int(prev_h * 0.7 + h * 0.3))
                    self.last_centers[matched_id] = (int(pcx * 0.7 + cx * 0.3), int(pcy * 0.7 + cy * 0.3))
                else:
                    # 부분만 잡혔다면? 닻(중심/크기)은 절대 업데이트 안 함! 무시!
                    pass
            else:
                # 2. 처음 보는 ID라면? 과거가 없으므로 무조건 일단 닻을 내림!
                self.last_sizes[matched_id] = (w, h)
                self.last_centers[matched_id] = (cx, cy)

            # 히트맵 갱신 및 박스 출력은 항상 '닻(가장 최근의 온전한 얼굴)'을 기준으로!
            curr_cx, curr_cy = self.last_centers[matched_id]
            curr_w, curr_h = self.last_sizes[matched_id]

            if matched_id not in self.channels:
                self.channels[matched_id] = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            
            weight = 2.0
            gx1, gy1 = max(0, int((curr_cx - curr_w/2) * self.scale)), max(0, int((curr_cy - curr_h/2) * self.scale))
            gx2, gy2 = min(self.grid_w, int((curr_cx + curr_w/2) * self.scale)), min(self.grid_h, int((curr_cy + curr_h/2) * self.scale))
            
            self.channels[matched_id][gy1:gy2, gx1:gx2] += weight
            self.channels[matched_id] = np.clip(self.channels[matched_id], 0, 10.0)
            
            # 무조건 닻 위치에 거대한 원래 박스를 던짐
            final_boxes.append([curr_cx - curr_w//2, curr_cy - curr_h//2, curr_cx + curr_w//2, curr_cy + curr_h//2])
            final_ids.append(matched_id)

        # 2. 아예 놓친 프레임 복원 (YOLO가 아무것도 못 잡음)
        missing_ids = set(self.channels.keys()) - seen_ids
        for f_id in missing_ids:
            heatmap = self.channels[f_id]
            if np.max(heatmap) > 0.5: 
                if f_id in self.last_centers and f_id in self.last_sizes:
                    curr_cx, curr_cy = self.last_centers[f_id]
                    curr_w, curr_h = self.last_sizes[f_id]
                    final_boxes.append([curr_cx - curr_w//2, curr_cy - curr_h//2, curr_cx + curr_w//2, curr_cy + curr_h//2])
                    final_ids.append(f_id)
            else:
                del self.channels[f_id]
                if f_id in self.last_centers: del self.last_centers[f_id]
                if f_id in self.last_sizes: del self.last_sizes[f_id]

        return final_boxes, final_ids

# ==========================================
# AdaFace 및 InsightFace 모듈 로드
# ==========================================
try:
    from net import build_model
except ImportError:
    print("AdaFace 'net.py'가 필요합니다.")
    exit()

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError:
    print("insightface 라이브러리가 없습니다.")
    exit()

# ==========================================
# 1. 딥러닝 모델 초기화
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 실행 디바이스: {device} (서버 연산 모드)")

try:
    face_detector = YOLO('models/yolov11n-face.pt').to(device)
except:
    face_detector = YOLO('yolov8n-face.pt').to(device)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=False, min_detection_confidence=0.3)
static_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)

face_aligner = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_aligner.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

adaface_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("AdaFace 모델 로드 중...")
adaface_model = build_model('ir_50').to(device)
checkpoint = torch.load('adaface_ir50_ms1mv2.ckpt', map_location=device)
state_dict = checkpoint.get('state_dict', checkpoint)
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
adaface_model.load_state_dict(state_dict, strict=False)
adaface_model.eval()
print("AdaFace 모델 로드 성공 (하이브리드 모드)")

def is_same_person(embed1, embed2, threshold=0.4):
    e1 = embed1.flatten() / (np.linalg.norm(embed1) + 1e-8)
    e2 = embed2.flatten() / (np.linalg.norm(embed2) + 1e-8)
    distance = np.dot(e1, e2)
    return distance > threshold, distance

def overlay_glasses(img, glasses_bgra, face_landmarks):
    h, w = img.shape[:2]
    lx, ly = int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h)
    rx, ry = int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h)
    
    dx, dy = rx - lx, ry - ly
    eye_dist = math.sqrt(dx**2 + dy**2)
    angle = math.degrees(math.atan2(dy, dx))
    
    scale = (eye_dist * 2.2) / glasses_bgra.shape[1]
    g_w, g_h = int(glasses_bgra.shape[1] * scale), int(glasses_bgra.shape[0] * scale)
    if g_w == 0 or g_h == 0: return img
    resized_glasses = cv2.resize(glasses_bgra, (g_w, g_h))
    
    center = (g_w // 2, g_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_glasses = cv2.warpAffine(resized_glasses, M, (g_w, g_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    cx, cy = (lx + rx) // 2, (ly + ry) // 2
    x1 = cx - g_w // 2
    y1 = cy - int(g_h * 0.45)
    x2, y2 = x1 + g_w, y1 + g_h
    
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return img 
        
    result = img.copy()
    alpha_s = rotated_glasses[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        result[y1:y2, x1:x2, c] = (alpha_s * rotated_glasses[:, :, c] + alpha_l * result[y1:y2, x1:x2, c])
        
    return result

# ==========================================
# 2. 인물 등록 (가상 템플릿 증강)
# ==========================================
print("\n인물 등록 및 가상 안경 템플릿 생성 시작...")
known_embeddings = []

glasses_paths = glob.glob("glasses/*.png")
glasses_list = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in glasses_paths]
glasses_list = [g for g in glasses_list if g is not None and g.shape[2] == 4]
print(f"준비된 가상 안경: {len(glasses_list)}개")

img_paths = glob.glob("Test_person/rei3.*")

for img_path in img_paths:
    img = cv2.imread(img_path) 
    if img is None: continue
    
    faces = face_aligner.get(img)
    if not faces:
        print(f"{os.path.basename(img_path)}에서 얼굴을 찾을 수 없습니다.")
        continue
        
    target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    aligned_bgr = face_align.norm_crop(img, landmark=target_face.kps, image_size=112)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    
    input_tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = adaface_model(input_tensor)[0].cpu().numpy()
    known_embeddings.append(embedding)
    print(f"✅ {os.path.basename(img_path)} 원본 등록 완료")

    if len(glasses_list) > 0:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_results = static_face_mesh.process(img_rgb)
        
        if mp_results.multi_face_landmarks:
            landmarks = mp_results.multi_face_landmarks[0]
            successful_augs = 0
            
            for glasses_bgra in glasses_list:
                augmented_img = overlay_glasses(img, glasses_bgra, landmarks)
                
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
video_path = os.path.abspath("Test_video/test_video4.mp4")
cap = cv2.VideoCapture(video_path)

out = cv2.VideoWriter('Test_video/output_hybrid_heatmap_rei6.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                      cap.get(cv2.CAP_PROP_FPS) or 30, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
heatmap_engine = HeatmapEngine(frame_width, frame_height, grid_scale=0.1)

checked_identities = {}
SIMILARITY_THRESHOLD = 0.35 
print("\n🔍 [1/2 단계] 영상 전체를 스캔하며 신원을 파악 중입니다...")

global_identities = {} 
identity_votes = {}    
VOTE_REQUIREMENT = 3
frame_tracking_data = [] 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = face_detector.track(frame, persist=True, conf=0.10, imgsz=640, device=device, verbose=False)
    
    current_boxes, current_ids, current_confs = [], [], []
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        current_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        current_ids = results[0].boxes.id.int().cpu().tolist()
        current_confs = results[0].boxes.conf.cpu().tolist()
        
    all_boxes, all_ids = heatmap_engine.update_and_recover(current_boxes, current_ids, current_confs)

    for box, f_id in zip(all_boxes, all_ids):
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
                    for i, known_emb in enumerate(known_embeddings):
                        same, sim = is_same_person(target_embed, known_emb, SIMILARITY_THRESHOLD)
                        if sim > best_sim: best_sim = sim
                    
                    if best_sim > SIMILARITY_THRESHOLD:
                        identity_votes[f_id] = identity_votes.get(f_id, 0) + 1
                        if identity_votes[f_id] >= VOTE_REQUIREMENT:
                            global_identities[f_id] = True
                            print(f"🟢 ID {f_id} 최종 인식 성공 -> 영구 저장 확정! (히트맵 연동)")
                    else:
                        if f_id not in global_identities:
                            global_identities[f_id] = False 
        except Exception as e:
            pass
            
    frame_tracking_data.append((all_boxes, all_ids))
    
    heat_img = heatmap_engine.get_visual_heatmap()
    heat_img_resized = cv2.resize(heat_img, (frame.shape[1], frame.shape[0]))
    overlay_frame = cv2.addWeighted(frame, 0.7, heat_img_resized, 0.5, 0)
    
    for box, f_id in zip(all_boxes, all_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"ID:{f_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    view_frame = cv2.resize(overlay_frame, (1280, 720)) 
    cv2.imshow("Heatmap Engine Live View", view_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("✅ 분석 완료! 렌더링을 시작합니다.")
cv2.destroyWindow("Heatmap Engine Live View")

# ==========================================
# [Pass 2] 최종 렌더링 및 비디오 저장 단계
# ==========================================
print("\n🎬 [2/2 단계] 최종 영상을 렌더링하여 저장합니다...")

cap.release()
time.sleep(0.5) 

cap = cv2.VideoCapture(video_path)
frame_idx = 0

if not cap.isOpened():
    print(f"2단계 영상 재열기 실패! 경로를 확인하세요: {video_path}")

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame_idx >= len(frame_tracking_data): break

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
                    box_w = rx2 - rx1
                    box_h = ry2 - ry1
                    radius = int(max(box_w, box_h) * 0.6) 
                    
                    cx_min = max(0, center_x - radius)
                    cy_min = max(0, center_y - radius)
                    cx_max = min(img_w, center_x + radius)
                    cy_max = min(img_h, center_y + radius)
                    
                    circle_roi = frame[cy_min:cy_max, cx_min:cx_max]
                    
                    if circle_roi.size > 0:
                        c_mask = np.zeros(circle_roi.shape[:2], dtype=np.uint8)
                        local_cx = center_x - cx_min
                        local_cy = center_y - cy_min
                        cv2.circle(c_mask, (local_cx, local_cy), radius, 255, -1)
                        
                        f_blur = cv2.GaussianBlur(circle_roi, (121, 121), 50)
                        frame[cy_min:cy_max, cx_min:cx_max] = np.where(c_mask[:,:,None] == 255, f_blur, circle_roi)

    out.write(frame)
    frame_idx += 1
    
    if frame_idx % 30 == 0:
        print(f"렌더링 진행 중... ({frame_idx} 프레임 완료)")

print("✅ 모든 처리가 완료되었습니다!")
cap.release()
out.release()
cv2.destroyAllWindows()