import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import math
from celery_worker import celery_app
import subprocess
import shutil
import firebase_admin
from firebase_admin import credentials, messaging

# ==========================================
# Firebase Admin 초기화
# ==========================================

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        print("Firebase Admin 초기화 완료")
    except Exception as e:
        print(f"Firebase Admin 초기화 실패 (알림 기능이 작동하지 않습니다): {e}")

# ==========================================
# 기존 모자이크 클래스 및 전역 모델 로드
# ==========================================
class HeatmapEngine:
    def __init__(self, frame_width, frame_height, grid_scale=0.1):
        self.grid_w = int(frame_width * grid_scale)
        self.grid_h = int(frame_height * grid_scale)
        self.scale = grid_scale
        self.channels = {}       
        self.last_sizes = {}     
        self.last_centers = {} 
        self.base_decay = 0.96
        
    def reset_memory(self):
        self.channels.clear()
        self.last_sizes.clear()
        self.last_centers.clear()

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

        for box, f_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            hcx, hcy = int(cx * self.scale), int(cy * self.scale)
            
            matched_id = f_id
            if f_id not in self.channels: 
                best_id = None
                best_heat = 0.2
                for old_id, heatmap in self.channels.items():
                    if 0 <= hcy < self.grid_h and 0 <= hcx < self.grid_w:
                        if heatmap[hcy, hcx] > best_heat:
                            best_heat = heatmap[hcy, hcx]
                            best_id = old_id
                if best_id is not None:
                    matched_id = best_id 
            
            if matched_id in seen_ids: continue
            seen_ids.add(matched_id)

            is_full_face = True
            if conf < 0.45:
                is_full_face = False
            elif matched_id in self.last_sizes:
                prev_w, prev_h = self.last_sizes[matched_id]
                if w < prev_w * 0.8 or h < prev_h * 0.8:
                    is_full_face = False

            if matched_id in self.last_sizes:
                if is_full_face:
                    prev_w, prev_h = self.last_sizes[matched_id]
                    pcx, pcy = self.last_centers[matched_id]
                    self.last_sizes[matched_id] = (int(prev_w * 0.7 + w * 0.3), int(prev_h * 0.7 + h * 0.3))
                    self.last_centers[matched_id] = (int(pcx * 0.7 + cx * 0.3), int(pcy * 0.7 + cy * 0.3))
            else:
                self.last_sizes[matched_id] = (w, h)
                self.last_centers[matched_id] = (cx, cy)

            curr_cx, curr_cy = self.last_centers[matched_id]
            curr_w, curr_h = self.last_sizes[matched_id]

            if matched_id not in self.channels:
                self.channels[matched_id] = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            
            weight = 2.0
            gx1, gy1 = max(0, int((curr_cx - curr_w/2) * self.scale)), max(0, int((curr_cy - curr_h/2) * self.scale))
            gx2, gy2 = min(self.grid_w, int((curr_cx + curr_w/2) * self.scale)), min(self.grid_h, int((curr_cy + curr_h/2) * self.scale))
            
            self.channels[matched_id][gy1:gy2, gx1:gx2] += weight
            self.channels[matched_id] = np.clip(self.channels[matched_id], 0, 10.0)
            
            final_boxes.append([curr_cx - curr_w//2, curr_cy - curr_h//2, curr_cx + curr_w//2, curr_cy + curr_h//2])
            final_ids.append(matched_id)

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


# 모듈 임포트 방어코드
try:
    from core.net import build_model
except ImportError:
    print("⚠️ AdaFace 'net.py'가 없습니다.")

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError:
    print("⚠️ insightface 라이브러리가 없습니다.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    face_detector = YOLO('models/yolov12s-face.pt').to(device)
except:
    face_detector = YOLO('models/yolov12n-face.pt').to(device)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=False, min_detection_confidence=0.3)
static_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)

try:
    face_aligner = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_aligner.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
except Exception as e:
    print(f"face_aligner 로드 에러: {e}")

adaface_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

adaface_model = build_model('ir_50').to(device)
try:
    checkpoint = torch.load('adaface_ir50_ms1mv2.ckpt', map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    adaface_model.load_state_dict(state_dict, strict=False)
    adaface_model.eval()
except:
    print("⚠️ adaface checkpoint를 찾지 못했습니다.")

def is_same_person(embed1, embed2, threshold=0.4):
    e1 = embed1.flatten() / (np.linalg.norm(embed1) + 1e-8)
    e2 = embed2.flatten() / (np.linalg.norm(embed2) + 1e-8)
    distance = np.dot(e1, e2)
    return distance > threshold, distance

def send_push_notification(token, output_path):
    print(f"발송 시도 시작! (파일명: {os.path.basename(output_path)})")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    file_size_str = f"{file_size_mb:.1f}MB"

    if not token:
        print("토큰(token)이 비어있습니다! 앱에서 토큰을 못 보낸 것 같아요.")
        return
        
    if not firebase_admin._apps:
        print("Firebase Admin이 초기화되지 않았습니다!")
        return

    try:
        print(f"🔗 [알림 디버그] FCM 서버로 전송 중... (토큰 앞글자: {token[:10]}...)")
        message = messaging.Message(
            notification=messaging.Notification(
                title='NEMO 비식별화 완료!',
                body=f'비식별화 처리가 끝났습니다! 앱으로 들어와서 다운로드해주세요.',
            ),
            data={
                "output_filename": os.path.basename(output_path),
                "file_size": file_size_str,
            },
            token=token,
        )
        response = messaging.send(message)
        print(f"[알림 성공] 서버 응답: {response}")
    except Exception as e:
        print(f"[알림 실패] 실제 에러 발생: {e}")

# ==========================================
# Celery 백그라운드 작업 
# ==========================================
@celery_app.task(name="process_video")
def process_video_task(input_path, output_path, device_token, user_id):
    print(f"🎬 [Celery 워커] 영상 처리 시작: {input_path}")
    video_name = os.path.splitext(os.path.basename(input_path))[0]  # "영상이름"
    original_name = video_name[37:] if len(video_name) > 37 else video_name  # ✅
    output_path = f"outputs/{original_name}_변환.mp4"
    known_embeddings = []
    user_faces_dir = f"user_faces/{user_id}"  # user_id 직접 사용
    face_files = glob.glob(f"{user_faces_dir}/*.*")
    print(f"👤 등록된 얼굴 {len(face_files)}개 로드")
    for img_path in face_files:
        img = cv2.imread(img_path) 
        if img is None: continue
        faces = face_aligner.get(img)
        if not faces: continue
        target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        aligned_bgr = face_align.norm_crop(img, landmark=target_face.kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = adaface_model(input_tensor)[0].cpu().numpy()
        known_embeddings.append(embedding)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다.")
        return False
        
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                          cap.get(cv2.CAP_PROP_FPS) or 30, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    heatmap_engine = HeatmapEngine(frame_width, frame_height, grid_scale=0.1)

    checked_identities = {}
    SIMILARITY_THRESHOLD = 0.35 
    global_identities = {} 
    identity_votes = {}    
    VOTE_REQUIREMENT = 3
    frame_tracking_data = [] 
    prev_frame = None
    scene_change_threshold = 30.0

    print("🔍 1단계 진행 중...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        if prev_frame is not None:
            small_curr = cv2.resize(frame, (64, 64))
            small_prev = cv2.resize(prev_frame, (64, 64))
            diff = cv2.absdiff(small_curr, small_prev)
            if np.mean(diff) > scene_change_threshold:
                heatmap_engine.reset_memory() 
                
        prev_frame = frame.copy() 

        results = face_detector.track(frame, persist=True, conf=0.30, imgsz=640, device=device, verbose=False)
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
                        for known_emb in known_embeddings:
                            same, sim = is_same_person(target_embed, known_emb, SIMILARITY_THRESHOLD)
                            if sim > best_sim: best_sim = sim
                        
                        if best_sim > SIMILARITY_THRESHOLD:
                            identity_votes[f_id] = identity_votes.get(f_id, 0) + 1
                            if identity_votes[f_id] >= VOTE_REQUIREMENT:
                                global_identities[f_id] = True
                        else:
                            if f_id not in global_identities:
                                global_identities[f_id] = False 
            except Exception as e: pass
                
        frame_tracking_data.append((all_boxes, all_ids))

    cap.release()
    print("🎬 2단계 렌더링 시작...")
    cap = cv2.VideoCapture(input_path)
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_idx >= len(frame_tracking_data): break

        boxes, ids = frame_tracking_data[frame_idx]
        for box, f_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            if global_identities.get(f_id) != True: # 모르는 사람이면 블러 처리
                img_h, img_w = frame.shape[:2]
                rx1, ry1 = max(0, x1), max(0, y1)
                rx2, ry2 = min(img_w, x2), min(img_h, y2)
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size > 0:
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
                        cv2.circle(c_mask, (center_x - cx_min, center_y - cy_min), radius, 255, -1)
                        f_blur = cv2.GaussianBlur(circle_roi, (121, 121), 50)
                        frame[cy_min:cy_max, cx_min:cx_max] = np.where(c_mask[:,:,None] == 255, f_blur, circle_roi)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    print("✅ 영상 처리 완료! ffmpeg로 재인코딩 중...")
    
    # ffmpeg로 H264 재인코딩 (모바일 호환)
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    os.rename(output_path, temp_path)
    
    result = subprocess.run([
        'ffmpeg', '-y',
        '-i', temp_path,
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '23',
        '-movflags', '+faststart',
        output_path
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    os.remove(temp_path)

    # ffmpeg 성공 여부 먼저 확인
    if result.returncode != 0:
        print(f"❌ ffmpeg 실패: {result.stderr}")
        # 실패해도 임시파일/얼굴은 정리
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(user_faces_dir):
            shutil.rmtree(user_faces_dir)
        return False  # ← 알림 발송 없이 종료

    print("✅ ffmpeg 재인코딩 완료!")

    # 성공했을 때만 정리 + 알림
    if os.path.exists(input_path):
        os.remove(input_path)
        print(f"입력 영상 삭제: {input_path}")


    send_push_notification(device_token, output_path)
    return True