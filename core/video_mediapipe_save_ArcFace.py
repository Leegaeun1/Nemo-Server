import os
import sys
from unittest.mock import MagicMock

# 1. 라이브러리 충돌 및 부재 우회 (반드시 최상단)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.modules["retinaface"] = MagicMock()
sys.modules["retinaface.RetinaFace"] = MagicMock()

import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace

# 2. 모델 로드
face_detector = YOLO('models/yolov11n-face.pt').to('cuda')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.3)

# --- Re-ID 및 상태 관리 변수 ---
face_bank = {}          # {고정_ID: 얼굴_임베딩}
yolo_to_fixed_map = {}   # {현재_YOLO_ID: 최종_고정_ID}
unblurred_ids = set()    # 블러 해제할 고정 ID들
current_face_locations = {} 

# 임계값 대폭 하향 (0.25 ~ 0.3): 낮을수록 "완전 똑같아야" 같은 사람으로 인식합니다.
# 4명이 한 명으로 묶이는 것을 방지하기 위해 0.28로 설정합니다.
REID_THRESHOLD = 0.28 

# 3. 영상 입출력 설정
video_path = "Test_video/input_video1.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
out = cv2.VideoWriter('Test_video/output_final_strict.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

def get_fixed_id(face_img, yolo_id):
    """ArcFace 임베딩 비교 (매우 엄격한 기준 적용)"""
    try:
        if face_img.size == 0 or face_img.shape[0] < 40: return yolo_id
        
        # ArcFace 특징 추출
        emb = DeepFace.represent(face_img, model_name='ArcFace', enforce_detection=False, detector_backend='skip')
        curr_vec = np.array(emb[0]["embedding"])

        best_match_id = None
        min_dist = REID_THRESHOLD 

        for f_id, stored_vec in face_bank.items():
            # 코사인 거리 계산 (0에 가까울수록 동일인)
            dist = 1 - (np.dot(curr_vec, stored_vec) / (np.linalg.norm(curr_vec) * np.linalg.norm(stored_vec)))
            if dist < min_dist:
                min_dist = dist
                best_match_id = f_id

        if best_match_id is not None:
            return best_match_id
        else:
            # 기존 뱅크에 없으면 새 인물로 등록
            face_bank[yolo_id] = curr_vec
            return yolo_id
    except:
        return yolo_id

def select_face(event, x, y, flags, param):
    global unblurred_ids
    if event == cv2.EVENT_LBUTTONDOWN:
        for f_id, (x1, y1, x2, y2) in current_face_locations.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if f_id in unblurred_ids: unblurred_ids.remove(f_id)
                else: unblurred_ids.add(f_id)
                break

cv2.namedWindow('Nemo Re-ID Strict', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Nemo Re-ID Strict', select_face)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = face_detector.track(frame, persist=True, conf=0.18, imgsz=640, device=0, verbose=False)
    new_locations = {}

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        yolo_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, y_id in zip(boxes, yolo_ids):
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

            # [Re-ID] 처음 보는 트랙 ID인 경우만 얼굴 분석 수행
            if y_id not in yolo_to_fixed_map:
                face_roi = frame[y1:y2, x1:x2].copy()
                yolo_to_fixed_map[y_id] = get_fixed_id(face_roi, y_id)
            
            fixed_id = yolo_to_fixed_map[y_id]
            new_locations[fixed_id] = (x1, y1, x2, y2)

            # 클릭으로 블러 해제된 경우
            if fixed_id in unblurred_ids:
                cv2.putText(frame, f"ID:{fixed_id} OPEN", (x1, y1-10), 0, 0.6, (0, 255, 0), 2)
                continue 

            # --- 블러 처리 로직 (고도화) ---
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                h, w = roi.shape[:2]
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(rgb_roi)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                
                if mesh_results.multi_face_landmarks:
                    # 1. MediaPipe 성공 시: 얼굴 굴곡에 맞춘 Convex Hull 블러
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in mesh_results.multi_face_landmarks[0].landmark]
                    hull = cv2.convexHull(np.array(points))
                    cv2.fillConvexPoly(mask, hull, 255)
                else:
                    # 2. MediaPipe 실패 시: 사각형이 아닌 '타원형(Ellipse)' 블러 적용
                    center = (w // 2, h // 2)
                    axes = (w // 2, h // 2)
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

                # 공통 블러 적용
                roi_blur = cv2.GaussianBlur(roi, (121, 121), 40)
                frame[y1:y2, x1:x2] = np.where(mask[:, :, None] == 255, roi_blur, roi)

    current_face_locations = new_locations
    out.write(frame)
    cv2.imshow('Nemo Re-ID Strict', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

print("✅ 처리 완료! 저장됨.")
cap.release()
out.release()
cv2.destroyAllWindows()