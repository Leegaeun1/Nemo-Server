import cv2
import numpy as np
import torch
import os

# --- 0. 환경 변수 및 의존성 에러 방지 ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import mediapipe as mp
# mp.solutions 에러 발생 시를 대비한 직접 참조
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from ultralytics import YOLO
from deepface import DeepFace

# --- 1. GPU 및 환경 설정 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ 현재 사용 장치: {device}")

# MediaPipe Face Mesh 초기화
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False, # 속도 향상을 위해 False 권장
    min_detection_confidence=0.5
)

# 2. 모델 로드 (엔진 파일 우선 로드)
try:
    person_model = YOLO('models/yolo11n.engine') 
    face_model = YOLO('models/yolov11n-face.engine')
    print("🚀 TensorRT (.engine) 모델 로드 성공!")
except Exception as e:
    print(f"⚠️ 엔진 파일 로드 실패: {e}")
    person_model = YOLO('models/yolo11n.pt').to(device)
    face_model = YOLO('models/yolov11n-face.pt').to(device)

known_face_db = {}
id_map = {}
verified_ids = set()
next_p_id = 1
frame_count = 0

# 얼굴 외곽선 인덱스
FACIAL_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def get_permanent_id(face_img):
    global next_p_id
    try:
        embeddings = DeepFace.represent(
            face_img, model_name="ArcFace", 
            enforce_detection=False, detector_backend='skip', align=True
        )
        current_vec = embeddings[0]["embedding"]
        best_match_id, min_dist = None, 0.68 

        for p_id, ref_vec in known_face_db.items():
            dist = DeepFace.verification.cosine_distance(current_vec, ref_vec)
            if dist < min_dist:
                min_dist, best_match_id = dist, p_id
        
        if best_match_id: return best_match_id
        else:
            new_id = next_p_id
            known_face_db[new_id] = current_vec
            next_p_id += 1
            return new_id
    except: return None

cap = cv2.VideoCapture("Test_video/input_video1.mp4")
cv2.namedWindow('Nemo High-Speed Tracker', cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1

    results_generator = person_model.track(
        frame, persist=True, tracker="botsort.yaml", 
        conf=0.5, iou=0.5, imgsz=640, 
        verbose=False, device=0, stream=True, classes=[0]
    )

    for result in results_generator:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            tracker_ids = result.boxes.id.int().cpu().tolist()

            for box, t_id in zip(boxes, tracker_ids):
                x1, y1, x2, y2 = map(int, box)
                roi_h = int((y2 - y1) * 0.45)
                roi = frame[max(0, y1):min(frame.shape[0], y1 + roi_h), max(0, x1):min(frame.shape[1], x2)]
                
                if roi.size == 0: continue

                final_id = id_map.get(t_id, f"Temp-{t_id}")
                
                # 얼굴 모델 실행
                face_res = face_model.predict(roi, conf=0.4, imgsz=640, verbose=False, device=0)
                
                if len(face_res[0].boxes) > 0:
                    # --- 1. 확장된 얼굴 영역(옆모습 대비) 계산 ---
                    fb = face_res[0].boxes.xyxy[0].cpu().numpy()
                    fx1, fy1, fx2, fy2 = map(int, fb)
                    face_w, face_h = fx2 - fx1, fy2 - fy1

                    # 상하좌우 패딩 적용된 좌표 (프레임 범위 제한 포함)
                    safe_abs_fx1 = max(0, x1 + fx1 - int(face_w * 0.15))
                    safe_abs_fy1 = max(0, y1 + fy1 - int(face_h * 0.20))
                    safe_abs_fx2 = min(frame.shape[1], x1 + fx2 + int(face_w * 0.15))
                    safe_abs_fy2 = min(frame.shape[0], y1 + fy2 + int(face_h * 0.20))

                    # 얼굴 원본 크롭 (식별용 - 블러 전 추출)
                    face_crop_for_id = roi[fy1:fy2, fx1:fx2].copy()

                    # --- 2. 블러 처리 로직 ---
                    # MediaPipe 정밀 마스킹 시도
                    rgb_roi = cv2.cvtColor(roi[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2RGB)
                    mesh_results = face_mesh.process(rgb_roi)

                    if mesh_results.multi_face_landmarks:
                        # [정면/반측면] MediaPipe 기반 정밀 블러
                        h, w = fy2 - fy1, fx2 - fx1
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for face_landmarks in mesh_results.multi_face_landmarks:
                            points = [(int(face_landmarks.landmark[idx].x * w), 
                                       int(face_landmarks.landmark[idx].y * h)) for idx in FACIAL_OUTLINE]
                            cv2.fillConvexPoly(mask, np.array(points), 255)
                        
                        target_roi = frame[y1+fy1:y1+fy2, x1+fx1:x1+fx2]
                        blurred_part = cv2.GaussianBlur(target_roi, (75, 75), 25)
                        frame[y1+fy1:y1+fy2, x1+fx1:x1+fx2] = np.where(mask[:, :, None] == 255, blurred_part, target_roi)
                    
                    else:
                        # [옆모습/특수각도] 제공해주신 패딩 기반 전체 블러
                        face_roi = frame[safe_abs_fy1:safe_abs_fy2, safe_abs_fx1:safe_abs_fx2]
                        if face_roi.size > 0:
                            blurred_face = cv2.GaussianBlur(face_roi, (111, 111), 35)
                            frame[safe_abs_fy1:safe_abs_fy2, safe_abs_fx1:safe_abs_fx2] = blurred_face

                    # --- 3. 식별 (DeepFace) ---
                    # 블러처리되지 않은 face_crop_for_id 사용
                    if t_id not in verified_ids or frame_count % 30 == 0:
                        if face_crop_for_id.size > 0:
                            p_id = get_permanent_id(face_crop_for_id)
                            if p_id:
                                id_map[t_id] = p_id
                                verified_ids.add(t_id)
                                final_id = p_id

                    # 정보 표시
                    cv2.putText(frame, f"ID: {final_id}", (safe_abs_fx1, safe_abs_fy1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.namedWindow('Nemo High-Speed Tracker', cv2.WINDOW_NORMAL)
    cv2.imshow("Nemo High-Speed Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()