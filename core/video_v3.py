import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace # pip install deepface 필수

# 1. 모델 로드
person_model = YOLO('models/yolo11n.pt') 
face_model = YOLO('models/yolov8n-face.pt')
known_face_db = {}
id_map = {}
verified_ids = set()
next_p_id = 1
frame_count = 0

def get_permanent_id(face_img):
    global next_p_id
    try:
        embeddings = DeepFace.represent(face_img, model_name="ArcFace", 
                                       enforce_detection=False, detector_backend='skip')
        current_vec = embeddings[0]["embedding"]
        best_match_id = None
        min_dist = 0.65 

        for p_id, ref_vec in known_face_db.items():
            dist = DeepFace.verification.cosine_distance(current_vec, ref_vec)
            if dist < min_dist:
                min_dist = dist
                best_match_id = p_id
        
        if best_match_id: return best_match_id
        else:
            new_id = next_p_id
            known_face_db[new_id] = current_vec
            next_p_id += 1
            return new_id
    except: return None

cap = cv2.VideoCapture("Test_video/input_video1.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1

    # 1. 전신 추적 (ID 유지용 자석 역할)
    results = person_model.track(frame, persist=True, tracker="botsort.yaml", 
                                 conf=0.4, iou=0.5, imgsz=480, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        tracker_ids = results[0].boxes.id.int().cpu().tolist()

        for box, t_id in zip(boxes, tracker_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # 몸 박스의 상단(얼굴 예상 부위) 크롭
            roi = frame[max(0, y1):y1+int((y2-y1)*0.45), max(0, x1):x2]
            if roi.size == 0: continue

            # 2. 얼굴 탐지
            face_res = face_model.predict(roi, conf=0.3, verbose=False)
            
            # 최종 ID 결정
            final_id = id_map.get(t_id, f"Temp-{t_id}")

            if len(face_res[0].boxes) > 0:
                # 얼굴 박스 좌표 계산
                fb = face_res[0].boxes.xyxy[0].cpu().numpy()
                fx1, fy1, fx2, fy2 = map(int, fb)
                
                # 얼굴 크롭 및 ArcFace 식별 (30프레임마다 혹은 처음 한 번)
                if t_id not in verified_ids or frame_count % 30 == 0:
                    face_crop = roi[fy1:fy2, fx1:fx2]
                    if face_crop.size > 0:
                        p_id = get_permanent_id(face_crop)
                        if p_id:
                            id_map[t_id] = p_id
                            verified_ids.add(t_id)
                            final_id = p_id

                # --- 시각화: 오직 얼굴에만 박스 그리기 ---
                # 원본 프레임 기준으로 좌표 변환 (roi 시작점 x1, y1 더하기)
                abs_fx1, abs_fy1 = x1 + fx1, y1 + fy1
                abs_fx2, abs_fy2 = x1 + fx2, y1 + fy2
                
                cv2.rectangle(frame, (abs_fx1, abs_fy1), (abs_fx2, abs_fy2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {final_id}", (abs_fx1, abs_fy1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # 얼굴이 안 보일 때 (뒤돌아 있을 때 등)
                # 몸 주변에 작게 ID만 띄워주거나, 아무것도 안 그리게 설정 가능
                cv2.putText(frame, f"Tracking ID: {final_id}", (x1, y1+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
    cv2.namedWindow('Nemo ArcFace + YOLO Tracker', cv2.WINDOW_NORMAL)
    cv2.imshow("Nemo ArcFace + YOLO Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()