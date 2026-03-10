import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from deepface import DeepFace

# --- 1. GPU 및 환경 설정 ---
# DeepFace가 GPU(TensorFlow/Keras)를 쓰도록 강제 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ 현재 사용 장치: {device}")

# 2. 모델 로드 및 GPU 전송
# YOLO 모델들은 .to(device)와 predict 시 device=0을 같이 써주는게 가장 확실합니다.
person_model = YOLO('models/yolo11n.pt').to(device) 
face_model = YOLO('models/yolov11n-face.pt').to(device)
face_model.export(format='engine', device=0)

known_face_db = {}
id_map = {}
verified_ids = set()
next_p_id = 1
frame_count = 0

def get_permanent_id(face_img):
    global next_p_id
    try:
        # DeepFace.represent는 내부적으로 TensorFlow 모델을 사용합니다.
        # 가급적 초기 호출 시 GPU 메모리를 잡도록 설정되어 있어야 합니다.
        embeddings = DeepFace.represent(
            face_img, 
            model_name="ArcFace", 
            enforce_detection=False, 
            detector_backend='skip',
            align=True # 정렬을 추가
        )
        current_vec = embeddings[0]["embedding"]
        
        best_match_id = None
        min_dist = 0.68 # 임계값

        for p_id, ref_vec in known_face_db.items():
            dist = DeepFace.verification.cosine_distance(current_vec, ref_vec)
            if dist < min_dist:
                min_dist = dist
                best_match_id = p_id
        
        if best_match_id: 
            return best_match_id
        else:
            new_id = next_p_id
            known_face_db[new_id] = current_vec
            next_p_id += 1
            return new_id
    except Exception as e:
        return None

# 비디오 경로 확인 필수
cap = cv2.VideoCapture("Test_video/input_video2.mp4")

# --- ⚡ 성능 향상을 위한 설정 ---
# 윈도우 창을 미리 생성하여 루프 밖으로 뺌
cv2.namedWindow('Nemo High-Speed Tracker', cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1

    # --- 3. YOLO 추적 시 GPU 최적화 ---
    # stream=True와 함께 device=0 사용
    results_generator = person_model.track(
        frame, persist=True, tracker="botsort.yaml", 
        conf=0.5, iou=0.5, imgsz=640, # imgsz는 32의 배수가 좋습니다 (640 권장)
        verbose=False,
        device=0, stream=True 
    )

    for result in results_generator:
        if result.boxes is not None and result.boxes.id is not None:
            # .cpu().numpy()는 필요한 순간에만 최소한으로 호출
            boxes = result.boxes.xyxy.cpu().numpy()
            tracker_ids = result.boxes.id.int().cpu().tolist()

            for box, t_id in zip(boxes, tracker_ids):
                x1, y1, x2, y2 = map(int, box)
                # ROI 추출 시 좌표 범위를 프레임 크기에 맞게 제한
                y_max = y1 + int((y2-y1)*0.45)
                roi = frame[max(0, y1):min(frame.shape[0], y_max), max(0, x1):min(frame.shape[1], x2)]
                
                if roi.size == 0: continue

                # 얼굴 모델 GPU 실행
                face_res = face_model.predict(roi, conf=0.3, verbose=False, device=0)
                final_id = id_map.get(t_id, f"Temp-{t_id}")

                if len(face_res[0].boxes) > 0:
                    fb = face_res[0].boxes.xyxy[0].cpu().numpy()
                    fx1, fy1, fx2, fy2 = map(int, fb)
                    
                    # 식별 주기 최적화: 처음 발견됐거나 10프레임마다 갱신
                    if t_id not in verified_ids or frame_count % 10 == 0:
                        face_crop = roi[max(0, fy1):fy2, max(0, fx1):fx2]
                        if face_crop.size > 0:
                            p_id = get_permanent_id(face_crop)
                            if p_id:
                                id_map[t_id] = p_id
                                verified_ids.add(t_id)
                                final_id = p_id

                    # 화면 표시 좌표 계산
                    abs_fx1, abs_fy1 = x1 + fx1, y1 + fy1
                    abs_fx2, abs_fy2 = x1 + fx2, y1 + fy2
                    
                    cv2.rectangle(frame, (abs_fx1, abs_fy1), (abs_fx2, abs_fy2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {final_id}", (abs_fx1, abs_fy1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Tracking ID: {final_id}", (x1, y1+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.namedWindow('Nemo High-Speed Tracker', cv2.WINDOW_NORMAL)
    cv2.imshow("Nemo High-Speed Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()