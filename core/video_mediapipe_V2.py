import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# 1. 모델 및 도구 로드
face_detector = YOLO('models/yolov8n-face.pt') 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.3)

# --- 인터랙티브 설정을 위한 변수 ---
unblurred_ids = set()    # 블러를 해제할 아이디 보관함
current_face_locations = {} # 현재 화면에 있는 얼굴들의 {ID: (x1, y1, x2, y2)} 저장

# 마우스 클릭 이벤트 함수
def select_face(event, x, y, flags, param):
    global unblurred_ids
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭 시
        for f_id, (x1, y1, x2, y2) in current_face_locations.items():
            # 클릭한 좌표(x, y)가 얼굴 박스 안에 있는지 확인
            if x1 <= x <= x2 and y1 <= y <= y2:
                if f_id in unblurred_ids:
                    unblurred_ids.remove(f_id)
                    print(f"🔒 ID {f_id} 다시 블러 처리")
                else:
                    unblurred_ids.add(f_id)
                    print(f"🔓 ID {f_id} 블러 해제!")
                break

video_path = "Test_video/input_video2.mp4"
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('Nemo Interactive Blur',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Nemo Interactive Blur', select_face) # 마우스 콜백 연결

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # YOLO 추적 (persist=True 필수!)
    results = face_detector.track(frame, persist=True, conf=0.2, imgsz=640, verbose=False)
    
    new_locations = {} # 이번 프레임의 위치 갱신용

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, f_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            new_locations[f_id] = (x1, y1, x2, y2) # 위치 저장
            
            # --- [핵심] 만약 클릭해서 '해제 리스트'에 들어있는 ID라면 블러 건너뜀 ---
            if f_id in unblurred_ids:
                # 얼굴 위에 ID 표시 (누가 해제됐는지 확인용)
                cv2.putText(frame, f"OPEN ID:{f_id}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                continue 

            # --- 블러 처리 로직 (이전과 동일) ---
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            blurred_done = False

            if roi.size > 0:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(rgb_roi)

                if mesh_results.multi_face_landmarks:
                    h, w, _ = roi.shape
                    all_points = [ (int(lm.x * w), int(lm.y * h)) for lm in mesh_results.multi_face_landmarks[0].landmark ]
                    hull = cv2.convexHull(np.array(all_points))
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, hull, 255)
                    roi_blur = cv2.GaussianBlur(roi, (121, 121), 40)
                    roi = np.where(mask[:, :, None] == 255, roi_blur, roi)
                    frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)] = roi
                    blurred_done = True

            # 실패 시 원형 블러 보완
            if not blurred_done:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = int(max(x2 - x1, y2 - y1) * 0.5)
                sub_x1, sub_y1 = max(0, center_x - radius), max(0, center_y - radius)
                sub_x2, sub_y2 = min(frame.shape[1], center_x + radius), min(frame.shape[0], center_y + radius)
                face_sub_img = frame[sub_y1:sub_y2, sub_x1:sub_x2]
                if face_sub_img.size > 0:
                    c_mask = np.zeros(face_sub_img.shape[:2], dtype=np.uint8)
                    cv2.circle(c_mask, (center_x - sub_x1, center_y - sub_y1), radius, 255, -1)
                    f_blur = cv2.GaussianBlur(face_sub_img, (151, 151), 50)
                    frame[sub_y1:sub_y2, sub_x1:sub_x2] = np.where(c_mask[:,:,None] == 255, f_blur, face_sub_img)

    current_face_locations = new_locations # 전역 변수 업데이트
    cv2.namedWindow('Nemo Interactive Blur',cv2.WINDOW_NORMAL)
    cv2.imshow('Nemo Interactive Blur', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()