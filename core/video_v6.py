import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace

# 1. 모델 초기화
face_detector = YOLO('models/yolov11n-face.pt').to('cuda')
face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=10, refine_landmarks=False, min_detection_confidence=0.3)

# 2. DeepFace 설정 및 DB
known_face_db = {}
id_map = {}
verified_ids = set()
next_p_id = 1

def get_permanent_id(face_img):
    global next_p_id
    try:
        embeddings = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False, detector_backend='skip')
        curr_vec = embeddings[0]["embedding"]
        best_id, min_dist = None, 0.65 # 임계값 (작을수록 엄격)

        for p_id, ref_vec in known_face_db.items():
            dist = DeepFace.verification.cosine_distance(curr_vec, ref_vec)
            if dist < min_dist:
                min_dist, best_id = dist, p_id
        
        if best_id: return best_id
        else:
            new_id = next_p_id
            known_face_db[new_id] = curr_vec
            next_p_id += 1
            return new_id
    except: return None

# 3. 영상 설정
video_path = "Test_video/input_video1.mp4"
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Test_video/final_output_deepface.mp4', fourcc, fps, (width, height))

unblurred_ids = set() # 영구 ID 기준 블러 해제 목록
current_face_locations = {}
frame_count = 0

def select_face(event, x, y, flags, param):
    global unblurred_ids
    if event == cv2.EVENT_LBUTTONDOWN:
        for p_id, (x1, y1, x2, y2) in current_face_locations.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if p_id in unblurred_ids: unblurred_ids.remove(p_id)
                else: unblurred_ids.add(p_id)
                break

cv2.namedWindow('Nemo Hybrid Tracker', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Nemo Hybrid Tracker', select_face)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1
    new_locations = {}

    results = face_detector.track(frame, persist=True, conf=0.3, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        tracker_ids = results[0].boxes.id.int().cpu().tolist()

        for box, t_id in zip(boxes, tracker_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # 30프레임마다 한 번만 DeepFace로 인물 확인
            if t_id not in verified_ids or frame_count % 30 == 0:
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                if face_crop.size > 0:
                    p_id = get_permanent_id(face_crop)
                    if p_id:
                        id_map[t_id] = p_id
                        verified_ids.add(t_id)

            final_id = id_map.get(t_id, f"T-{t_id}")
            new_locations[final_id] = (x1, y1, x2, y2)

            # 블러 여부 판단
            if final_id in unblurred_ids:
                cv2.putText(frame, f"FREE ID:{final_id}", (x1, y1-10), 0, 0.7, (0, 255, 0), 2)
                continue

            # 얼굴 내부 전체 블러
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size > 0:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mesh_res = face_mesh.process(rgb_roi)
                
                if mesh_res.multi_face_landmarks:
                    h, w = roi.shape[:2]
                    points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in mesh_res.multi_face_landmarks[0].landmark])
                    hull = cv2.convexHull(points)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, hull, 255)
                    
                    roi_blur = cv2.GaussianBlur(roi, (101, 101), 35)
                    frame[max(0, y1):y2, max(0, x1):x2] = np.where(mask[:,:,None]==255, roi_blur, roi)
                else:
                    # Fallback 박스 블러
                    frame[max(0, y1):y2, max(0, x1):x2] = cv2.GaussianBlur(roi, (101, 101), 35)

            cv2.putText(frame, f"ID:{final_id}", (x1, y1-10), 0, 0.5, (255, 255, 255), 1)

    current_face_locations = new_locations
    out.write(frame)
    cv2.imshow('Nemo Hybrid Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()