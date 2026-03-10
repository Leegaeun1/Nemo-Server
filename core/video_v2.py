import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 로드 (전신 추적용 m모델 + 얼굴 탐지용 모델)
person_model = YOLO('models/yolo11m.pt') 
face_model = YOLO('models/yolov8n-face.pt')

def get_rotated_face(roi, face_model):
    """ROI(사람 몸)를 회전시켜가며 얼굴을 탐지 (rotate.py의 핵심 활용)"""
    h, w = roi.shape[:2]
    center = (w // 2, h // 2)
    
    # 0도(정면), -30도, 30도 세 각도만 확인 (속도와 정확도의 타협점)
    for angle in [0, -30, 30]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_roi = cv2.warpAffine(roi, M, (w, h))
        
        # 얼굴 탐지 (문턱을 낮춰서 끈질기게 찾음)
        results = face_model.predict(rotated_roi, conf=0.25, verbose=False)
        
        if len(results[0].boxes) > 0:
            # 얼굴을 찾았다면 좌표를 다시 원래 ROI 좌표로 역변환
            M_inv = cv2.invertAffineTransform(M)
            f_box = results[0].boxes.xyxy[0].cpu().numpy()
            
            # 박스의 네 모서리 변환
            pts = np.array([[f_box[0], f_box[1]], [f_box[2], f_box[1]], 
                            [f_box[2], f_box[3]], [f_box[0], f_box[3]]])
            ones = np.ones(shape=(len(pts), 1))
            pts_ones = np.hstack([pts, ones])
            orig_pts = M_inv.dot(pts_ones.T).T
            
            # 역변환된 좌표의 최소/최대값으로 박스 재구성
            return [np.min(orig_pts[:,0]), np.min(orig_pts[:,1]), 
                    np.max(orig_pts[:,0]), np.max(orig_pts[:,1])]
    return None

cap = cv2.VideoCapture("Test_video/input_video2.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 2. 전신 모델로 ID 고정 (iou와 conf를 높여서 박스 중복 및 ID 폭주 차단)
    # imgsz=640으로 정교하게 잡습니다.
    results = person_model.track(
        frame, 
        persist=True, 
        tracker="botsort.yaml", 
        conf=0.4,         # 문턱을 적절히 조절
        iou=0.5,          # 박스 겹침 방지
        imgsz=640,        # [핵심] 속도를 높여야 ID가 안 바뀝니다
        classes=[0],      # 사람만
        verbose=False
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # 3. 사람 몸 안에서 얼굴 찾기 (회전 로직 적용)
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0: continue
            
            face_box = get_rotated_face(roi, face_model)

            if face_box:
                fx1, fy1, fx2, fy2 = map(int, face_box)
                # 얼굴에 초록색 박스와 몸의 ID 표시
                cv2.rectangle(frame, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person ID: {track_id}", (x1+fx1, y1+fy1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # 얼굴을 못 찾아도 몸의 ID를 파란색으로 표시 (ID 유지 확인)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(frame, f"ID: {track_id} (Searching...)", (x1, y1+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.namedWindow('Nemo Final Hybrid Tracker', cv2.WINDOW_NORMAL)
    cv2.imshow("Nemo Final Hybrid Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()