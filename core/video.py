from ultralytics import YOLO
import cv2

# 모델 로드
person_model = YOLO('models/yolo11m.pt') 
face_model = YOLO('models/yolov8n-face.pt')

cap = cv2.VideoCapture("Test_video/input_video2.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 2. 강력한 전신 추적
    # imgsz=1280: 거리 사진처럼 작은 얼굴을 잡기 위해 해상도를 높입니다.
    # track_buffer=150: 가려져도 5초 동안 기억력을 유지합니다.
    results = person_model.track(
        frame, 
        persist=True, 
        tracker="botsort.yaml", 
        conf=0.45,   # 너무 낮은 신뢰도는 버려 ID 폭주 차단
        iou=0.3,     # 박스 겹침 방지 (중복 ID 차단)
        imgsz=1280,  # [중요] 거리 풍경에 필수적인 고해상도 연산
        classes=[0], 
        verbose=False
    )

    # results[0].plot()은 절대 사용 금지 (중복 출력의 주범)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # ROI 추출 (사람 몸의 상단 40% 영역에서만 얼굴 탐색 - 성능 최적화)
            roi_y2 = y1 + int((y2 - y1) * 0.4)
            roi = frame[max(0, y1):roi_y2, max(0, x1):x2]
            
            if roi.size == 0: continue
            
            # 3. 얼굴 탐지 (conf를 낮춰서 측면/흐릿한 얼굴도 포착)
            face_res = face_model.predict(roi, conf=0.2, verbose=False)

            # --- 시각화: 오직 한 명당 하나의 정보만 표시 ---
            if len(face_res[0].boxes) > 0:
                fb = face_res[0].boxes.xyxy[0].cpu().numpy()
                fx1, fy1, fx2, fy2 = map(int, fb)
                # 얼굴에 초록 박스
                cv2.rectangle(frame, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {track_id}", (x1+fx1, y1+fy1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # 얼굴을 놓쳐도 몸체 위치에 ID 표시하여 추적 유지 확인
                cv2.putText(frame, f"ID:{track_id}", (x1, y1+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 5. 시각화
    cv2.namedWindow('Hybrid Face-Person Tracking', cv2.WINDOW_NORMAL)

    cv2.imshow("Hybrid Face-Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()