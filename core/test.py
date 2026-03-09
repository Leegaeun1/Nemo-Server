from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO('yolov8n-face.pt')

# 이미지 로드
image = cv2.imread('input3.jpg')

# 객체 감지
results = model(image)

# 첫 번째 결과 (이미지 1개)
r = results[0]

# 모든 박스 순회
for box in results[0].boxes:
    # 좌표 (x1, y1, x2, y2)
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # 중심점 + 크기 (x_center, y_center, width, height)
    cx, cy, w, h = box.xywh[0].tolist()

    # 신뢰도
    confidence = box.conf[0].item()

    # 클래스 ID
    class_id = int(box.cls[0].item())

    # 클래스 이름
    class_name = model.names[class_id]

    print(f"{class_name}: {confidence:.2f} at ({cx:.0f}, {cy:.0f})")

print(model.names)

# 결과 시각화
annotated = results[0].plot()
cv2.imshow('YOLO Detection', annotated) # 이미지 보여줌 
cv2.waitKey(0) # 키 입력을 기다림. 0은 무한 대기임
