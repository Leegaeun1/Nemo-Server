import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-face.pt')

image = 'input5.jpg'

# 1. 'Face Detection'이라는 이름의 창을 먼저 만듭니다.
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)

# 2. 창의 크기를 원하는 대로 조절합니다. (가로, 세로)
cv2.resizeWindow('Face Detection', 800, 1500)
results = model.predict(source=image, conf=0.1, imgsz=1024,stream=True,augment=True) # stream=True는 실시간 처리에 효율적입니다.

for r in results:
    # 모델이 그린 결과 이미지(박스 등 포함)를 가져옵니다.
    frame = r.plot()
    # 조절된 창에 이미지를 띄웁니다.
    cv2.imshow('Face Detection', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()