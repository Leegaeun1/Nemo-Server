import torch # 추가
from ultralytics import YOLO
# GPU가 사용 가능하면 'cuda', 아니면 'cpu'를 자동으로 선택합니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 현재 사용 장치: {device}")

# 모델 로드 시 선택된 장치 적용
face_detector = YOLO('models/yolov11n-face.pt').to(device)