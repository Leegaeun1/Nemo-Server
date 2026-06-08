import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
import os

VIDEOS = ["video_01", "video_02", "video_03", "video_04", "video_05", "video_06", "video_07", "video_08", "video_09", "video_10"]

for video_name in VIDEOS:
    VIDEO_PATH  = f"videos/{video_name}.mp4"
    OUTPUT_JSON = f"results/{video_name}_coords_old.json"
    MODEL_PATH = "models/yolov10n-face.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_detector = YOLO(MODEL_PATH).to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"총 프레임: {total}")

    all_coords = {}
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = face_detector.track(frame, persist=True, conf=0.3,
                                    imgsz=640, device=device, verbose=False)

        boxes = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(float, box)
                boxes.append([x1, y1, x2, y2])

        key = f"frame_{frame_idx:04d}"
        all_coords[key] = boxes

        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} 처리 중...")

        frame_idx += 1

    cap.release()


    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_coords, f, indent=2)

print(f"✅ 완료! {OUTPUT_JSON} 저장됨")