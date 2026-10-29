import json
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

VIDEOS = ["video_01", "video_02", "video_03", "video_04", "video_05",
          "video_06", "video_07", "video_08", "video_09", "video_10"]
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "yolov12s-face"
OUT_DIR = f"results_sweep/{MODEL_NAME}"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector = YOLO(f"models/{MODEL_NAME}.pt").to(device)


def reset_tracker():
    try:
        for tr in face_detector.predictor.trackers:
            tr.reset()
    except Exception:
        pass


def run_pass(frames, label):
    reset_tracker()
    out, prev = [], None
    for idx, frame in enumerate(frames):
        reset = False
        if prev is not None:
            sc = cv2.resize(frame, (64, 64))
            sp = cv2.resize(prev, (64, 64))
            reset = bool(np.mean(cv2.absdiff(sc, sp)) > 30.0)
        prev = frame.copy()

        r = face_detector.track(frame, persist=True, conf=0.30, imgsz=640,
                                device=device, verbose=False)
        boxes, ids, confs = [], [], []
        if r[0].boxes is not None and r[0].boxes.id is not None:
            boxes = r[0].boxes.xyxy.cpu().numpy().tolist()
            ids = r[0].boxes.id.int().cpu().tolist()
            confs = r[0].boxes.conf.cpu().tolist()
        out.append({"boxes": boxes, "ids": ids, "confs": confs, "reset": reset})
        if idx % 200 == 0:
            print(f"    {label} {idx}/{len(frames)}")
    return out


os.makedirs(OUT_DIR, exist_ok=True)
for video_name in VIDEOS:
    path = f"{OUT_DIR}/{video_name}_tracks.json"
    if os.path.exists(path):
        print(f"[{video_name}] 이미 존재, 스킵")
        continue
    cap = cv2.VideoCapture(f"videos/{video_name}.mp4")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"[{video_name}] {len(frames)}프레임")

    fwd = run_pass(frames, "정방향")
    rev = list(reversed(run_pass(list(reversed(frames)), "역방향")))  # 원래 프레임 순서로 복원
    # 주의: rev 의 "reset" 은 역방향 진행 기준이므로, sweep 에서는 역순으로 다시 돌려야 한다 (rev 는 원래 순서로 저장됨)
    with open(path, "w") as f:
        json.dump({"width": W, "height": H, "fwd": fwd, "rev": rev}, f)
    print(f"  저장 → {path}")
