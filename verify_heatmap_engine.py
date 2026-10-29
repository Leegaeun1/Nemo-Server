import json
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from heatmap_engine import HeatmapEngine

MODEL_PATH = "models/yolov12s-face.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector = YOLO(MODEL_PATH).to(device)


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def overlaps_any(box, existing, thresh=0.3):
    return any(iou(box, e) >= thresh for e in existing)


def same_boxes(a, b, thr=0.9):
    if len(a) != len(b):
        return False
    return all(max((iou(x, y) for y in b), default=0) >= thr for x in a)


def verify(video_name):
    cap = cv2.VideoCapture(f"videos/{video_name}.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with open(f"results/{video_name}_coords_new.json") as f:
        saved = json.load(f)

    face_detector.predictor = None
    engine = HeatmapEngine(w, h)
    prev, idx, n_diff = None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if prev is not None:
            sc = cv2.resize(frame, (64, 64))
            sp = cv2.resize(prev, (64, 64))
            if np.mean(cv2.absdiff(sc, sp)) > 30.0:
                engine.reset_memory()
        prev = frame.copy()

        r = face_detector.track(frame, persist=True, conf=0.30, imgsz=640,
                                device=device, verbose=False)
        boxes, ids, confs = [], [], []
        if r[0].boxes is not None and r[0].boxes.id is not None:
            boxes = r[0].boxes.xyxy.cpu().numpy().tolist()
            ids = r[0].boxes.id.int().cpu().tolist()
            confs = r[0].boxes.conf.cpu().tolist()
        raw, _, rec, _ = engine.detect_with_recovery_ids(boxes, ids, confs)

        yolo_boxes = [[float(v) for v in b] for b in raw]
        merged = list(yolo_boxes)
        heat = []
        for b in rec:
            b = [float(v) for v in b]
            if not overlaps_any(b, merged):
                heat.append(b)
                merged.append(b)

        s = saved[f"frame_{idx:04d}"]
        if not same_boxes(yolo_boxes, s["yolo"]) or not same_boxes(heat, s["heatmap"]):
            n_diff += 1
            if n_diff <= 5:
                print(f"  [{video_name}] frame {idx} 불일치")
        idx += 1
    cap.release()
    print(f"[{video_name}] {idx} 프레임 중 불일치 {n_diff} 프레임")
    return n_diff == 0


if __name__ == "__main__":
    videos = sys.argv[1:] or ["video_01"]
    print("ALL MATCH" if all([verify(v) for v in videos]) else "MISMATCH 있음")
