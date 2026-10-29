import csv
import json
import os
import sys

import cv2
import numpy as np

BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "results_reset/yolov12s-face"
VIDEOS = [f"video_{i:02d}" for i in range(1, 11)]
IOU_THRESH = 0.3


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def load_boxes(label_dir):
    out = {}
    if not os.path.isdir(label_dir):
        return out
    for jf in sorted(os.listdir(label_dir)):
        if jf.endswith(".json"):
            with open(os.path.join(label_dir, jf)) as f:
                lm = json.load(f)
            # 꼭짓점 순서 정규화 (LabelMe 드래그 방향에 따라 뒤집힘)
            out[jf[:-5]] = [[min(s["points"][0][0], s["points"][1][0]), min(s["points"][0][1], s["points"][1][1]),
                             max(s["points"][0][0], s["points"][1][0]), max(s["points"][0][1], s["points"][1][1])]
                            for s in lm["shapes"]]
    return out


rows = []
for v in VIDEOS:
    cap = cv2.VideoCapture(f"videos/{v}.mp4")
    if not cap.isOpened():
        continue
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    gt = load_boxes(f"data/{v}/labels_labelme")
    reg = load_boxes(f"data/{v}/registered_sample")
    gt_frames = {k: b for k, b in gt.items() if b}
    n_faces = sum(len(b) for b in gt_frames.values())
    sizes = [(b[3] - b[1]) / h * 100 for bs in gt_frames.values() for b in bs]

    n_hard = None
    old_p = f"{BASE_DIR}/{v}_coords_old.json"
    if os.path.exists(old_p):
        with open(old_p) as f:
            old = json.load(f)
        n_hard = 0
        for k, bs in gt_frames.items():
            pred = old.get(k, [])
            # compare_all_V3 / analyze_ablation 과 동일: 1:1 greedy 매칭으로 GT 를 하나라도 놓친 프레임
            used = set()
            for p in pred:
                for i, g in enumerate(bs):
                    if i not in used and iou(g, p) >= IOU_THRESH:
                        used.add(i)
                        break
            if len(used) < len(bs):
                n_hard += 1

    rows.append({
        "video": v, "resolution": f"{w}x{h}", "fps": round(fps, 1), "frames": n,
        "duration_s": round(n / fps, 1) if fps else "",
        "gt_frames": len(gt_frames), "gt_faces": n_faces,
        "faces_per_gt_frame": round(n_faces / len(gt_frames), 2) if gt_frames else "",
        "face_height_pct": round(float(np.mean(sizes)), 1) if sizes else "",
        "registered_gt_frames": sum(1 for b in reg.values() if b),
        "hard_frames": n_hard if n_hard is not None else "",
        "hard_ratio_pct": round(100 * n_hard / len(gt_frames), 1) if n_hard is not None and gt_frames else "",
        "scene_note": "",
    })

os.makedirs("results_reset", exist_ok=True)
path = "results_reset/video_stats.csv"
with open(path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"{'video':<10}{'res':>11}{'fps':>6}{'frames':>8}{'GTfr':>6}{'faces':>7}{'f/fr':>6}{'h%':>6}{'regGT':>7}{'hard':>6}{'hard%':>7}")
for r in rows:
    print(f"{r['video']:<10}{r['resolution']:>11}{r['fps']:>6}{r['frames']:>8}{r['gt_frames']:>6}"
          f"{r['gt_faces']:>7}{r['faces_per_gt_frame']:>6}{r['face_height_pct']:>6}"
          f"{r['registered_gt_frames']:>7}{str(r['hard_frames']):>6}{str(r['hard_ratio_pct']):>7}")
print(f"\n저장: {path}  (scene_note 열에 장소/인원/가림 상황을 직접 기입)")
