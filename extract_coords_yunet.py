import cv2
import json
import numpy as np
import os
import urllib.request
import torch
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes

from heatmap_engine import HeatmapEngine  # 서버/YOLO 평가와 공용 (중복 ID 박스 보존, 복원 박스 ID 반환)
from types import SimpleNamespace

OUT_DIR = "results_reset/yolov12s-face"   # Table 2: YOLOv12s Baseline/Proposed 와 같은 폴더에 저장
VIDEOS = ["video_01", "video_02", "video_03", "video_04", "video_05",
          "video_06", "video_07", "video_08", "video_09", "video_10"]

CONF_THRESH = 0.3          # YuNet score_threshold (YOLO conf=0.30 과 맞춤)
NMS_THRESH  = 0.3
TOP_K       = 5000
MODEL_PATH  = "models/face_detection_yunet_2023mar.onnx"


# ─── 모델 파일 자동 다운로드 ──────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        url = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
               "face_detection_yunet/face_detection_yunet_2023mar.onnx")
        print("YuNet 모델 다운로드 중...")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print(f"  ✅ 저장 → {MODEL_PATH}")

ensure_model()


# ─── ByteTracker 초기화 헬퍼 ──────────────────────────────────
def make_bytetracker(frame_rate=30):
    args = SimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    return BYTETracker(args, frame_rate=frame_rate)

def reset_bytetracker(frame_rate=30):
    return make_bytetracker(frame_rate)


# ─── YuNet 디텍터 생성 ────────────────────────────────────────
def make_face_detector(frame_width, frame_height):
    """
    영상마다 프레임 크기가 고정이므로, 그 크기에 맞춰 setInputSize 해줌.
    (프레임 크기가 바뀌면 그때마다 setInputSize 다시 호출해야 함)
    """
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH, "", (frame_width, frame_height),
        score_threshold=CONF_THRESH,
        nms_threshold=NMS_THRESH,
        top_k=TOP_K,
    )
    return detector


# ─── YuNet 박스 추출 ──────────────────────────────────────────
def yunet_detect(detector, frame):
    """
    Returns:
        boxes_xyxy: [[x1,y1,x2,y2], ...]  절대 픽셀 좌표
        confs:      [float, ...]
    """
    h, w = frame.shape[:2]
    _, faces = detector.detect(frame)   # faces: (N, 15) or None — BGR 입력 그대로 받음

    boxes_xyxy, confs = [], []
    if faces is not None:
        for f in faces:
            score = float(f[14])
            if score < CONF_THRESH:
                continue
            x1, y1, bw, bh = f[0], f[1], f[2], f[3]
            x2, y2 = x1 + bw, y1 + bh
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w, int(x2)); y2 = min(h, int(y2))
            if x2 > x1 and y2 > y1:
                boxes_xyxy.append([x1, y1, x2, y2])
                confs.append(score)
    return boxes_xyxy, confs


# ─── ByteTracker 업데이트) ─
def make_boxes(boxes_xyxy, confs, orig_shape):
    """orig_shape = (h, w)"""
    if not boxes_xyxy:
        data = torch.zeros((0, 6), dtype=torch.float32)
    else:
        data = torch.tensor(
            [[x1, y1, x2, y2, c, 0.0] for (x1, y1, x2, y2), c in zip(boxes_xyxy, confs)],
            dtype=torch.float32,
        )
    return Boxes(data, orig_shape)


def bytetrack_update(tracker, boxes_xyxy, confs, frame):
    """frame: 실제 이미지 ndarray (frame.shape 아님!)"""
    det_results = make_boxes(boxes_xyxy, confs, frame.shape[:2])
    tracks = tracker.update(det_results, frame)

    tracked_boxes, tracked_ids, tracked_confs = [], [], []
    for t in tracks:
        x1, y1, x2, y2, tid, conf = t[0], t[1], t[2], t[3], t[4], t[5]
        tracked_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        tracked_ids.append(int(tid))
        tracked_confs.append(float(conf))
    return tracked_boxes, tracked_ids, tracked_confs


# ─── 박스 중복 판정 ──────────────────────────────────────────
def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def overlaps_any(box, existing, thresh=0.3):
    return any(iou(box, e) >= thresh for e in existing)


# ─── 단일 패스 실행 ───────────────────────────────────────────
def run_pass(frames, frame_width, frame_height, detector, fps=30, label=""):
    engine  = HeatmapEngine(frame_width, frame_height)
    tracker = make_bytetracker(frame_rate=int(fps))
    prev_frame = None
    scene_change_threshold = 30.0
    results_per_frame = []

    total = len(frames)
    for idx, frame in enumerate(frames):
        if prev_frame is not None:
            sc = cv2.resize(frame, (64, 64))
            sp = cv2.resize(prev_frame, (64, 64))
            if np.mean(cv2.absdiff(sc, sp)) > scene_change_threshold:
                engine.reset_memory()
        prev_frame = frame.copy()

        # YuNet 검출
        mp_boxes, mp_confs = yunet_detect(detector, frame)

        # ByteTrack으로 ID 부여
        det_boxes, det_ids, det_confs = bytetrack_update(
            tracker, mp_boxes, mp_confs, frame)

        # HeatmapEngine
        raw, _, recovered, _ = engine.detect_with_recovery_ids(det_boxes, det_ids, det_confs)
        results_per_frame.append((raw, recovered))

        if idx % 100 == 0:
            print(f"    {label} {idx}/{total} 처리 중...")

    return results_per_frame


# ═══════════════════════════════════════════════════════════════
# 메인 루프
# ═══════════════════════════════════════════════════════════════
for video_name in VIDEOS:
    VIDEO_PATH = f"videos/{video_name}.mp4"
    OUT_ONLY   = f"{OUT_DIR}/{video_name}_coords_yunet_only.json"
    OUT_FULL   = f"{OUT_DIR}/{video_name}_coords_yunet_full.json"

    if os.path.exists(OUT_ONLY) and os.path.exists(OUT_FULL):
        print(f"[{video_name}] 이미 존재, 스킵")
        continue

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    total = len(frames)
    print(f"\n{'='*40}\n[{video_name}] 총 {total}프레임  FPS={fps:.1f}")

    # 디텍터는 영상마다 새로 만들어야 함 (setInputSize가 해상도에 묶여 있음)
    detector = make_face_detector(frame_width, frame_height)

    # ── yunet_only: 순수 YuNet + ByteTrack ──────────────────
    if not os.path.exists(OUT_ONLY):
        print("  🔍 [only] 정방향 패스 시작...")
        only_tracker = make_bytetracker(frame_rate=int(fps))
        only_coords  = {}
        prev_frame   = None

        for idx, frame in enumerate(frames):
            mp_boxes, mp_confs = yunet_detect(detector, frame)
            det_boxes, _, _ = bytetrack_update(
                only_tracker, mp_boxes, mp_confs, frame)

            only_coords[f"frame_{idx:04d}"] = [
                [float(v) for v in b] for b in det_boxes]

            if idx % 100 == 0:
                print(f"    [only] {idx}/{total} 처리 중...")

        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT_ONLY, "w") as f:
            json.dump(only_coords, f, indent=2)
        print(f"  ✅ [only] 저장 → {OUT_ONLY}")

    # ── yunet_full: 히트맵 + 역방향 추가 ────────────────────
    if not os.path.exists(OUT_FULL):
        print("  🔍 [full] 정방향 패스 시작...")
        forward_results = run_pass(
            frames, frame_width, frame_height, detector, fps, label="[full] 정방향")

        print("  🔄 [full] 역방향 패스 시작...")
        reversed_results_rev = run_pass(
            list(reversed(frames)), frame_width, frame_height, detector, fps, label="[full] 역방향")
        reverse_results = list(reversed(reversed_results_rev))

        full_coords = {}
        sum_det = sum_hm = sum_rev = 0

        for idx in range(total):
            fwd_raw, fwd_recovered = forward_results[idx]
            rev_raw, rev_recovered = reverse_results[idx]

            reverse_candidates = rev_raw

            det_boxes     = [[float(v) for v in b] for b in fwd_raw]
            heatmap_added = []
            reverse_added = []
            reverse_heat_added = []
            merged = list(det_boxes)

            for b in fwd_recovered:
                box = [float(v) for v in b]
                if not overlaps_any(box, merged, thresh=0.3):
                    heatmap_added.append(box)
                    merged.append(box)

            for b in reverse_candidates:
                box = [float(v) for v in b]
                if not overlaps_any(box, merged, thresh=0.3):
                    reverse_added.append(box)
                    merged.append(box)

            for b in rev_recovered:
                box = [float(v) for v in b]
                if not overlaps_any(box, merged, thresh=0.3):
                    reverse_heat_added.append(box)
                    merged.append(box)

            full_coords[f"frame_{idx:04d}"] = {
                "det":          det_boxes,
                "heatmap":      heatmap_added,
                "reverse":      reverse_added,
                "reverse_heat": reverse_heat_added,
            }

            sum_det += len(det_boxes)
            sum_hm  += len(heatmap_added)
            sum_rev += len(reverse_added)

            if heatmap_added or reverse_added:
                print(f"    frame_{idx:04d}: det={len(det_boxes)} "
                      f"+heatmap={len(heatmap_added)} +reverse={len(reverse_added)}")

        with open(OUT_FULL, "w") as f:
            json.dump(full_coords, f, indent=2)
        print(f"\n  📊 [{video_name}] det={sum_det}, heatmap={sum_hm}, reverse={sum_rev}")
        print(f"  ✅ [full] 저장 → {OUT_FULL}")

print("\n YuNet 실험 완료!")
print("compare_all_V3.py의 VARIANTS 리스트에 yunet_only / yunet_full 추가 후 비교하세요.")