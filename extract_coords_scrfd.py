import cv2
import json
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from ultralytics.trackers.byte_tracker import BYTETracker
from types import SimpleNamespace
import torch
from ultralytics.engine.results import Boxes

VIDEOS = ["video_01", "video_02", "video_03", "video_04", "video_05",
          "video_06", "video_07", "video_08", "video_09", "video_10"]

CONF_THRESH = 0.3          # SCRFD 최소 신뢰도 (YOLO conf=0.30 과 맞춤)
DET_SIZE    = (640, 640)   # SCRFD 입력 해상도


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


# ─── SCRFD 박스 추출 ──────────────────────────────────────────
def scrfd_detect(app, frame):
    """
    SCRFD로 프레임에서 얼굴 박스와 신뢰도를 추출.
    insightface FaceAnalysis.get()은 BGR 입력을 받음.
    Returns:
        boxes_xyxy: [[x1,y1,x2,y2], ...]  절대 픽셀 좌표
        confs:      [float, ...]
    """
    faces = app.get(frame)
    boxes_xyxy, confs = [], []
    for face in faces:
        score = float(face.det_score)
        if score < CONF_THRESH:
            continue
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            boxes_xyxy.append([x1, y1, x2, y2])
            confs.append(score)
    return boxes_xyxy, confs

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

# ─── ByteTracker 업데이트 ─────────────────────────────────────
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


# ─── HeatmapEngine (extract_coords_new.py와 동일) ────────────
def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def overlaps_any(box, existing, thresh=0.3):
    return any(iou(box, e) >= thresh for e in existing)


class HeatmapEngine:
    def __init__(self, frame_width, frame_height, grid_scale=0.1):
        self.grid_w = int(frame_width * grid_scale)
        self.grid_h = int(frame_height * grid_scale)
        self.scale  = grid_scale
        self.channels     = {}
        self.last_sizes   = {}
        self.last_centers = {}
        self.base_decay        = 0.96
        self.recovery_heat_min = 1.5
        self.conf_full_face    = 0.45

    def reset_memory(self):
        self.channels.clear()
        self.last_sizes.clear()
        self.last_centers.clear()

    def detect_with_recovery(self, det_boxes, det_ids, det_confs):
        for f_id in list(self.channels.keys()):
            self.channels[f_id] *= self.base_decay

        raw_boxes = []
        seen_ids  = set()

        for box, f_id, conf in zip(det_boxes, det_ids, det_confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            matched_id = f_id
            if f_id not in self.channels:
                best_id, best_heat = None, 0.2
                hcx = int(cx * self.scale)
                hcy = int(cy * self.scale)
                for old_id, hm in self.channels.items():
                    if 0 <= hcy < self.grid_h and 0 <= hcx < self.grid_w:
                        if hm[hcy, hcx] > best_heat:
                            best_heat = hm[hcy, hcx]
                            best_id   = old_id
                if best_id is not None:
                    matched_id = best_id

            if matched_id in seen_ids:
                continue
            seen_ids.add(matched_id)
            raw_boxes.append([x1, y1, x2, y2])

            if conf >= self.conf_full_face:
                self.last_sizes[matched_id]   = (w, h)
                self.last_centers[matched_id] = (cx, cy)

            if matched_id not in self.channels:
                self.channels[matched_id] = np.zeros(
                    (self.grid_h, self.grid_w), dtype=np.float32)
            gx1 = max(0, int((cx - w/2) * self.scale))
            gy1 = max(0, int((cy - h/2) * self.scale))
            gx2 = min(self.grid_w, int((cx + w/2) * self.scale))
            gy2 = min(self.grid_h, int((cy + h/2) * self.scale))
            self.channels[matched_id][gy1:gy2, gx1:gx2] += 2.0
            self.channels[matched_id] = np.clip(self.channels[matched_id], 0, 10.0)

        recovered_boxes = []
        for missing_id in list(set(self.channels.keys()) - seen_ids):
            if missing_id not in self.last_sizes:
                continue
            if np.max(self.channels[missing_id]) < self.recovery_heat_min:
                continue
            cx, cy = self.last_centers[missing_id]
            w, h   = self.last_sizes[missing_id]
            recovered_boxes.append(
                [cx - w//2, cy - h//2, cx + w//2, cy + h//2])

        for f_id in list(set(self.channels.keys()) - seen_ids):
            if np.max(self.channels[f_id]) <= 0.5:
                del self.channels[f_id]
                self.last_centers.pop(f_id, None)
                self.last_sizes.pop(f_id, None)

        return raw_boxes, recovered_boxes


# ─── 단일 패스 실행 ───────────────────────────────────────────
def run_pass(frames, frame_width, frame_height, scrfd_app, fps=30, label=""):
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
                tracker = reset_bytetracker(frame_rate=int(fps))
        prev_frame = frame.copy()

        # SCRFD 검출
        mp_boxes, mp_confs = scrfd_detect(scrfd_app, frame)

        # ByteTrack으로 ID 부여
        det_boxes, det_ids, det_confs = bytetrack_update(
            tracker, mp_boxes, mp_confs, frame)

        # HeatmapEngine
        raw, recovered = engine.detect_with_recovery(det_boxes, det_ids, det_confs)
        results_per_frame.append((raw, recovered))

        if idx % 100 == 0:
            print(f"    {label} {idx}/{total} 처리 중...")

    return results_per_frame


# ═══════════════════════════════════════════════════════════════
# 메인 루프
# ═══════════════════════════════════════════════════════════════

scrfd_app = FaceAnalysis(
    name='buffalo_sc',
    allowed_modules=['detection'],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
scrfd_app.prepare(ctx_id=0, det_size=DET_SIZE)

for video_name in VIDEOS:
    VIDEO_PATH = f"videos/{video_name}.mp4"
    OUT_ONLY   = f"results/{video_name}_coords_scrfd_only.json"
    OUT_FULL   = f"results/{video_name}_coords_scrfd_full.json"

    if os.path.exists(OUT_ONLY) and os.path.exists(OUT_FULL):
        print(f"[{video_name}] 이미 존재, 스킵")
        continue

    # 프레임 일괄 로드
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

    # ── scrfd_only: 순수 SCRFD + ByteTrack ──────────────────
    if not os.path.exists(OUT_ONLY):
        print("  🔍 [only] 정방향 패스 시작...")
        only_tracker = make_bytetracker(frame_rate=int(fps))
        only_coords  = {}
        prev_frame   = None

        for idx, frame in enumerate(frames):
            if prev_frame is not None:
                sc = cv2.resize(frame, (64, 64))
                sp = cv2.resize(prev_frame, (64, 64))
                if np.mean(cv2.absdiff(sc, sp)) > 30.0:
                    only_tracker = reset_bytetracker(frame_rate=int(fps))
            prev_frame = frame.copy()

            mp_boxes, mp_confs = scrfd_detect(scrfd_app, frame)
            det_boxes, _, _ = bytetrack_update(
                only_tracker, mp_boxes, mp_confs, frame)

            only_coords[f"frame_{idx:04d}"] = [
                [float(v) for v in b] for b in det_boxes]

            if idx % 100 == 0:
                print(f"    [only] {idx}/{total} 처리 중...")

        os.makedirs("results", exist_ok=True)
        with open(OUT_ONLY, "w") as f:
            json.dump(only_coords, f, indent=2)
        print(f"  ✅ [only] 저장 → {OUT_ONLY}")

    # ── scrfd_full: 히트맵 + 역방향 추가 ────────────────────
    if not os.path.exists(OUT_FULL):
        print("  🔍 [full] 정방향 패스 시작...")
        forward_results = run_pass(
            frames, frame_width, frame_height, scrfd_app, fps, label="[full] 정방향")

        print("  🔄 [full] 역방향 패스 시작...")
        reversed_results_rev = run_pass(
            list(reversed(frames)), frame_width, frame_height, scrfd_app, fps, label="[full] 역방향")
        reverse_results = list(reversed(reversed_results_rev))

        full_coords = {}
        sum_det = sum_hm = sum_rev = 0

        for idx in range(total):
            fwd_raw, fwd_recovered = forward_results[idx]
            rev_raw, rev_recovered = reverse_results[idx]

            reverse_candidates = rev_raw + rev_recovered

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
        print(f"\n   [{video_name}] det={sum_det}, heatmap={sum_hm}, reverse={sum_rev}")
        print(f"   [full] 저장 → {OUT_FULL}")

print("\n SCRFD 실험 완료!")
print("compare_all 스크립트에 scrfd_only / scrfd_full 컬럼 추가 후 비교하세요.")