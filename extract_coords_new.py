#   - yolo:    YOLO 원본 박스
#   - heatmap: YOLO가 놓친 ID에 대해 히트맵 메모리로 복원한 박스
#   - reverse: 정방향(YOLO+히트맵)으로도 못 잡았는데 역방향에서 잡은 박스

import cv2
import json
import torch
import numpy as np
import os
import sys
from ultralytics import YOLO

VIDEOS = ["video_01", "video_02", "video_03", "video_04", "video_05",
          "video_06", "video_07", "video_08", "video_09", "video_10"]

# 사용법: python extract_coords_new.py [모델이름, 기본 yolov12s-face]
#   결과는 results_reset/<모델이름>/ 에 저장된다 (기존 results/ 는 덮어쓰지 않음)
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "yolov12s-face"
MODEL_PATH = f"models/{MODEL_NAME}.pt"
OUT_DIR    = f"results_reset/{MODEL_NAME}"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector = YOLO(MODEL_PATH).to(device)


def iou(b1, b2): # IoU로 정방향 + 역방향..
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def reset_tracker():
    # predictor=None 으로 밀면 다음 track() 에서 탐지 박스가 크게 줄어드는 문제가 있어(video_02 200프레임: 693 -> 462)
    # predictor 는 유지하고 tracker 상태만 초기화한다.
    try:
        for tr in face_detector.predictor.trackers:
            tr.reset()
    except Exception:
        pass

def overlaps_any(box, existing, thresh=0.3): # 박스들 하나하나와 IoU계산해서 하나라도 임계값 넘으면 True = 있는것임. 
    return any(iou(box, e) >= thresh for e in existing)


class HeatmapEngine:
    """
    핵심 변경점:
      - YOLO 박스는 절대 스무딩/변형하지 않고 원본 그대로 반환.
      - YOLO가 놓친 ID(이전엔 검출됐는데 이번 프레임엔 없음) 중
        히트맵 메모리가 충분히 남아있는 ID만 복원박스로 별도 반환.
    """
    def __init__(self, frame_width, frame_height, grid_scale=0.1):
        self.grid_w = int(frame_width * grid_scale)
        self.grid_h = int(frame_height * grid_scale)
        self.scale = grid_scale
        self.channels = {}          # id -> heatmap (grid_h, grid_w)
        self.last_sizes = {}        # id -> (w, h)
        self.last_centers = {}      # id -> (cx, cy)
        self.base_decay = 0.96
        self.recovery_heat_min = 1.5   # 이 강도 이상 메모리가 남았을 때만 복원
        self.conf_full_face = 0.45     # 풀페이스(메모리 업데이트 자격) 기준

    def reset_memory(self):
        self.channels.clear()
        self.last_sizes.clear()
        self.last_centers.clear()

    def detect_with_recovery(self, yolo_boxes, yolo_ids, yolo_confs):
        """
        Returns:
            raw_boxes:       YOLO 원본 박스 그대로
            recovered_boxes: YOLO가 놓친 ID에 대해 히트맵 메모리로 복원한 박스
        """
        # 1) 모든 채널 decay
        for f_id in list(self.channels.keys()):
            self.channels[f_id] *= self.base_decay

        raw_boxes = []
        seen_ids = set()

        # 2) YOLO 검출 처리 — 박스는 원본 유지, 메모리만 업데이트
        for box, f_id, conf in zip(yolo_boxes, yolo_ids, yolo_confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # 트래킹 연속성: YOLO가 새 ID를 줬으면 히트맵에서 옛 ID 찾기
            matched_id = f_id
            if f_id not in self.channels:
                best_id, best_heat = None, 0.2
                hcx, hcy = int(cx * self.scale), int(cy * self.scale)
                for old_id, hm in self.channels.items():
                    if 0 <= hcy < self.grid_h and 0 <= hcx < self.grid_w:
                        if hm[hcy, hcx] > best_heat:
                            best_heat = hm[hcy, hcx]
                            best_id = old_id
                if best_id is not None:
                    matched_id = best_id

            if matched_id in seen_ids:
                # 같은 ID 로 매칭된 중복 박스: 박스는 버리지 않고(비식별화 누락 방지) 메모리 갱신만 건너뜀
                raw_boxes.append([x1, y1, x2, y2])
                continue
            seen_ids.add(matched_id)

            # 원본 YOLO 박스 그대로 추가
            raw_boxes.append([x1, y1, x2, y2])

            # 메모리 업데이트는 conf 충분히 높을 때만 (가려진 박스로 오염되는 거 방지)
            if conf >= self.conf_full_face:
                self.last_sizes[matched_id] = (w, h)
                self.last_centers[matched_id] = (cx, cy)

            # 히트맵은 매번 업데이트
            if matched_id not in self.channels:
                self.channels[matched_id] = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            gx1 = max(0, int((cx - w / 2) * self.scale))
            gy1 = max(0, int((cy - h / 2) * self.scale))
            gx2 = min(self.grid_w, int((cx + w / 2) * self.scale))
            gy2 = min(self.grid_h, int((cy + h / 2) * self.scale))
            self.channels[matched_id][gy1:gy2, gx1:gx2] += 2.0
            self.channels[matched_id] = np.clip(self.channels[matched_id], 0, 10.0)

        # 3) 히트맵 메모리 복원 — YOLO가 놓친 ID만
        recovered_boxes = []
        for missing_id in list(set(self.channels.keys()) - seen_ids):
            if missing_id not in self.last_sizes:
                continue
            if np.max(self.channels[missing_id]) < self.recovery_heat_min:
                continue
            cx, cy = self.last_centers[missing_id]
            w, h = self.last_sizes[missing_id]
            recovered_boxes.append([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2])

        # 4) 메모리 소멸한 ID 제거
        for f_id in list(set(self.channels.keys()) - seen_ids):
            if np.max(self.channels[f_id]) <= 0.5:
                del self.channels[f_id]
                self.last_centers.pop(f_id, None)
                self.last_sizes.pop(f_id, None)

        return raw_boxes, recovered_boxes


def run_pass(frames, frame_width, frame_height, label=""):
    """
    프레임 시퀀스에 대해 YOLO + 히트맵 엔진을 돌림.
    Returns: 프레임별 (raw_boxes, recovered_boxes) 리스트
    """
    reset_tracker()  # 패스/영상마다 트래커 상태 초기화 (이전 패스 상태가 이어지지 않도록)
    engine = HeatmapEngine(frame_width, frame_height)
    prev_frame = None
    scene_change_threshold = 30.0
    results_per_frame = []

    total = len(frames)
    for idx, frame in enumerate(frames):
        # 장면 전환 감지 -> 메모리 리셋
        if prev_frame is not None:
            sc = cv2.resize(frame, (64, 64))
            sp = cv2.resize(prev_frame, (64, 64))
            if np.mean(cv2.absdiff(sc, sp)) > scene_change_threshold:
                engine.reset_memory()
        prev_frame = frame.copy()

        results = face_detector.track(frame, persist=True, conf=0.30,
                                       imgsz=640, device=device, verbose=False) # 얼굴 검출 + 트래킹 ㅇㅇ 
        cur_boxes, cur_ids, cur_confs = [], [], [] # 얼굴 박스들 좌표, 각 박스에 대응하는 트래킹 ID, 각 박스의 검출 conf
        if results[0].boxes is not None and results[0].boxes.id is not None: # 박스들이 있음 + id들도 있음 
            cur_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            cur_ids   = results[0].boxes.id.int().cpu().tolist()
            cur_confs = results[0].boxes.conf.cpu().tolist()

        raw, recovered = engine.detect_with_recovery(cur_boxes, cur_ids, cur_confs)
        results_per_frame.append((raw, recovered))

        if idx % 100 == 0:
            print(f"    {label} {idx}/{total} 처리 중...")

    return results_per_frame


for video_name in VIDEOS:
    VIDEO_PATH  = f"videos/{video_name}.mp4"
    OUTPUT_JSON = f"{OUT_DIR}/{video_name}_coords_new.json"

    if os.path.exists(OUTPUT_JSON):
        print(f"[{video_name}] 이미 존재, 스킵")
        continue

    # 프레임 일괄 로드
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    total = len(frames)
    print(f"\n{'='*40}\n[{video_name}] 총 {total}프레임")

    # 정방향 패스
    print("  🔍 정방향 패스 시작...")
    forward_results = run_pass(frames, frame_width, frame_height, label="정방향")

    # 역방향 패스
    print("  🔄 역방향 패스 시작...")
    
    reversed_results_rev = run_pass(list(reversed(frames)), frame_width, frame_height, label="역방향")
    reverse_results = list(reversed(reversed_results_rev))  # 원래 순서로 복원

    # 세 소스 통합 (yolo -> heatmap -> reverse 우선순위로 IoU 중복 제거)
    all_coords = {}
    sum_yolo = sum_hm = sum_rev = 0

    for idx in range(total):
        fwd_raw, fwd_recovered = forward_results[idx]
        rev_raw, rev_recovered = reverse_results[idx]

        # 역방향 실제 탐지(raw)만 "reverse", 역방향 히트맵 복원은 아래에서 "reverse_heat"로 분리
        reverse_candidates = rev_raw

        yolo_boxes    = [[float(v) for v in b] for b in fwd_raw]
        heatmap_added = []
        reverse_added = []
        reverse_heat_added = [] # 

        merged = list(yolo_boxes)

        # 히트맵 복원박스: YOLO와 겹치는 건 제외
        for b in fwd_recovered:
            box = [float(v) for v in b]
            if not overlaps_any(box, merged, thresh=0.3):
                heatmap_added.append(box)
                merged.append(box)

        # 역방향: YOLO + 히트맵과 겹치는 건 제외
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
        all_coords[f"frame_{idx:04d}"] = {
            "yolo":    yolo_boxes,
            "heatmap": heatmap_added,
            "reverse": reverse_added,
            "reverse_heat": reverse_heat_added,
            # ablation 용: 중복 제거 전 소스별 원본 박스 (analyze_ablation.py 에서 사용)
            "fwd_raw":      [[float(v) for v in b] for b in fwd_raw],
            "fwd_rec_all":  [[float(v) for v in b] for b in fwd_recovered],
            "rev_raw_all":  [[float(v) for v in b] for b in rev_raw],
            "rev_rec_all":  [[float(v) for v in b] for b in rev_recovered],
        }

        sum_yolo += len(yolo_boxes)
        sum_hm   += len(heatmap_added)
        sum_rev  += len(reverse_added)

        if heatmap_added or reverse_added:
            print(f"    frame_{idx:04d}: yolo={len(yolo_boxes)} "
                  f"+heatmap={len(heatmap_added)} +reverse={len(reverse_added)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_coords, f, indent=2)

    print(f"\n  📊 [{video_name}] 박스 합계 — yolo={sum_yolo}, heatmap={sum_hm}, reverse={sum_rev}")
    print(f"  ✅ [{video_name}] 완료 → {OUTPUT_JSON}")

print("\n🎉 전체 완료! compare_all.py 실행하세요.")