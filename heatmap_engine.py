import numpy as np


class HeatmapEngine:
    def __init__(self, frame_width, frame_height, grid_scale=0.1):
        self.grid_w = int(frame_width * grid_scale)
        self.grid_h = int(frame_height * grid_scale)
        self.scale = grid_scale
        self.channels = {}          # id -> heatmap (grid_h, grid_w)
        self.last_sizes = {}        # id -> (w, h)
        self.last_centers = {}      # id -> (cx, cy)
        self.base_decay = 0.96      # lambda
        self.recovery_heat_min = 1.5
        self.conf_full_face = 0.45  # 메모리 업데이트 자격 기준
        self.id_match_heat_min = 0.2  # tau_H: 새 ID를 기존 ID에 대응시키는 최소 heat
        self.accumulate = 2.0       # beta
        self.heat_max = 10.0        # H_max
        self.delete_heat = 0.5      # tau_del

    def reset_memory(self):
        self.channels.clear()
        self.last_sizes.clear()
        self.last_centers.clear()

    def detect_with_recovery_ids(self, yolo_boxes, yolo_ids, yolo_confs):
        """
        Returns:
            raw_boxes, raw_ids : YOLO 원본 박스와 (히트맵으로 보정된) ID
            rec_boxes, rec_ids : 히트맵 메모리로 복원한 박스와 해당 ID
        """
        # 1) 모든 채널 decay
        for f_id in list(self.channels.keys()):
            self.channels[f_id] *= self.base_decay

        raw_boxes, raw_ids = [], []
        seen_ids = set()

        # 2) YOLO 검출 처리 — 박스는 원본 유지, 메모리만 업데이트
        for box, f_id, conf in zip(yolo_boxes, yolo_ids, yolo_confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # 트래킹 연속성: YOLO가 새 ID를 줬으면 히트맵에서 옛 ID 찾기
            matched_id = f_id
            if f_id not in self.channels:
                best_id, best_heat = None, self.id_match_heat_min
                hcx, hcy = int(cx * self.scale), int(cy * self.scale)
                for old_id, hm in self.channels.items():
                    if 0 <= hcy < self.grid_h and 0 <= hcx < self.grid_w:
                        if hm[hcy, hcx] > best_heat:
                            best_heat = hm[hcy, hcx]
                            best_id = old_id
                if best_id is not None:
                    matched_id = best_id

            if matched_id in seen_ids:
                # 같은 ID 로 매칭된 중복 박스: 박스는 버리지 않고(비식별화 누락 방지)
                # 원래 트래커 ID 로 반환하며 메모리 갱신만 건너뜀
                raw_boxes.append([x1, y1, x2, y2])
                raw_ids.append(f_id)
                continue
            seen_ids.add(matched_id)

            raw_boxes.append([x1, y1, x2, y2])
            raw_ids.append(matched_id)

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
            self.channels[matched_id][gy1:gy2, gx1:gx2] += self.accumulate
            self.channels[matched_id] = np.clip(self.channels[matched_id], 0, self.heat_max)

        # 3) 히트맵 메모리 복원 — YOLO가 놓친 ID만
        rec_boxes, rec_ids = [], []
        for missing_id in list(set(self.channels.keys()) - seen_ids):
            if missing_id not in self.last_sizes:
                continue
            if np.max(self.channels[missing_id]) < self.recovery_heat_min:
                continue
            cx, cy = self.last_centers[missing_id]
            w, h = self.last_sizes[missing_id]
            rec_boxes.append([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2])
            rec_ids.append(missing_id)

        # 4) 메모리 소멸한 ID 제거
        for f_id in list(set(self.channels.keys()) - seen_ids):
            if np.max(self.channels[f_id]) <= self.delete_heat:
                del self.channels[f_id]
                self.last_centers.pop(f_id, None)
                self.last_sizes.pop(f_id, None)

        return raw_boxes, raw_ids, rec_boxes, rec_ids
