import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from heatmap_engine import HeatmapEngine

VIDEOS = ["video_01", "video_02", "video_03", "video_04", "video_05",
          "video_06", "video_07", "video_08", "video_09", "video_10"]
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "yolov12s-face"
SWEEP_DIR = f"results_sweep/{MODEL_NAME}"
IOU_THRESH = 0.3
MOSAIC_RADIUS_FACTOR = 0.6

DEFAULT = {"decay": 0.96, "rec_min": 1.5, "id_match": 0.2, "conf_full": 0.45,
           "beta": 2.0, "hmax": 10.0, "del_min": 0.5, "grid": 0.1}
SWEEP = {
    "decay":     [0.90, 0.93, 0.96, 0.98, 0.99],
    "rec_min":   [0.5, 1.0, 1.5, 2.0, 3.0],
    "id_match":  [0.05, 0.1, 0.2, 0.5],
    "conf_full": [0.30, 0.40, 0.45, 0.60],
    # beta/hmax 는 다른 임계값과 별개로 비율(beta/hmax)이 의미를 가지므로 따로 훑는다
    "beta":      [1.0, 2.0, 5.0, 10.0],
    "hmax":      [5.0, 10.0, 20.0],
    "del_min":   [0.1, 0.5, 1.0],
    "grid":      [0.05, 0.1, 0.2],
}


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def overlaps_any(box, existing, thresh=IOU_THRESH):
    return any(iou(box, e) >= thresh for e in existing)


def circle_coverage(gt_box, pred_boxes, factor=MOSAIC_RADIUS_FACTOR):
    # compare_all_V3.gt_circle_coverage 와 동일
    gx1, gy1, gx2, gy2 = gt_box
    gw_full, gh_full = gx2 - gx1, gy2 - gy1
    if gw_full <= 0 or gh_full <= 0:
        return 0.0
    scale = 1.0
    if gw_full * gh_full > 10000:
        scale = (10000.0 / (gw_full * gh_full)) ** 0.5
    gw, gh = max(1, int(gw_full * scale)), max(1, int(gh_full * scale))
    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
    xs_o, ys_o = xs / scale + gx1, ys / scale + gy1
    mask = np.zeros((gh, gw), dtype=bool)
    for p in pred_boxes:
        cx, cy = (p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0
        r = max(p[2] - p[0], p[3] - p[1]) * factor
        if r > 0:
            mask |= ((xs_o - cx) ** 2 + (ys_o - cy) ** 2 <= r * r)
    return float(mask.sum()) / (gw * gh)


def greedy_tp(gt, pred):
    # compare_all_V3.precision_recall 의 TP 계산과 동일 (pred 순서대로 미매칭 GT 1:1)
    matched = set()
    for p in pred:
        for i, g in enumerate(gt):
            if i not in matched and iou(g, p) >= IOU_THRESH:
                matched.add(i)
                break
    return len(matched)


def run_engine(seq, W, H, cfg):
    eng = HeatmapEngine(W, H, grid_scale=cfg["grid"])
    eng.accumulate = cfg["beta"]
    eng.heat_max = cfg["hmax"]
    eng.delete_heat = cfg["del_min"]
    eng.base_decay = cfg["decay"]
    eng.recovery_heat_min = cfg["rec_min"]
    eng.id_match_heat_min = cfg["id_match"]
    eng.conf_full_face = cfg["conf_full"]
    out = []
    for t in seq:
        if t["reset"]:
            eng.reset_memory()
        raw, _, rec, _ = eng.detect_with_recovery_ids(t["boxes"], t["ids"], t["confs"])
        out.append((raw, rec))
    return out


def load_gt(video_name):
    d = f"data/{video_name}/labels_labelme"
    gt = {}
    for jf in sorted(os.listdir(d)):
        if jf.endswith(".json"):
            with open(os.path.join(d, jf)) as f:
                lm = json.load(f)
            gt[jf[:-5]] = [[min(s["points"][0][0], s["points"][1][0]), min(s["points"][0][1], s["points"][1][1]),
                            max(s["points"][0][0], s["points"][1][0]), max(s["points"][0][1], s["points"][1][1])]
                           for s in lm["shapes"]]
    return gt


def eval_video(args):
    video_name, configs = args
    with open(f"{SWEEP_DIR}/{video_name}_tracks.json") as f:
        tr = json.load(f)
    W, H = tr["width"], tr["height"]
    gt = load_gt(video_name)
    n = len(tr["fwd"])
    res = {}
    for name, cfg in configs:
        fwd = run_engine(tr["fwd"], W, H, cfg)
        rev = list(reversed(run_engine(list(reversed(tr["rev"])), W, H, cfg)))
        s = dict(frames=0, gt=0, tp=0, rec_fr=0.0, cov=0.0, exposed_fr=0, fp_extra=0)
        for idx in range(n):
            key = f"frame_{idx:04d}"
            if key not in gt:
                continue
            g = gt[key]
            fwd_raw, fwd_rec = fwd[idx]
            rev_raw, rev_rec = rev[idx]
            merged = [[float(v) for v in b] for b in fwd_raw]
            extra = []
            for b in fwd_rec:                       # heatmap
                b = [float(v) for v in b]
                if not overlaps_any(b, merged):
                    merged.append(b); extra.append(b)
            for b in rev_raw:                       # reverse
                b = [float(v) for v in b]
                if not overlaps_any(b, merged):
                    merged.append(b); extra.append(b)
            for b in rev_rec:                       # reverse_heat
                b = [float(v) for v in b]
                if not overlaps_any(b, merged):
                    merged.append(b); extra.append(b)
            tp = greedy_tp(g, merged)
            s["frames"] += 1
            s["gt"] += len(g)
            s["tp"] += tp
            s["rec_fr"] += tp / len(g) if g else 1.0
            s["cov"] += float(np.mean([circle_coverage(x, merged) for x in g])) if g else 1.0
            s["exposed_fr"] += int(tp < len(g))
            s["fp_extra"] += sum(1 for b in extra if not any(iou(b, x) >= IOU_THRESH for x in g))
        res[name] = s
    return res


def main():
    configs, seen = [], set()
    for param, values in SWEEP.items():
        for v in values:
            cfg = dict(DEFAULT); cfg[param] = v
            name = "default" if cfg == DEFAULT else f"{param}={v}"
            if name not in seen:
                seen.add(name)
                configs.append((name, cfg))

    with Pool(min(len(VIDEOS), os.cpu_count() or 1)) as p:
        per_video = p.map(eval_video, [(v, configs) for v in VIDEOS])

    rows = []
    for name, cfg in configs:
        tot = {k: sum(r[name][k] for r in per_video) for k in per_video[0][name]}
        rows.append({
            "config": name,
            "decay": cfg["decay"], "rec_min": cfg["rec_min"],
            "id_match": cfg["id_match"], "conf_full": cfg["conf_full"],
            "beta": cfg["beta"], "hmax": cfg["hmax"],
            "del_min": cfg["del_min"], "grid": cfg["grid"],
            "recall": round(tot["rec_fr"] / tot["frames"], 4),      # 프레임별 Recall 의 평균 (본문 정의)
            "recall_gt": round(tot["tp"] / tot["gt"], 4),           # GT 얼굴 단위 합산 (참고)
            "covC": round(tot["cov"] / tot["frames"], 4),
            "exposed_fr": round(tot["exposed_fr"] / tot["frames"], 4),
            "fp_per_frame": round(tot["fp_extra"] / tot["frames"], 4),
        })
        print(rows[-1])
    out = f"{SWEEP_DIR}/sensitivity.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"저장 → {out}")


if __name__ == "__main__":
    main()
