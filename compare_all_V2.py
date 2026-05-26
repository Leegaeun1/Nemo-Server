# 추가 지표:
#   - Coverage (bbox): GT 영역 중 예측 박스 union으로 덮인 비율
#   - Coverage (circle): GT 영역 중 실제 모자이크 원(radius=0.6*max(w,h)) union으로 덮인 비율
#     -> 이게 production 시각 결과에 진짜로 가까움
#   - Well-Covered ≥95% (bbox & circle 두 버전)
# ALL frames vs HARD frames(OLD가 놓친 게 있는 프레임) 분리해서 출력

import json
import os
import csv
import numpy as np

VIDEOS = ["video_01", "video_02", "video_03", "video_04", "video_05",
          "video_06", "video_07", "video_08", "video_09", "video_10"]

IOU_THRESH = 0.3
WELL_COVERED_THR = 0.95
MOSAIC_RADIUS_FACTOR = 0.6   # apply_mosaic 에서 radius = 0.6 * max(w, h)


# ─── 기본 유틸 ────────────────────────────────────────────────
def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def flatten(entry):
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        return entry.get("yolo", []) + entry.get("heatmap", []) + entry.get("reverse", [])
    return []


def best_iou_per_gt(gt, pred):
    if not pred:
        return [0.0] * len(gt)
    return [max(iou(g, p) for p in pred) for g in gt]


def precision_recall(gt, pred, thr=IOU_THRESH):
    if not pred or not gt:
        return 0.0, 0.0
    tp = 0
    matched = set()
    for p in pred:
        for i, g in enumerate(gt):
            if i not in matched and iou(g, p) >= thr:
                tp += 1
                matched.add(i)
                break
    return tp / len(pred), tp / len(gt)


def miss_rate(gt, pred, thr=IOU_THRESH):
    if not gt:
        return 0.0
    if not pred:
        return 1.0
    return sum(1 for g in gt
               if max((iou(g, p) for p in pred), default=0) < thr) / len(gt)


# ─── Coverage (bbox) ───────────────────────────────────────────
def gt_bbox_coverage(gt_box, pred_boxes):
    gx1, gy1, gx2, gy2 = gt_box
    gw_full = gx2 - gx1
    gh_full = gy2 - gy1
    if gw_full <= 0 or gh_full <= 0:
        return 0.0
    scale = 1.0
    if gw_full * gh_full > 10000:
        scale = (10000.0 / (gw_full * gh_full)) ** 0.5
    gw = max(1, int(gw_full * scale))
    gh = max(1, int(gh_full * scale))
    mask = np.zeros((gh, gw), dtype=bool)
    for p in pred_boxes:
        ix1 = max(int((p[0] - gx1) * scale), 0)
        iy1 = max(int((p[1] - gy1) * scale), 0)
        ix2 = min(int((p[2] - gx1) * scale), gw)
        iy2 = min(int((p[3] - gy1) * scale), gh)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = True
    return float(mask.sum()) / (gw * gh)


# ─── Coverage (circle = 실제 모자이크 모양) ──────────────────────
def gt_circle_coverage(gt_box, pred_boxes, factor=MOSAIC_RADIUS_FACTOR):
    gx1, gy1, gx2, gy2 = gt_box
    gw_full = gx2 - gx1
    gh_full = gy2 - gy1
    if gw_full <= 0 or gh_full <= 0:
        return 0.0

    scale = 1.0
    if gw_full * gh_full > 10000:
        scale = (10000.0 / (gw_full * gh_full)) ** 0.5
    gw = max(1, int(gw_full * scale))
    gh = max(1, int(gh_full * scale))

    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
    xs_orig = xs / scale + gx1
    ys_orig = ys / scale + gy1

    mask = np.zeros((gh, gw), dtype=bool)
    for p in pred_boxes:
        px1, py1, px2, py2 = p
        cx = (px1 + px2) / 2.0
        cy = (py1 + py2) / 2.0
        w  = px2 - px1
        h  = py2 - py1
        r  = max(w, h) * factor
        if r <= 0:
            continue
        d2 = (xs_orig - cx) ** 2 + (ys_orig - cy) ** 2
        mask |= (d2 <= r * r)
    return float(mask.sum()) / (gw * gh)


def frame_coverage(gt_boxes, pred_boxes, mode="bbox"):
    if not gt_boxes:
        return 1.0
    fn = gt_circle_coverage if mode == "circle" else gt_bbox_coverage
    return float(np.mean([fn(g, pred_boxes) for g in gt_boxes]))


def well_covered_count(gt_boxes, pred_boxes, thresh=WELL_COVERED_THR, mode="bbox"):
    if not gt_boxes:
        return 0
    fn = gt_circle_coverage if mode == "circle" else gt_bbox_coverage
    return sum(1 for g in gt_boxes if fn(g, pred_boxes) >= thresh)


# ─── NEW 박스 소스별 GT 매칭 ────────────────────────────────────
def attribute_gt_match(gt, new_entry, thr=IOU_THRESH):
    if isinstance(new_entry, list):
        return {"yolo": 0, "heatmap": 0, "reverse": 0, "miss": len(gt)}
    counts = {"yolo": 0, "heatmap": 0, "reverse": 0}
    matched_gt = set()
    for src in ("yolo", "heatmap", "reverse", "reverse_heat"):
        for pred in new_entry.get(src, []):
            best_v, best_i = 0, -1
            for i, g in enumerate(gt):
                if i in matched_gt:
                    continue
                v = iou(g, pred)
                if v >= thr and v > best_v:
                    best_v, best_i = v, i
            if best_i >= 0:
                counts[src] += 1
                matched_gt.add(best_i)
    counts["miss"] = len(gt) - sum(counts.values())
    return counts


# ─── 한 변형의 지표 ────────────────────────────────────────────
def compute_metrics(gt, pred):
    ious = best_iou_per_gt(gt, pred)
    avg_iou = float(np.mean(ious)) if ious else 0.0
    prec, rec = precision_recall(gt, pred)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return {
        "count": len(pred), "iou": avg_iou, "prec": prec, "rec": rec, "f1": f1,
        "miss":       miss_rate(gt, pred),
        "cov_bbox":   frame_coverage(gt, pred, mode="bbox"),
        "cov_circle": frame_coverage(gt, pred, mode="circle"),
        "well_bbox":   well_covered_count(gt, pred, mode="bbox"),
        "well_circle": well_covered_count(gt, pred, mode="circle"),
    }


# ─── 요약 출력 ────────────────────────────────────────────────
def print_summary(rows):
    if not rows:
        print("  (해당 프레임 없음)")
        return

    def avg(k):   return float(np.mean([r[k] for r in rows]))
    def total(k): return int(sum(r[k] for r in rows))

    variants = ["old", "new"]
    cols = ["OLD", "NEW"]

    print(f"  {'':<22}" + "".join(f"{c:>10}" for c in cols))
    print("  " + "-" * (22 + 10 * len(cols)))

    def row(label, key_pat):
        vals = [avg(key_pat.format(v)) for v in variants]
        print(f"  {label:<22}" + "".join(f"{v:>10.3f}" for v in vals))

    row("Precision",        "{}_prec")
    row("Recall",           "{}_rec")
    row("F1",               "{}_f1")
    row("IoU",              "{}_iou")
    row("Miss Rate",        "{}_miss")
    row("Coverage (bbox)",  "{}_cov_bbox")
    row("Coverage (circle)","{}_cov_circle")

    gt_t = total("gt_count")
    if gt_t > 0:
        for tag, fmt_key in [("bbox", "well_bbox"), ("circle", "well_circle")]:
            wells = [total(f"{v}_{fmt_key}") for v in variants]
            pcts  = [100 * w / gt_t for w in wells]
            print(f"  {'Well-Cov ≥95% ('+tag+')':<22}" +
                  "".join(f"{v:>9.1f}%" for v in pcts))


# ═══════════════════════════════════════════════════════════════
# 메인 루프
# ═══════════════════════════════════════════════════════════════
overall = []

for video_name in VIDEOS:
    GT_DIR     = f"data/{video_name}/labels_labelme"
    OLD_JSON   = f"results/{video_name}_coords_old.json"
    NEW_JSON   = f"results/{video_name}_coords_new.json"
    OUTPUT_CSV = f"results/comparison_{video_name}.csv"

    if not (os.path.exists(OLD_JSON) and os.path.exists(NEW_JSON) and os.path.isdir(GT_DIR)):
        print(f"[{video_name}] 필수 파일 없음, 스킵")
        continue

    with open(OLD_JSON) as f: old_coords = json.load(f)
    with open(NEW_JSON) as f: new_coords = json.load(f)

    gt_coords = {}
    for jf in sorted(os.listdir(GT_DIR)):
        if not jf.endswith(".json"): continue
        with open(os.path.join(GT_DIR, jf)) as f:
            lm = json.load(f)
        key = jf.replace(".json", "")
        gt_coords[key] = [
            [s["points"][0][0], s["points"][0][1],
             s["points"][1][0], s["points"][1][1]]
            for s in lm["shapes"]
        ]

    rows = []
    src_total = {"yolo": 0, "heatmap": 0, "reverse": 0, "miss": 0}
    print(f"\n📹 {video_name} 처리 중 (circle coverage 계산이 좀 느릴 수 있음)...")

    for key in sorted(gt_coords.keys()):
        gt      = gt_coords[key]
        old_raw = old_coords.get(key, [])
        new_raw = new_coords.get(key, [])

        old = flatten(old_raw)
        new = flatten(new_raw)

        m_old = compute_metrics(gt, old)
        m_new = compute_metrics(gt, new)

        src = attribute_gt_match(gt, new_raw)
        for k in src_total:
            src_total[k] += src[k]

        row = {"frame": key, "gt_count": len(gt)}
        for tag, m in [("old", m_old), ("new", m_new)]:
            row[f"{tag}_count"]       = m["count"]
            row[f"{tag}_iou"]         = round(m["iou"], 3)
            row[f"{tag}_prec"]        = round(m["prec"], 3)
            row[f"{tag}_rec"]         = round(m["rec"], 3)
            row[f"{tag}_f1"]          = round(m["f1"], 3)
            row[f"{tag}_miss"]        = round(m["miss"], 3)
            row[f"{tag}_cov_bbox"]    = round(m["cov_bbox"], 3)
            row[f"{tag}_cov_circle"]  = round(m["cov_circle"], 3)
            row[f"{tag}_well_bbox"]   = m["well_bbox"]
            row[f"{tag}_well_circle"] = m["well_circle"]
        row.update({
            "matched_by_yolo":    src["yolo"],
            "matched_by_heatmap": src["heatmap"],
            "matched_by_reverse": src["reverse"],
            "missed":             src["miss"],
        })
        rows.append(row)

    if not rows:
        continue

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*60}\n📹 {video_name}\n{'='*60}")
    print(f"\n[ALL FRAMES — n={len(rows)}]")
    print_summary(rows)

    hard_rows = [r for r in rows if r["old_miss"] > 0]
    print(f"\n[HARD FRAMES — OLD가 놓친 GT 있는 프레임만, n={len(hard_rows)}/{len(rows)}]")
    print_summary(hard_rows)

    gt_total = sum(r["gt_count"] for r in rows)
    if gt_total > 0:
        print(f"\n[NEW 박스 소스별 GT 매칭 — 전체 GT {gt_total}개]")
        for k, label in [("yolo", "YOLO"), ("heatmap", "히트맵"),
                         ("reverse", "역방향"), ("miss", "미검출")]:
            print(f"  {label:<8} {src_total[k]:5d}  ({100*src_total[k]/gt_total:5.1f}%)")

    print(f"\n📊 저장: {OUTPUT_CSV}")

    def avgr(rs, k): return float(np.mean([r[k] for r in rs])) if rs else 0.0
    def wellpct(rs, k):
        gt_t = sum(r["gt_count"] for r in rs)
        return 100 * sum(r[k] for r in rs) / gt_t if gt_t else 0.0

    overall.append({
        "video": video_name, "n_all": len(rows), "n_hard": len(hard_rows),
        **{f"all_{t}_{k}": avgr(rows, f"{t}_{k}")
           for t in ["old", "new"]
           for k in ["f1", "cov_bbox", "cov_circle"]},
        **{f"all_{t}_well_circle": wellpct(rows, f"{t}_well_circle")
           for t in ["old", "new"]},
        **{f"hard_{t}_{k}": avgr(hard_rows, f"{t}_{k}")
           for t in ["old", "new"]
           for k in ["rec", "cov_bbox", "cov_circle"]},
    })


# ═══════════════════════════════════════════════════════════════
# 전체 종합
# ═══════════════════════════════════════════════════════════════
if overall:
    print(f"\n\n{'='*80}")
    print("🏁 ALL FRAMES — F1 / Coverage(bbox) / Coverage(circle) / Well-Covered(circle)")
    print('='*80)
    hdr = f"{'video':<12} {'OLD_F1':>8} {'NEW_F1':>8} {'OLD_CovB':>9} {'NEW_CovB':>9} {'OLD_CovC':>9} {'NEW_CovC':>9} {'OLD_WC%':>8} {'NEW_WC%':>8}"
    print(hdr)
    print('-'*len(hdr))
    for s in overall:
        print(f"{s['video']:<12}"
              f" {s['all_old_f1']:>8.3f} {s['all_new_f1']:>8.3f}"
              f" {s['all_old_cov_bbox']:>9.3f} {s['all_new_cov_bbox']:>9.3f}"
              f" {s['all_old_cov_circle']:>9.3f} {s['all_new_cov_circle']:>9.3f}"
              f" {s['all_old_well_circle']:>7.1f}% {s['all_new_well_circle']:>7.1f}%")

    print(f"\n{'='*80}")
    print("🎯 HARD FRAMES — Recall / Coverage(bbox) / Coverage(circle)")
    print("(NEW가 가려진 얼굴을 회수했는지 + 실제 모자이크가 얼마나 덮는지)")
    print('='*80)
    hdr = f"{'video':<12} {'n_hard':>7} {'OLD_Rec':>8} {'NEW_Rec':>8} {'OLD_CovB':>9} {'NEW_CovB':>9} {'OLD_CovC':>9} {'NEW_CovC':>9}"
    print(hdr)
    print('-'*len(hdr))
    for s in overall:
        if s["n_hard"] == 0:
            print(f"{s['video']:<12} {0:>7}  (OLD가 놓친 프레임 없음)")
            continue
        print(f"{s['video']:<12} {s['n_hard']:>7d}"
              f" {s['hard_old_rec']:>8.3f} {s['hard_new_rec']:>8.3f}"
              f" {s['hard_old_cov_bbox']:>9.3f} {s['hard_new_cov_bbox']:>9.3f}"
              f" {s['hard_old_cov_circle']:>9.3f} {s['hard_new_cov_circle']:>9.3f}")