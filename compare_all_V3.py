import json
import os
import csv
import numpy as np

VIDEOS = ["video_01", "video_02", "video_03", "video_04", "video_05",
          "video_06", "video_07", "video_08", "video_09", "video_10"]

IOU_THRESH = 0.3
WELL_COVERED_THR = 0.95
MOSAIC_RADIUS_FACTOR = 0.6   # apply_mosaic 에서 radius = 0.6 * max(w, h)

VARIANTS = [
    {"key": "old",         "file": "coords_old.json",           "kind": "list", "label": "YOLO-OLD"},
    {"key": "new",         "file": "coords_new.json",           "kind": "dict", "label": "YOLO-NEW"},
    {"key": "scrfd_only",  "file": "coords_scrfd_only.json",    "kind": "list", "label": "SCRFD-ONLY"},
    {"key": "scrfd_full",  "file": "coords_scrfd_full.json",    "kind": "dict", "label": "SCRFD-FULL"},
    {"key": "yunet_only",  "file": "coords_yunet_only.json",    "kind": "list", "label": "YUNET-ONLY"},
    {"key": "yunet_full",  "file": "coords_yunet_full.json",    "kind": "dict", "label": "YUNET-FULL"},
]

BASELINE_KEY = "old"


# ─── 기본 유틸 ────────────────────────────────────────────────
def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def flatten(entry, kind):
    """entry(해당 프레임의 raw json 값)를 박스 리스트로 평탄화.
    kind='list' -> entry 자체가 이미 [[x1,y1,x2,y2],...]
    kind='dict' -> {"yolo"/"det": [...], "heatmap": [...], "reverse": [...], "reverse_heat": [...]} 를 전부 합침
    """
    if entry is None:
        return []
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        primary = entry.get("yolo", entry.get("det", []))
        return (primary
                + entry.get("heatmap", [])
                + entry.get("reverse", [])
                + entry.get("reverse_heat", []))
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


# ─── dict 형식(NEW/FULL) 박스 소스별 GT 매칭 ────────────────────
def attribute_gt_match(gt, entry, thr=IOU_THRESH):
    """entry가 dict(kind='dict')일 때만 의미 있음. list면 소스 분리가 불가능."""
    if not isinstance(entry, dict):
        return {"primary": 0, "heatmap": 0, "reverse": 0, "miss": len(gt)}

    counts = {"primary": 0, "heatmap": 0, "reverse": 0}
    matched_gt = set()
    primary_key = "yolo" if "yolo" in entry else "det"
    for src, out_key in [(primary_key, "primary"), ("heatmap", "heatmap"),
                         ("reverse", "reverse"), ("reverse_heat", "reverse")]:
        for pred in entry.get(src, []):
            best_v, best_i = 0, -1
            for i, g in enumerate(gt):
                if i in matched_gt:
                    continue
                v = iou(g, pred)
                if v >= thr and v > best_v:
                    best_v, best_i = v, i
            if best_i >= 0:
                counts[out_key] += 1
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


# ─── 요약 출력 ─────────────
def print_summary(rows, variant_keys, labels):
    if not rows:
        print("  (해당 프레임 없음)")
        return

    def avg(k):   return float(np.mean([r[k] for r in rows]))
    def total(k): return int(sum(r[k] for r in rows))

    col_w = 11
    print(f"  {'':<22}" + "".join(f"{labels[v]:>{col_w}}" for v in variant_keys))
    print("  " + "-" * (22 + col_w * len(variant_keys)))

    def row(label, key_pat):
        vals = [avg(key_pat.format(v)) for v in variant_keys]
        print(f"  {label:<22}" + "".join(f"{v:>{col_w}.3f}" for v in vals))

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
            wells = [total(f"{v}_{fmt_key}") for v in variant_keys]
            pcts  = [100 * w / gt_t for w in wells]
            print(f"  {'Well-Cov ≥95% ('+tag+')':<22}" +
                  "".join(f"{p:>{col_w-1}.1f}%" for p in pcts))


# ═══════════════════════════════════════════════════════════════
# 메인 루프
# ═══════════════════════════════════════════════════════════════
overall = []

for video_name in VIDEOS:
    GT_DIR = f"data/{video_name}/labels_labelme"
    if not os.path.isdir(GT_DIR):
        print(f"[{video_name}] GT 폴더 없음, 스킵")
        continue

    # 이 영상에서 실제로 존재하는 파일만 로드 (없는 디텍터는 자동으로 스킵됨)
    loaded = {}
    for v in VARIANTS:
        path = f"results/{video_name}_{v['file']}"
        if os.path.exists(path):
            with open(path) as f:
                loaded[v["key"]] = json.load(f)

    if len(loaded) < 2:
        print(f"[{video_name}] 비교 가능한 결과가 2개 미만, 스킵 (존재: {list(loaded.keys())})")
        continue

    variant_keys = [v["key"] for v in VARIANTS if v["key"] in loaded]
    kind_of      = {v["key"]: v["kind"] for v in VARIANTS}
    label_of     = {v["key"]: v["label"] for v in VARIANTS}

    gt_coords = {}
    for jf in sorted(os.listdir(GT_DIR)):
        if not jf.endswith(".json"):
            continue
        with open(os.path.join(GT_DIR, jf)) as f:
            lm = json.load(f)
        key = jf.replace(".json", "")
        gt_coords[key] = [
            [s["points"][0][0], s["points"][0][1],
             s["points"][1][0], s["points"][1][1]]
            for s in lm["shapes"]
        ]

    rows = []
    # 소스별 GT 매칭 총계 (dict 변형에 대해서만)
    src_total = {k: {"primary": 0, "heatmap": 0, "reverse": 0, "miss": 0}
                 for k in variant_keys if kind_of[k] == "dict"}

    print(f"\n📹 {video_name} 처리 중 (변형: {', '.join(label_of[k] for k in variant_keys)})...")

    for key in sorted(gt_coords.keys()):
        gt = gt_coords[key]
        row = {"frame": key, "gt_count": len(gt)}

        for vk in variant_keys:
            raw_entry = loaded[vk].get(key, [] if kind_of[vk] == "list" else {})
            pred = flatten(raw_entry, kind_of[vk])
            m = compute_metrics(gt, pred)

            row[f"{vk}_count"]       = m["count"]
            row[f"{vk}_iou"]         = round(m["iou"], 3)
            row[f"{vk}_prec"]        = round(m["prec"], 3)
            row[f"{vk}_rec"]         = round(m["rec"], 3)
            row[f"{vk}_f1"]          = round(m["f1"], 3)
            row[f"{vk}_miss"]        = round(m["miss"], 3)
            row[f"{vk}_cov_bbox"]    = round(m["cov_bbox"], 3)
            row[f"{vk}_cov_circle"]  = round(m["cov_circle"], 3)
            row[f"{vk}_well_bbox"]   = m["well_bbox"]
            row[f"{vk}_well_circle"] = m["well_circle"]

            if kind_of[vk] == "dict":
                src = attribute_gt_match(gt, raw_entry)
                for k in src_total[vk]:
                    src_total[vk][k] += src[k]
                row[f"{vk}_matched_primary"] = src["primary"]
                row[f"{vk}_matched_heatmap"] = src["heatmap"]
                row[f"{vk}_matched_reverse"] = src["reverse"]
                row[f"{vk}_missed"]          = src["miss"]

        rows.append(row)

    if not rows:
        continue

    OUTPUT_CSV = f"results/comparison_{video_name}.csv"
    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*60}\n📹 {video_name}\n{'='*60}")
    print(f"\n[ALL FRAMES — n={len(rows)}]")
    print_summary(rows, variant_keys, label_of)

    if BASELINE_KEY in variant_keys:
        hard_rows = [r for r in rows if r[f"{BASELINE_KEY}_miss"] > 0]
        print(f"\n[HARD FRAMES — {label_of[BASELINE_KEY]}가 놓친 GT 있는 프레임만, "
              f"n={len(hard_rows)}/{len(rows)}]")
        print_summary(hard_rows, variant_keys, label_of)
    else:
        hard_rows = []
        print(f"\n[HARD FRAMES 생략 — 기준 변형 '{BASELINE_KEY}' 결과 없음]")

    gt_total = sum(r["gt_count"] for r in rows)
    if gt_total > 0:
        for vk in variant_keys:
            if vk not in src_total:
                continue
            print(f"\n[{label_of[vk]} 박스 소스별 GT 매칭 — 전체 GT {gt_total}개]")
            for k, lbl in [("primary", "원본검출"), ("heatmap", "히트맵"),
                           ("reverse", "역방향"), ("miss", "미검출")]:
                print(f"  {lbl:<8} {src_total[vk][k]:5d}  "
                      f"({100*src_total[vk][k]/gt_total:5.1f}%)")

    print(f"\n📊 저장: {OUTPUT_CSV}")

    def avgr(rs, k): return float(np.mean([r[k] for r in rs])) if rs else 0.0
    def wellpct(rs, k):
        gt_t = sum(r["gt_count"] for r in rs)
        return 100 * sum(r[k] for r in rs) / gt_t if gt_t else 0.0

    summary_row = {"video": video_name, "n_all": len(rows), "n_hard": len(hard_rows)}
    for vk in variant_keys:
        for k in ["f1", "cov_bbox", "cov_circle"]:
            summary_row[f"all_{vk}_{k}"] = avgr(rows, f"{vk}_{k}")
        summary_row[f"all_{vk}_well_circle"] = wellpct(rows, f"{vk}_well_circle")
        if hard_rows:
            for k in ["rec", "cov_bbox", "cov_circle"]:
                summary_row[f"hard_{vk}_{k}"] = avgr(hard_rows, f"{vk}_{k}")
    summary_row["variant_keys"] = variant_keys
    overall.append(summary_row)


# ═══════════════════════════════════════════════════════════════
# 전체 종합
# ═══════════════════════════════════════════════════════════════
if overall:
    label_of_all = {v["key"]: v["label"] for v in VARIANTS}

    print(f"\n\n{'='*100}")
    print("🏁 ALL FRAMES — F1 / Coverage(circle) / Well-Covered(circle)  (영상마다 존재하는 변형만 표시)")
    print('='*100)
    for s in overall:
        vks = s["variant_keys"]
        print(f"\n[{s['video']}]  n={s['n_all']}")
        hdr = f"    {'변형':<12}{'F1':>8}{'CovB':>8}{'CovC':>8}{'WellC%':>9}"
        print(hdr)
        for vk in vks:
            print(f"    {label_of_all[vk]:<12}"
                  f"{s.get(f'all_{vk}_f1', 0):>8.3f}"
                  f"{s.get(f'all_{vk}_cov_bbox', 0):>8.3f}"
                  f"{s.get(f'all_{vk}_cov_circle', 0):>8.3f}"
                  f"{s.get(f'all_{vk}_well_circle', 0):>8.1f}%")

    print(f"\n{'='*100}")
    print(f" HARD FRAMES (기준: {BASELINE_KEY.upper()}가 놓친 프레임) — Recall / Coverage(circle)")
    print('='*100)
    for s in overall:
        if s["n_hard"] == 0:
            print(f"\n[{s['video']}]  (하드 프레임 없음 또는 기준 변형 결과 없음)")
            continue
        vks = s["variant_keys"]
        print(f"\n[{s['video']}]  n_hard={s['n_hard']}/{s['n_all']}")
        hdr = f"    {'변형':<12}{'Recall':>8}{'CovB':>8}{'CovC':>8}"
        print(hdr)
        for vk in vks:
            if f"hard_{vk}_rec" not in s:
                continue
            print(f"    {label_of_all[vk]:<12}"
                  f"{s.get(f'hard_{vk}_rec', 0):>8.3f}"
                  f"{s.get(f'hard_{vk}_cov_bbox', 0):>8.3f}"
                  f"{s.get(f'hard_{vk}_cov_circle', 0):>8.3f}")

print("\n 비교 완료!")