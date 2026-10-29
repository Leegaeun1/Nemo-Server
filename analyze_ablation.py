"""
analyze_ablation.py
===================
히트맵 / 양방향 처리의 기여를 분리하는 ablation 과, 노출 지표를 계산한다.
모델 추론은 하지 않고 extract_coords_old.py / extract_coords_new.py 의 저장 결과만 사용한다.

변형 (모두 같은 탐지기, 같은 트래커 초기화 조건)
  Baseline       : 정방향 YOLO 박스만                         (coords_old.json)
  Fwd-YOLO(new)  : new 파이프라인의 정방향 YOLO 박스만 (fwd_raw). Baseline 과 같아야 트래커 차이가 없다는 확인용
  +Heatmap       : 정방향 YOLO + 정방향 히트맵 복원 박스
  +Bidirectional : 정방향 YOLO + 역방향 YOLO 박스 (히트맵 없음)
  +Bidir+RevHeat : 정방향 YOLO + 역방향 YOLO + 역방향 히트맵 복원 (정방향 히트맵 없음)
  Full           : 위 소스 전부 (정방향/역방향 YOLO + 양방향 히트맵 복원)
  hard 프레임 = Baseline(old) 이 GT 를 놓친 프레임

지표 (compare_all_V3.py 와 같은 정의: IoU >= 0.3 로 GT 와 매칭)
  Recall      : 프레임별 (매칭된 GT 수 / GT 수) 의 평균
  CovC        : GT 얼굴 영역이 원형 블러(반경 0.6*max(w,h))로 덮인 비율의 평균
  Exposed-GT  : 매칭되지 않은 GT 얼굴 비율 (= 1 - GT 단위 recall)
  Exposed-Fr  : 매칭 안 된 GT 얼굴이 하나라도 있는 프레임 비율 (= 원본 얼굴이 노출된 프레임)
  각 지표를 전체 프레임(All)과 hard 프레임(Baseline 이 GT 를 놓친 프레임)에 대해 계산한다.

사용법: python analyze_ablation.py [결과폴더 예: results_reset/yolov12s-face]
출력  : 콘솔 표 + <결과폴더>/ablation.csv
"""
import csv
import json
import os
import sys

import numpy as np

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results_reset/yolov12s-face"
VIDEOS = [f"video_{i:02d}" for i in range(1, 11)]
IOU_THRESH = 0.3
RADIUS_FACTOR = 0.6   # apply_mosaic 의 radius = 0.6 * max(w, h)
VARIANTS = ["Baseline", "Fwd-YOLO(new)", "+Heatmap", "+Bidirectional",
            "+Bidir+RevHeat", "Full"]


# ─── compare_all_V3.py 와 동일한 지표 정의 ─────────────────────
def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def matched_count(gt, pred, thr=IOU_THRESH):
    if not pred or not gt:
        return 0
    tp, used = 0, set()
    for p in pred:
        for i, g in enumerate(gt):
            if i not in used and iou(g, p) >= thr:
                tp += 1
                used.add(i)
                break
    return tp


def circle_coverage(gt_box, pred_boxes, factor=RADIUS_FACTOR):
    gx1, gy1, gx2, gy2 = gt_box
    gw_full, gh_full = gx2 - gx1, gy2 - gy1
    if gw_full <= 0 or gh_full <= 0:
        return 0.0
    scale = 1.0
    if gw_full * gh_full > 10000:
        scale = (10000.0 / (gw_full * gh_full)) ** 0.5
    gw, gh = max(1, int(gw_full * scale)), max(1, int(gh_full * scale))
    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    xs_o, ys_o = xs / scale + gx1, ys / scale + gy1
    mask = np.zeros((gh, gw), dtype=bool)
    for px1, py1, px2, py2 in pred_boxes:
        r = max(px2 - px1, py2 - py1) * factor
        if r <= 0:
            continue
        mask |= ((xs_o - (px1 + px2) / 2.0) ** 2 + (ys_o - (py1 + py2) / 2.0) ** 2) <= r * r
    return float(mask.sum()) / (gw * gh)


def load_gt(video):
    gt_dir = f"data/{video}/labels_labelme"
    out = {}
    if not os.path.isdir(gt_dir):
        return out
    for jf in sorted(os.listdir(gt_dir)):
        if jf.endswith(".json"):
            with open(os.path.join(gt_dir, jf)) as f:
                lm = json.load(f)
            # 꼭짓점 순서 정규화 (LabelMe 드래그 방향에 따라 뒤집힘)
            out[jf[:-5]] = [[min(s["points"][0][0], s["points"][1][0]), min(s["points"][0][1], s["points"][1][1]),
                             max(s["points"][0][0], s["points"][1][0]), max(s["points"][0][1], s["points"][1][1])]
                            for s in lm["shapes"]]
    return out


def variant_boxes(old, new, key):
    """프레임 key 에서 변형별 박스 목록."""
    e = new.get(key, {})
    fwd_raw = e.get("fwd_raw", e.get("yolo", []))
    fwd_rec = e.get("fwd_rec_all", [])
    rev_raw = e.get("rev_raw_all", [])
    rev_rec = e.get("rev_rec_all", [])
    return {
        "Baseline":       old.get(key, []),
        "Fwd-YOLO(new)":  fwd_raw,
        "+Heatmap":       merge_sources(fwd_raw, fwd_rec),
        "+Bidirectional": merge_sources(fwd_raw, rev_raw),
        "+Bidir+RevHeat": merge_sources(fwd_raw, rev_raw, rev_rec),
        "Full":           merge_sources(fwd_raw, fwd_rec, rev_raw, rev_rec),
    }


def merge_sources(*sources, thresh=0.3):
    """extract_coords_new.py 의 병합과 동일: 첫 소스는 그대로, 이후 소스는 기존 박스와
    IoU >= thresh 로 겹치지 않는 박스만 순서대로 추가한다 (Table 1 의 Full 과 같은 값이 되도록)."""
    merged = list(sources[0])
    for src in sources[1:]:
        for b in src:
            if not any(iou(b, e) >= thresh for e in merged):
                merged.append(b)
    return merged


def per_frame(gt, pred):
    n = len(gt)
    m = matched_count(gt, pred)
    return {
        "recall":  m / n,
        "cov":     float(np.mean([circle_coverage(g, pred) for g in gt])),
        "gt":      n,
        "missed":  n - m,
        "exposed": 1 if n - m > 0 else 0,
    }


def aggregate(rows):
    if not rows:
        return None
    gt = sum(r["gt"] for r in rows)
    return {
        "frames":     len(rows),
        "recall":     float(np.mean([r["recall"] for r in rows])),
        "covc":       float(np.mean([r["cov"] for r in rows])),
        "exposed_gt": sum(r["missed"] for r in rows) / gt if gt else 0.0,
        "exposed_fr": float(np.mean([r["exposed"] for r in rows])),
    }


def main():
    # frames[variant] = {"all": [...], "hard": [...]}  (영상 구분 없이 프레임 풀링)
    pooled = {v: {"all": [], "hard": []} for v in VARIANTS}
    per_video = []

    for video in VIDEOS:
        old_p = f"{RESULTS_DIR}/{video}_coords_old.json"
        new_p = f"{RESULTS_DIR}/{video}_coords_new.json"
        gt_all = load_gt(video)
        if not (os.path.exists(old_p) and os.path.exists(new_p) and gt_all):
            print(f"[{video}] 결과 또는 GT 없음, 스킵")
            continue
        with open(old_p) as f:
            old = json.load(f)
        with open(new_p) as f:
            new = json.load(f)
        if not any("fwd_rec_all" in e for e in new.values()):
            print(f"[{video}] coords_new.json 에 ablation 용 필드가 없습니다. "
                  f"extract_coords_new.py 를 수정 후 다시 실행해야 합니다. 스킵")
            continue

        vrows = {v: {"all": [], "hard": []} for v in VARIANTS}
        for key, gt in gt_all.items():
            if not gt:
                continue
            boxes = variant_boxes(old, new, key)
            res = {v: per_frame(gt, boxes[v]) for v in VARIANTS}
            is_hard = res["Baseline"]["missed"] > 0
            for v in VARIANTS:
                vrows[v]["all"].append(res[v])
                if is_hard:
                    vrows[v]["hard"].append(res[v])
        for v in VARIANTS:
            pooled[v]["all"] += vrows[v]["all"]
            pooled[v]["hard"] += vrows[v]["hard"]
        per_video.append((video, {v: (aggregate(vrows[v]["all"]), aggregate(vrows[v]["hard"]))
                                  for v in VARIANTS}))

    if not per_video:
        print("분석할 영상이 없습니다.")
        return

    out_rows = []
    for scope in ("all", "hard"):
        n = len(pooled["Baseline"][scope])
        print(f"\n{'=' * 78}\n{RESULTS_DIR}  [{'전체 프레임' if scope == 'all' else 'HARD 프레임 (Baseline 이 놓친 프레임)'}]"
              f"  n={n}\n{'=' * 78}")
        print(f"{'Variant':<16}{'Recall':>9}{'CovC':>9}{'Exposed-GT':>12}{'Exposed-Fr':>12}")
        for v in VARIANTS:
            a = aggregate(pooled[v][scope])
            if a is None:
                continue
            print(f"{v:<16}{a['recall']:>9.3f}{a['covc']:>9.3f}{a['exposed_gt']:>12.3f}{a['exposed_fr']:>12.3f}")
            out_rows.append({"scope": scope, "variant": v, **a})

    for video, d in per_video:
        for v in VARIANTS:
            for scope, a in zip(("all", "hard"), d[v]):
                if a:
                    out_rows.append({"scope": f"{scope}:{video}", "variant": v, **a})

    path = f"{RESULTS_DIR}/ablation.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "variant", "frames", "recall", "covc",
                                          "exposed_gt", "exposed_fr"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
