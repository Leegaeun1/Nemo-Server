import csv
import os

import cv2
import numpy as np

import measure_identity as mi   # 모델 로드 + extract_embedding 재사용 (평가 루프는 __main__ 에서만 실행됨)

OUT_CSV = "results/identity_similarity.csv"
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def main():
    rows = []
    for video in mi.VIDEOS:
        face_dir = f"data/{video}/registered_face"
        reg_dir = f"data/{video}/registered_sample"
        unreg_dir = f"data/{video}/labels_labelme"
        video_path = f"videos/{video}.mp4"
        if not (os.path.exists(video_path) and os.path.isdir(face_dir) and os.path.isdir(reg_dir)):
            continue
        reg_gt = mi.load_labelme_boxes(reg_dir)
        reg_gt = {k: v for k, v in reg_gt.items() if v}
        if not reg_gt:
            continue
        unreg_gt = mi.load_labelme_boxes(unreg_dir)
        known = mi.load_known_embeddings(face_dir)
        if not known:
            print(f"[{video}] 등록 임베딩 없음, 스킵")
            continue

        cap = cv2.VideoCapture(video_path)
        for key in sorted(reg_gt):
            idx = int(key.replace("frame_", ""))
            frame = read_frame(cap, idx)
            if frame is None:
                continue
            rboxes = reg_gt[key]
            items = [("genuine", b) for b in rboxes]
            items += [("impostor", b) for b in unreg_gt.get(key, [])
                      if not any(mi.iou(b, r) >= 0.3 for r in rboxes)]
            for kind, box in items:
                emb = mi.extract_embedding(frame, box)
                if emb is None:
                    sim = None   # 얼굴 정렬 실패 (임베딩 불가)
                else:
                    sim = max(mi.cosine_sim(emb, k) for k in known)
                rows.append({"video": video, "frame": idx, "kind": kind,
                             "sim": "" if sim is None else round(sim, 4)})
        cap.release()
        print(f"[{video}] 완료")

    os.makedirs("results", exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame", "kind", "sim"])
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {OUT_CSV}")

    # 요약
    def vals(video, kind):
        return [r["sim"] for r in rows
                if (video is None or r["video"] == video) and r["kind"] == kind and r["sim"] != ""]

    def nofail(video, kind):
        return sum(1 for r in rows
                   if (video is None or r["video"] == video) and r["kind"] == kind and r["sim"] == "")

    print("\n영상별 유사도 (genuine=등록자, impostor=미등록자) / 임베딩 실패 수")
    print(f"{'video':<10}{'gen n':>6}{'gen mean':>10}{'gen min':>9}{'gen fail':>9}"
          f"{'imp n':>7}{'imp mean':>10}{'imp max':>9}{'imp>0.35':>9}{'imp fail':>9}")
    for v in sorted({r["video"] for r in rows}):
        g, i = vals(v, "genuine"), vals(v, "impostor")
        print(f"{v:<10}{len(g):>6}{np.mean(g) if g else 0:>10.3f}{min(g) if g else 0:>9.3f}"
              f"{nofail(v, 'genuine'):>9}{len(i):>7}{np.mean(i) if i else 0:>10.3f}"
              f"{max(i) if i else 0:>9.3f}{sum(1 for s in i if s > 0.35):>9}{nofail(v, 'impostor'):>9}")

    g, i = vals(None, "genuine"), vals(None, "impostor")
    print("\n임계값 스윕 (임베딩 단위, 전체 합산; 얼굴 정렬 실패는 제외)")
    print(f"{'thr':>6}{'FRR(genuine<=thr)':>20}{'FAR(impostor>thr)':>20}")
    for t in THRESHOLDS:
        frr = sum(1 for s in g if s <= t) / max(len(g), 1)
        far = sum(1 for s in i if s > t) / max(len(i), 1)
        print(f"{t:>6.2f}{frr * 100:>19.1f}%{far * 100:>19.1f}%")

    # EER
    best = min(((abs(sum(1 for s in g if s <= t) / max(len(g), 1)
                     - sum(1 for s in i if s > t) / max(len(i), 1)), t)
                for t in np.arange(0.0, 0.9, 0.01)), key=lambda x: x[0])
    t = best[1]
    print(f"\nEER 근처 임계값 ~ {t:.2f}  (FRR {100 * sum(1 for s in g if s <= t) / max(len(g), 1):.1f}%, "
          f"FAR {100 * sum(1 for s in i if s > t) / max(len(i), 1):.1f}%)")


if __name__ == "__main__":
    main()
