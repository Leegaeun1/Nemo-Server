"""
sample_frames.py
================
등록자 FRR 라벨링을 위한 샘플 프레임 추출 스크립트.

추출 방식
---------
1. 균등 간격 샘플링       : 전체 프레임에서 N_UNIFORM개 균등 추출
2. 저신뢰도 프레임 추가   : YOLO conf가 낮은 프레임 (가려짐/측면 가능성)
   → 두 방식을 합쳐서 중복 제거 후 최대 MAX_FRAMES개 저장

출력
----
data/video_XX/registered_sample/
    frame_XXXX.jpg   ← LabelMe로 열어서 등록자 얼굴만 bbox 그리기

라벨링 완료 후
--------------
LabelMe에서 저장하면 같은 폴더에 frame_XXXX.json 생성됨.
measure_identity.py 실행 시 이 폴더를 labels_registered로 자동 인식.
(measure_identity.py의 reg_label_dir를 registered_sample로 바꿔도 되고,
 또는 labels_registered/ 폴더로 복사해도 됨)
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ==========================================
# ★ 설정
# ==========================================
VIDEOS = [f"video_{i:02d}" for i in range(1, 11)]

FACE_MODEL_PATH = "models/yolov12s-face.pt"
CONF_THRESHOLD  = 0.30

N_UNIFORM    = 30   # 균등 간격 샘플 수
N_HARD       = 20   # 저신뢰도(어려운) 프레임 추가 수
MAX_FRAMES   = 50   # 최대 출력 프레임 수

# 저신뢰도 기준: conf 평균이 이 값 이하인 프레임을 "어려운 프레임"으로 분류
HARD_CONF_THRESH = 0.40
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}")

face_detector = YOLO(FACE_MODEL_PATH).to(device)


def extract_sample_frames(video_name: str):
    video_path = f"videos/{video_name}.mp4"
    out_dir    = f"data/{video_name}/registered_sample"

    if not os.path.exists(video_path):
        print(f"[{video_name}] 영상 없음, 스킵")
        return

    # 이미 추출된 경우 스킵
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"[{video_name}] 이미 존재 ({len(os.listdir(out_dir))}개), 스킵")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ── 프레임 로드 ──────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while cap.isOpened():
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(frm)
    cap.release()
    total = len(frames)
    print(f"\n[{video_name}] 총 {total}프레임 로드")

    # ── 1. 균등 간격 샘플링 ──────────────────────────────────
    uniform_indices = set(
        int(i * total / N_UNIFORM) for i in range(N_UNIFORM)
    )

    # ── 2. YOLO 실행 → 저신뢰도 프레임 탐지 ─────────────────
    print(f"  YOLO 실행 중 (저신뢰도 프레임 탐지)...")
    frame_conf = {}   # {frame_idx: avg_conf or None}

    for idx, frm in enumerate(frames):
        if idx % 50 == 0:
            print(f"    {idx}/{total} 처리 중...")

        res = face_detector(frm, conf=CONF_THRESHOLD,
                            imgsz=640, device=device, verbose=False)

        if res[0].boxes is not None and len(res[0].boxes) > 0:
            confs = res[0].boxes.conf.cpu().tolist()
            frame_conf[idx] = float(np.mean(confs))
        else:
            frame_conf[idx] = None   # 탐지 자체 실패

    # 탐지는 됐지만 conf가 낮은 프레임 → "어려운 프레임"
    hard_candidates = [
        (idx, conf) for idx, conf in frame_conf.items()
        if conf is not None and conf < HARD_CONF_THRESH
    ]
    # conf 낮은 순으로 정렬
    hard_candidates.sort(key=lambda x: x[1])
    hard_indices = set(idx for idx, _ in hard_candidates[:N_HARD])

    # ── 3. 합산 및 중복 제거 ─────────────────────────────────
    selected = sorted(uniform_indices | hard_indices)[:MAX_FRAMES]

    # ── 4. 저장 ──────────────────────────────────────────────
    saved = 0
    uniform_cnt = 0
    hard_cnt    = 0

    for idx in selected:
        fname = f"frame_{idx:04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), frames[idx])
        saved += 1
        if idx in uniform_indices:
            uniform_cnt += 1
        if idx in hard_indices:
            hard_cnt += 1

    print(f"  ✅ [{video_name}] {saved}프레임 저장 완료")
    print(f"     균등 샘플: {uniform_cnt}개")
    print(f"     저신뢰도 추가: {hard_cnt}개  "
          f"(conf < {HARD_CONF_THRESH}인 프레임)")
    print(f"     저장 위치: {out_dir}/")
    print(f"\n  📌 LabelMe로 열기:")
    print(f"     labelme {out_dir}")
    print(f"  📌 등록자 얼굴에만 bbox 그리고 저장하세요.")
    print(f"     label명: face_registered (또는 아무거나)")


# ── 실행 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  등록자 라벨링용 샘플 프레임 추출")
    print(f"  균등 {N_UNIFORM}개 + 저신뢰도 {N_HARD}개 = 최대 {MAX_FRAMES}개/영상")
    print("=" * 55)

    for vname in VIDEOS:
        extract_sample_frames(vname)

    print("\n\n" + "=" * 55)
    print("  전체 완료!")
    print("=" * 55)
    print(f"\n다음 단계:")
    print(f"  1. 각 영상의 registered_sample/ 폴더를 LabelMe로 열기")
    print(f"  2. 등록자 얼굴에만 bbox 그리기 (다른 사람 bbox는 그리지 말 것)")
    print(f"  3. 저장 (json 파일이 같은 폴더에 생성됨)")
    print(f"  4. measure_identity.py 실행")
    print(f"\n  ※ measure_identity.py의 reg_label_dir를")
    print(f"     'labels_registered' → 'registered_sample' 로 바꿔야 합니다.")