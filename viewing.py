# 박스 라벨을 소스별로 다르게 줘서 LabelMe가 색깔로 구분해줌.

import json
import os
import cv2

# ────────────── 설정 ──────────────
VIDEO_NAME       = "video_03"            # 보고 싶은 비디오
NEW_JSON         = f"results/{VIDEO_NAME}_coords_new.json"
IMG_DIR          = f"data/{VIDEO_NAME}/frames"           # 프레임 이미지 폴더
GT_DIR           = f"data/{VIDEO_NAME}/labels_labelme"   # GT
OUT_DIR          = f"results/{VIDEO_NAME}_labelme_new"

IMG_EXT          = ".jpg"     # 프레임 이미지 확장자 (.jpg or .png)
INCLUDE_GT       = True       # GT 박스도 함께 표시할지
ONLY_ADDED       = False      # True면 heatmap/reverse가 추가된 프레임만 저장 (빠른 디버깅용)
# ─────────────────────────────────


def make_shape(box, label):
    x1, y1, x2, y2 = box
    return {
        "label": label,
        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
    }


def get_image_size():
    """샘플 이미지 한 장 읽어서 해상도 추출."""
    files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))])
    if not files:
        raise FileNotFoundError(f"이미지가 없음: {IMG_DIR}")
    img = cv2.imread(os.path.join(IMG_DIR, files[0]))
    return img.shape[1], img.shape[0]   # W, H


os.makedirs(OUT_DIR, exist_ok=True)
W, H = get_image_size()
abs_img_dir = os.path.abspath(IMG_DIR)

with open(NEW_JSON) as f:
    new_coords = json.load(f)

# GT 로드 (있으면)
gt_coords = {}
if INCLUDE_GT and os.path.isdir(GT_DIR):
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

n_written = 0
n_skipped = 0

for frame_key, entry in new_coords.items():
    # dict 포맷 가정 (NEW). list로 들어와도 yolo로 간주.
    if isinstance(entry, dict):
        yolo_boxes    = entry.get("yolo", [])
        heatmap_boxes = entry.get("heatmap", [])
        reverse_boxes = entry.get("reverse", [])
    else:
        yolo_boxes, heatmap_boxes, reverse_boxes = entry, [], []

    if ONLY_ADDED and not (heatmap_boxes or reverse_boxes):
        n_skipped += 1
        continue

    shapes = []

    # GT 먼저 (LabelMe에서 보통 위에 그려짐)
    if frame_key in gt_coords:
        for b in gt_coords[frame_key]:
            shapes.append(make_shape(b, "face_GT"))

    for b in yolo_boxes:
        shapes.append(make_shape(b, "face_yolo"))
    for b in heatmap_boxes:
        shapes.append(make_shape(b, "face_heatmap"))
    for b in reverse_boxes:
        shapes.append(make_shape(b, "face_reverse"))

    image_name = f"{frame_key}{IMG_EXT}"
    image_full = os.path.join(abs_img_dir, image_name)

    labelme_json = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_full,   # 절대경로 — LabelMe가 어디서든 찾을 수 있음
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }

    out_path = os.path.join(OUT_DIR, f"{frame_key}.json")
    with open(out_path, "w") as f:
        json.dump(labelme_json, f, indent=2)
    n_written += 1

print(f"\n✅ 완료!")
print(f"   저장: {OUT_DIR}")
print(f"   작성: {n_written}개 프레임" + (f" (스킵: {n_skipped}개)" if ONLY_ADDED else ""))
print(f"\n📂 LabelMe로 열기:")
print(f"   labelme {OUT_DIR}")
print(f"\n🎨 라벨 색깔로 자동 구분돼서 보임:")
print(f"   face_GT       (정답)")
print(f"   face_yolo     (YOLO 원본 검출)")
print(f"   face_heatmap  (히트맵 복원으로 추가됨)")
print(f"   face_reverse  (역방향 패스로 추가됨)")
print(f"\n💡 Tip: LabelMe 좌측 Label List에서 체크박스로 끄고/켜기 가능")
print(f"        → face_yolo만 끄면 추가된 박스(heatmap/reverse)만 보여서 오탐 찾기 쉬움")