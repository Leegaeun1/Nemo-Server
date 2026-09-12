import json, os, base64

labelme_dir = "data/video_09/labels_labelme"
frames_dir = "data/video_09/frames"

files = [f for f in os.listdir(labelme_dir) if f.endswith(".json")]
print(f"총 {len(files)}개 수정 시작...")

for i, json_file in enumerate(sorted(files)):
    json_path = os.path.join(labelme_dir, json_file)
    img_file = json_file.replace(".json", ".jpg")
    img_path = os.path.join(frames_dir, img_file)

    with open(json_path) as f:
        data = json.load(f)

    # imagePath 수정
    data["imagePath"] = f"../frames/{img_file}"

    # imageData 없으면 추가
    if not data.get("imageData"):
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                data["imageData"] = base64.b64encode(f.read()).decode("utf-8")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(files)} 완료...")

print("전체 수정 완료!")