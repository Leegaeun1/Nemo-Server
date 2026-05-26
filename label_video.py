import cv2
import os
import json
import base64
import shutil
from PIL import Image
from ultralytics import YOLO

# ==================== 설정 ====================
VIDEO_DIR = "videos"           # 영상 파일들이 있는 폴더
DATA_DIR = "data"              # 결과 저장 폴더
MODEL_PATH = "models/yolov12s-face.pt"  # YOLO 모델 경로
CONF_THRESHOLD = 0.5           # 신뢰도 임계값
# ==============================================

model = YOLO(MODEL_PATH)

def get_video_name(video_file):
    """확장자 제거한 영상 이름 반환"""
    return os.path.splitext(video_file)[0]

def setup_dirs(video_name):
    """디렉토리 생성"""
    dirs = {
        "frames":   os.path.join(DATA_DIR, video_name, "frames"),
        "yolo":     os.path.join(DATA_DIR, video_name, "labels_yolo"),
        "labelme":  os.path.join(DATA_DIR, video_name, "labels_labelme"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

# ==================== 1단계: 프레임 추출 ====================
def extract_frames(video_name, dirs):
    video_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
    output_dir = dirs["frames"]

    # 이미 추출된 경우 스킵
    if os.listdir(output_dir):
        print(f"  [1단계] {video_name} 프레임 이미 존재, 스킵")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [1단계] {video_name} | FPS: {fps:.1f} | 총 프레임: {total}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{idx:04d}.jpg"), frame)
        idx += 1

    cap.release()
    print(f"  [1단계] 완료 → {idx}개 프레임 저장")

# ==================== 2단계: YOLO 실행 ====================
def run_yolo(video_name, dirs):
    frames_dir = dirs["frames"]
    labels_dir = dirs["yolo"]

    # 이미 실행된 경우 스킵
    if os.listdir(labels_dir):
        print(f"  [2단계] {video_name} YOLO 결과 이미 존재, 스킵")
        return

    img_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    print(f"  [2단계] {video_name} | YOLO 실행 중... ({len(img_files)}개)")

    for img_file in img_files:
        img_path = os.path.join(frames_dir, img_file)
        results = model(img_path, conf=CONF_THRESHOLD, verbose=False)

        label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))
        results[0].save_txt(label_path, save_conf=True)

    print(f"  [2단계] 완료 → {len(img_files)}개 txt 저장")

# ==================== 3단계: LabelMe 변환 ====================
def convert_to_labelme(video_name, dirs):
    frames_dir = dirs["frames"]
    labels_dir = dirs["yolo"]
    labelme_dir = dirs["labelme"]

    # 이미 변환된 경우 스킵
    if os.listdir(labelme_dir):
        print(f"  [3단계] {video_name} LabelMe json 이미 존재, 스킵")
        return

    txt_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    print(f"  [3단계] {video_name} | LabelMe 변환 중... ({len(txt_files)}개)")

    for txt_file in txt_files:
        img_file = txt_file.replace(".txt", ".jpg")
        img_path = os.path.join(frames_dir, img_file)

        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        W, H = img.size

        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        shapes = []
        with open(os.path.join(labels_dir, txt_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, w, h = map(float, parts[:5])
                x1 = (xc - w/2) * W
                y1 = (yc - h/2) * H
                x2 = (xc + w/2) * W
                y2 = (yc + h/2) * H
                shapes.append({
                    "label": "face",
                    "points": [[x1, y1], [x2, y2]],
                    "shape_type": "rectangle",
                    "flags": {}
                })

        json_data = {
            "version": "5.0.1",
            "imagePath": img_file,
            "imageHeight": H,
            "imageWidth": W,
            "imageData": image_data,
            "shapes": shapes,
            "flags": {}
        }

        json_path = os.path.join(labelme_dir, txt_file.replace(".txt", ".json"))
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    print(f"  [3단계] 완료 → {len(txt_files)}개 json 저장")

# ==================== 전체 실행 ====================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # videos/ 폴더의 mp4 파일 자동 감지
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])

    if not video_files:
        print("videos/ 폴더에 mp4 파일이 없어요!")
        return

    print(f"총 {len(video_files)}개 영상 발견: {video_files}\n")

    for video_file in video_files:
        video_name = get_video_name(video_file)
        print(f"{'='*40}")
        print(f"처리 중: {video_name}")
        print(f"{'='*40}")

        dirs = setup_dirs(video_name)
        extract_frames(video_name, dirs)
        run_yolo(video_name, dirs)
        convert_to_labelme(video_name, dirs)
        print(f"✅ {video_name} 완료!\n")

    print("모든 영상 처리 완료!")
    print("LabelMe에서 수정 후 4단계 비교 코드를 실행하세요.")

if __name__ == "__main__":
    main()