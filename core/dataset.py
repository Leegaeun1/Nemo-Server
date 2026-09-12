import cv2
import os
import random
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm 

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
model = YOLO('models/yolov11n-face.pt') 

input_base_dir = 'dataset/images' 
output_base_dir = 'dataset_extreme_occ' # 혹시 모르니 새로운 폴더에 저장
splits = ['train', 'val', 'test']

print("🔄 '눈/코/입 하나만 보여도 인식' - 극단적 가림(Extreme Occlusion) 증강을 시작합니다...\n")

total_success = 0

for split in splits:
    input_split_dir = os.path.join(input_base_dir, split)
    if not os.path.exists(input_split_dir): continue

    out_img_dir = os.path.join(output_base_dir, 'images', split)
    out_label_dir = os.path.join(output_base_dir, 'labels', split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    print(f"\n▶️ [{split.upper()}] 폴더 처리 중...")
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(root, f) for root, _, files in os.walk(input_split_dir) for f in files if f.lower().endswith(valid_extensions)]

    if not image_paths: continue

    split_count = 0
    
    for img_path in tqdm(image_paths, desc=f"{split.upper()} 진행률", unit="장"):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        results = model(img, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0: continue

        yolo_labels = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # 80% 확률로 가림막 적용 (학습 강도를 높이기 위해 확률 증가)
            if random.random() < 0.80:
                face_w, face_h = x2 - x1, y2 - y1
                
                # 가림막 스타일 결정 (30%: 무작위 박스, 70%: 반갈죽 가림막)
                occ_style = random.random()
                
                if occ_style < 0.3:
                    # 스타일 1: 기존 무작위 박스 (비율을 35% ~ 60%로 상향)
                    occ_w = int(face_w * random.uniform(0.35, 0.60))
                    occ_h = int(face_h * random.uniform(0.35, 0.60))
                    occ_x1 = random.randint(x1, max(x1, x2 - occ_w))
                    occ_y1 = random.randint(y1, max(y1, y2 - occ_h))
                    occ_x2, occ_y2 = occ_x1 + occ_w, occ_y1 + occ_h
                else:
                    # 스타일 2: 기둥/팔 가림 모사 (가로/세로 절반 이상을 날려버림)
                    cut_type = random.choice(['left', 'right', 'top', 'bottom'])
                    occ_x1, occ_y1, occ_x2, occ_y2 = x1, y1, x2, y2
                    
                    if cut_type == 'left':
                        occ_x2 = x1 + int(face_w * random.uniform(0.4, 0.65)) # 좌측 40~65% 가림
                    elif cut_type == 'right':
                        occ_x1 = x2 - int(face_w * random.uniform(0.4, 0.65)) # 우측 40~65% 가림
                    elif cut_type == 'top':
                        occ_y2 = y1 + int(face_h * random.uniform(0.4, 0.65)) # 상단(눈/이마) 가림
                    else: # bottom
                        occ_y1 = y2 - int(face_h * random.uniform(0.4, 0.65)) # 하단(입/코) 가림

                # 색상 결정 (50% 검은색, 50% 피부색과 무관한 랜덤 단색)
                if random.random() < 0.5:
                    color = (0, 0, 0) # 검정 (어둠 가림)
                else:
                    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)) # 랜덤 색상 (옷, 사물 모사)
                    
                cv2.rectangle(img, (occ_x1, occ_y1), (occ_x2, occ_y2), color, -1)

        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        unique_name = f"{base_name}_extreme_occ" 
        
        cv2.imwrite(os.path.join(out_img_dir, f"{unique_name}.jpg"), img)
        with open(os.path.join(out_label_dir, f"{unique_name}.txt"), 'w') as f:
            f.write('\n'.join(yolo_labels))
            
        split_count += 1
        total_success += 1

print(f"\n✅ 극단적 가림 훈련 데이터셋 {total_success}장 생성 완료!")