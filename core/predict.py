import cv2
import numpy as np
from ultralytics import YOLO

def transform_coords(coords, angle, orig_w, orig_h):
    """회전된 좌표를 원본 좌표로 역변환"""
    x, y = coords
    if angle == 90:   return [y, orig_h - x]
    if angle == 180:  return [orig_w - x, orig_h - y]
    if angle == 270:  return [orig_w - y, x]
    return [x, y]

model = YOLO('../models/yolov8n-face.pt') # 사용할 모델 
img = cv2.imread('Test_img/test3.jpg')
h, w = img.shape[:2] # 세로, 가로 길이 

all_faces = [] # [box, score, keypoints] 형태로 저장

for angle in [0, 90, 180, 270]:
    # 이미지 회전 및 추론
    if angle == 0: rotated = img
    else: rotated = cv2.rotate(img, {90:0, 180:1, 270:2}[angle])
    
    results = model.predict(rotated, conf=0.4, imgsz=1024)
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []

        for box, score, kp in zip(boxes, scores, kpts):
            # 1. 박스 좌표 역변환 (좌상단, 우하단)
            p1 = transform_coords([box[0], box[1]], angle, w, h)
            p2 = transform_coords([box[2], box[3]], angle, w, h)
            # 회전 후에는 p1, p2의 대소관계가 바뀔 수 있으므로 재정렬
            new_box = [min(p1[0], p2[0]), min(p1[1], p2[1]), 
                       max(p1[0], p2[0]), max(p1[1], p2[1])]
            
            new_kp = []
            for pt in kp:
                if pt[0] > 0 and pt[1] > 0: # 유효한 좌표인 경우만 변환
                    new_kp.append(transform_coords(pt, angle, w, h))
                else:
                    new_kp.append([0, 0]) # 못 찾은 점은 0으로 유지
                    
            all_faces.append({'box': new_box, 'score': score, 'kpts': new_kp})


# 3. 중복 제거 (단순 거리 기준 혹은 NMS)
# 동일한 얼굴이 여러 각도에서 중복 검출될 수 있으므로 처리가 필요합니다.
def calculate_iou(box1, box2):
    """두 박스 간의 IoU(교집합/합집합 비율) 계산"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 교집합 영역 계산
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        inter_area = 0

    # 각 박스의 넓이 및 합집합 영역 계산
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

# --- 3. 중복 제거 실행 ---
def merge_faces(face1, face2):
    """두 얼굴 데이터를 합침 (박스는 가중 평균, 점은 최고 신뢰도 선택)"""
    s1, s2 = face1['score'], face2['score']
    total_s = s1 + s2
    
    # 박스는 부드럽게 합침
    b1, b2 = face1['box'], face2['box']
    new_box = [(b1[i] * s1 + b2[i] * s2) / total_s for i in range(4)]
    
    # 특징점은 점수가 더 높은 쪽의 것을 그대로 가져옴 (평균 X)
    # 한쪽이 잘못 잡은 점이 섞이는 것을 방지합니다.
    new_kpts = face1['kpts'] if s1 > s2 else face2['kpts']
    
    return {'box': new_box, 'score': max(s1, s2), 'kpts': new_kpts}

final_faces = []
all_faces.sort(key=lambda x: x['score'], reverse=True) # 신뢰도 높은 순서로 정렬

# 중복 판단 기준 (이 값을 낮출수록 더 적극적으로 합칩니다)
iou_threshold = 0.3 

for face in all_faces:
    matched = False
    for i, top_face in enumerate(final_faces):
        # IoU가 기준치 이상이면 동일 인물로 간주하고 병합
        if calculate_iou(face['box'], top_face['box']) > iou_threshold:
            final_faces[i] = merge_faces(top_face, face)
            matched = True
            break
            
    if not matched:
        # 겹치는 얼굴이 없으면 새로운 얼굴로 등록
        final_faces.append(face)

# --- 4. 최종 결과 시각화 ---
for f in final_faces:
    box = f['box']
    kpts = f['kpts']
    
    # 얼굴 박스 그리기
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    
    # 5개 특징점(눈, 코, 입꼬리) 그리기
    for pt in kpts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

# 1. 'Face Detection'이라는 이름의 창을 먼저 만듭니다.
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)

# 2. 창의 크기를 원하는 대로 조절합니다. (가로, 세로)
cv2.resizeWindow('Face Detection', 800, 1500)
cv2.imshow('Face Detection', img)

print(f"Total unique faces found: {len(final_faces)}")
cv2.waitKey(0)
cv2.destroyAllWindows()