import cv2
import numpy as np
import math
from ultralytics import YOLO

def rotate_image_and_get_matrix(image, angle):
    """이미지를 특정 각도로 회전시키고 변환 행렬을 반환"""
    h, w = image.shape[:2] # 세로, 가로 길이 
    center = (w // 2, h // 2) # 중심점
    # 회전 행렬 생성 (크기 유지)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0) # 시계 방향 회전(-), 배율 1
    # 이미지 회전 결과
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

def transform_coords_general(pts, M):
    """Affine 행렬의 역행렬을 사용하여 좌표를 원본으로 되돌림"""
    M_inv = cv2.invertAffineTransform(M) # 역행렬 
    # pts shape: (N, 2)
    ones = np.ones(shape=(len(pts), 1)) # len(pts) * 1 을 1로 채움 
    pts_ones = np.hstack([pts, ones]) # pts랑 ones를 가로로 결합시킴 
    transformed_pts = M_inv.dot(pts_ones.T).T # 세로로 전치 후 역행렬을 수행한 후 다시 가로로 전치
    return transformed_pts # 원래 좌표

def calculate_dist(p1, p2):
    """두 점 사이의 유클리드 거리 계산"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

def is_center_inside(center, box):
    """중심점이 특정 박스 안에 들어있는지 확인"""
    x, y = center
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

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

model = YOLO('yolov8n-face.pt') # 모델 사용
img = cv2.imread('Test_img/input5.jpg') # 사용할 그림 
h, w = img.shape[:2] # 세로, 가로 길이 

all_faces = []
# 45도 단위로 8개 각도 검사
angles = [0, 45, 90, 135, 180, 225, 270, 315]

for angle in angles:
    rotated, M = rotate_image_and_get_matrix(img, angle) # 이미지 회전, 회전 행렬
    # 정밀도를 위해 imgsz를 높이고 augment 적용(데이터 증강 허용)
    results = model.predict(rotated, conf=0.25, imgsz=1024, augment=True)
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() # 바운더리 박스 좌표
        scores = r.boxes.conf.cpu().numpy() # 맞을 확률
        kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else [] # 눈,코,입 주요 지점

        for box, score, kp in zip(boxes, scores, kpts):
            # 1. 박스 좌표 역변환 (네 모서리를 모두 변환 후 감싸는 최소 사각형 찾기)
            # box는 [x_min,y_min,x_max,y_max]임.
            # 각 좌표들 (시계 반대 방향) 
            corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
            orig_corners = transform_coords_general(corners, M) # 원래 좌표
            # 원본 이미지의 box
            new_box = [np.min(orig_corners[:,0]), np.min(orig_corners[:,1]), 
                       np.max(orig_corners[:,0]), np.max(orig_corners[:,1])]
            
            # 2. 특징점(5개) 좌표 역변환
            # 유효한(0이 아닌) 점만 변환
            valid_idx = np.where((kp[:, 0] > 0) & (kp[:, 1] > 0))[0] # 만족하는 위치의 인덱스 가져옴
            new_kp = np.zeros_like(kp) # kp와 같은 크기지만 0으로 채움
            if len(valid_idx) > 0: # 있으면 유효한 특징점들만 원래 위치로 되돌림
                new_kp[valid_idx] = transform_coords_general(kp[valid_idx], M)
            
            all_faces.append({'box': new_box, 'score': score, 'kpts': new_kp.tolist()})


# --- 3. 중복 제거 및 '최고 점수 특징점' 선택 ---
final_faces = []
all_faces.sort(key=lambda x: x['score'], reverse=True)



for face in all_faces:
    matched = False
    f_box = face['box']
    f_center = [(f_box[0] + f_box[2]) / 2, (f_box[1] + f_box[3]) / 2] # 중심점 
    f_nose = face['kpts'][2] # 코 위치

    # 설정값
    f_w = f_box[2] - f_box[0]
    dist_threshold = f_w * 0.2 # 얼굴 너비의 20%정도

    for i, top_face in enumerate(final_faces):
        t_box = top_face['box']
        t_nose = top_face['kpts'][2] # 코 위치
        
        # 1. IoU 기반
        iou = calculate_iou(f_box, t_box) # 두개의 위치의 IoU
        
        # 2. 중심점 포함 여부
        # 한 박스의 중심이 다른 박스 안에 있다면 거의 100% 동일인
        center_in = is_center_inside(f_center, t_box)
        
        # 3. 코(Nose) 거리 기반
        dist = 10000
        if f_nose[0] > 0 and t_nose[0] > 0:
            dist = calculate_dist(f_nose, t_nose)

        # 세 조건 중 하나라도 만족하면 병합
        if iou > 0.25 or center_in or dist < dist_threshold:
            final_faces[i] = merge_faces(top_face, face)
            matched = True
            break
            
    if not matched:
        final_faces.append(face)

# --- 4. 최종 결과 시각화 ---
for f in final_faces:
    box = f['box']
    kpts = f['kpts']
    # 이미지, 왼쪽 상단 꼭지점 좌표, 오른쪽 하단 꼭지점좌표, 색, 선 두께
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2) 
    for pt in kpts:
        if pt[0] > 0: # 유효한 점만 그리기
            cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1) # 이미지, 중심점, 반지름, 색상, 두께(-1은 채우기)

# 1. '45-Degree Integrated Results'이라는 이름의 창을 먼저 만듭니다.
cv2.namedWindow('45-Degree Integrated Results', cv2.WINDOW_NORMAL)

# 2. 창의 크기를 원하는 대로 조절합니다. (가로, 세로)
#cv2.resizeWindow('45-Degree Integrated Results', 800, 1500)
cv2.imshow('45-Degree Integrated Results', img)
cv2.waitKey(0)
cv2.destroyAllWindows()