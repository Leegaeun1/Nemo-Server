import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# 1. 모델 및 도구 로드
face_detector = YOLO('models/yolov11n-face.pt').to('cuda')
mp_face_mesh = mp.solutions.face_mesh
# max_num_faces : 탐지할 최대 얼굴 갯수
# refine_landmarks : 정교 랜드마크. 기본이 False
# min_detection_confidence : 탐지가 성공된 것으로 간주하는 최소 신뢰 값
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.3)

video_path = "Test_video/input_video1.mp4"
cap = cv2.VideoCapture(video_path) # 비디오 실행

# --- 💾 영상 저장 설정 추가 ---
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30 # FPS를 못 읽어올 경우 대비

# 저장할 파일명, 코덱(mp4v), FPS, 해상도 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('Test_video/output_result64.mp4', fourcc, fps, (width, height))

# --- 인터랙티브 설정 ---
unblurred_ids = set() # 블러x id 저장
current_face_locations = {} # 현재 얼굴 위치

def select_face(event, x, y, flags, param): # 얼굴 선택
    global unblurred_ids
    if event == cv2.EVENT_LBUTTONDOWN: # 좌클릭
        for f_id, (x1, y1, x2, y2) in current_face_locations.items(): 
            if x1 <= x <= x2 and y1 <= y <= y2: # 현재 얼굴들 중에 클릭된 위치에 있으면
                if f_id in unblurred_ids: # 블러처리 안돼있음
                    unblurred_ids.remove(f_id) # 블러처리
                else: # 블러처리 돼있음
                    unblurred_ids.add(f_id) # 블러추가
                break

cv2.namedWindow('Nemo Interactive & Saving',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Nemo Interactive & Saving', select_face) # 마우스 클릭시 실행할 콜백 함수

print(f"🎬 영상 처리를 시작합니다. 'output_result.mp4'로 저장됩니다.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = face_detector.track(
        frame, 
        persist=True,
        conf=0.18, 
        imgsz=640,  # 원본 해상도가 커도 모델 입력은 640 정도가 적당합니다.
        device=0,   # GPU 사용 강제
        verbose=False
    )
    new_locations = {} # 새로운 위치

    if results[0].boxes is not None and results[0].boxes.id is not None: # id나 위치정보가 존재
        boxes = results[0].boxes.xyxy.cpu().numpy() # 해당 박스 위치정보
        ids = results[0].boxes.id.int().cpu().tolist() # 해당 id
        
        for box, f_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            new_locations[f_id] = (x1, y1, x2, y2) # id의 위치는 x1,y1,x2,y2라고 저장
            
            if f_id in unblurred_ids: # 블러처리되지 않음 
                cv2.putText(frame, f"OPEN ID:{f_id}", (x1, y1-10), 0, 0.6, (0, 255, 0), 2)
                continue 

            # 블러 처리 로직 (Convex Hull 방식)
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)] # 원하는 위치 지정
            blurred_done = False # 블러 끝났는지 확인
            if roi.size > 0:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) # bgr을 rgb로 지정. mediapipe는 rgb
                mesh_results = face_mesh.process(rgb_roi) # 얼굴 특징점 검출 
                if mesh_results.multi_face_landmarks: # x,y,z로 검출됨. x는 폭, y는 높이로 0.0~1.0로 정규화 -> 실제 픽셀 좌표로 변환
                    h, w, _ = roi.shape # 높이, 너비
                    all_points = [(int(lm.x * w), int(lm.y * h)) for lm in mesh_results.multi_face_landmarks[0].landmark]

                    hull = cv2.convexHull(np.array(all_points)) # 오목한 도형 -> 볼록한 도형
                    mask = np.zeros((h, w), dtype=np.uint8) # 까만 행렬 생성
                    cv2.fillConvexPoly(mask, hull, 255) # 흰색으로 채워진 볼록 다각형 그림 

                    roi_blur = cv2.GaussianBlur(roi, (121, 121), 40) # 강한 가우시안 블러
                    # 마스크가 흰색인 곳을 블러, 아니면 원본 선택하여 합성
                    roi = np.where(mask[:, :, None] == 255, roi_blur, roi)

                    # 원본 프레임에 결과 덮어쓰기
                    frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)] = roi
                    blurred_done = True
            
            if not blurred_done: # Fallback 원형 블러
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2 # 중심점
                radius = int(max(x2 - x1, y2 - y1) * 0.5) # 반지름
                sub_x1, sub_y1 = max(0, center_x - radius), max(0, center_y - radius) # 원이 그려질 영역 좌표 계산
                sub_x2, sub_y2 = min(frame.shape[1], center_x + radius), min(frame.shape[0], center_y + radius)
                face_sub_img = frame[sub_y1:sub_y2, sub_x1:sub_x2] # 해당 영역만 잘라냄
                if face_sub_img.size > 0:
                    c_mask = np.zeros(face_sub_img.shape[:2], dtype=np.uint8) # 원형 마스크 생성
                    cv2.circle(c_mask, (center_x - sub_x1, center_y - sub_y1), radius, 255, -1) # 흰색으로 원 그림
                    f_blur = cv2.GaussianBlur(face_sub_img, (151, 151), 50) # 가우시안블러 강하게
                    frame[sub_y1:sub_y2, sub_x1:sub_x2] = np.where(c_mask[:,:,None] == 255, f_blur, face_sub_img) # 흰부분만 블러적용

    current_face_locations = new_locations
    
    # --- 한 프레임씩 파일에 쓰기 ---
    out.write(frame)
    
    cv2.imshow('Nemo Interactive & Saving', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- 리소스 해제 ---
print("✅ 저장이 완료되었습니다!")
cap.release()
out.release() # 파일 닫기 필수!
cv2.destroyAllWindows()