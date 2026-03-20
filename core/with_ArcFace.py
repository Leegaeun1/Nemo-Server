import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import os
import glob
from deepface import DeepFace

# ==========================================
# 1. 딥러닝 모델 및 도구 로드
# ==========================================
face_detector = YOLO('models/yolov11n-face.pt').to('cuda')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.3)

# ==========================================
# 2. 사전 등록된 얼굴(Test_img) 임베딩 추출 (ArcFace)
# ==========================================
print("🔄 Test_person 폴더의 얼굴 특징을 추출하는 중입니다...")
known_embeddings = []
# Test_img 폴더 내의 jpg, png 파일 모두 검색
img_paths = glob.glob("Test_person/*.jpg") + glob.glob("Test_person/*.png") + glob.glob("Test_person/*.jpeg")

for img_path in img_paths:
    try:
        # ArcFace 모델을 사용하여 얼굴의 특징점(임베딩) 추출
        result = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=True)
        known_embeddings.append(result[0]["embedding"])
        print(f"✅ {os.path.basename(img_path)} 등록 완료")
    except Exception as e:
        print(f"❌ {os.path.basename(img_path)} 얼굴 인식 실패 (얼굴이 명확하지 않을 수 있음)")

# 두 임베딩 벡터 간의 거리를 계산하는 함수 (코사인 거리)
def calculate_cosine_distance(emb1, emb2):
    a = np.array(emb1)
    b = np.array(emb2)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ==========================================
# 3. 영상 설정
# ==========================================
video_path = "Test_video/input_video1.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30 

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('Test_video/output_result.mp4', fourcc, fps, (width, height))

cv2.namedWindow('Auto Face Blur', cv2.WINDOW_NORMAL)

# ==========================================
# 4. 신원 확인 캐싱 딕셔너리
# ==========================================
# YOLO ID를 키로, 알고 있는 사람인지 여부(True/False)를 값으로 저장
# 매 프레임 ArcFace를 돌리면 너무 느려지므로 최초 1회만 계산하고 저장합니다.
checked_identities = {} 
ARCFACE_THRESHOLD = 0.60 # 이 값이 작을수록 엄격하게 검사 (보통 0.6 ~ 0.68 사이 권장)

print(f"🎬 영상 처리를 시작합니다. 'output_result.mp4'로 저장됩니다.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = face_detector.track(
        frame, 
        persist=True,
        conf=0.18, 
        imgsz=640,  
        device=0,   
        verbose=False
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, f_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            
            # ---------------------------------------------------------
            # [신원 확인 로직] 해당 ID가 처음 등장한 얼굴이면 ArcFace 확인
            # ---------------------------------------------------------
            if f_id not in checked_identities:
                # 1. 얼굴 영역 크롭 (여백을 조금 주면 인식률이 올라갑니다)
                face_crop = frame[max(0, y1-10):min(frame.shape[0], y2+10), max(0, x1-10):min(frame.shape[1], x2+10)]
                
                is_known_person = False
                if face_crop.size > 0:
                    try:
                        # 영상 속 얼굴의 임베딩 추출 (enforce_detection=False로 영상 내 흔들린 얼굴도 처리)
                        target_result = DeepFace.represent(img_path=face_crop, model_name="ArcFace", enforce_detection=False)
                        target_embedding = target_result[0]["embedding"]
                        
                        # Test_img의 얼굴들과 비교
                        for known_emb in known_embeddings:
                            distance = calculate_cosine_distance(target_embedding, known_emb)
                            if distance < ARCFACE_THRESHOLD:
                                is_known_person = True # 동일인 발견!
                                break
                    except Exception as e:
                        pass # 추출 실패 시 기본값인 False(모르는 사람) 유지
                
                # 결과 저장 (캐싱)
                checked_identities[f_id] = is_known_person
            
            # ---------------------------------------------------------
            # [블러 처리 로직]
            # ---------------------------------------------------------
            # 알고 있는 사람이면 초록색 테두리/텍스트만 띄우고 블러 생략
            if checked_identities[f_id]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"KNOWN:{f_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                continue 

            # 모르는 사람이면 블러 처리 (기존 작성하신 Convex Hull 방식 그대로)
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            blurred_done = False
            
            if roi.size > 0:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(rgb_roi)
                
                if mesh_results.multi_face_landmarks:
                    h, w, _ = roi.shape
                    all_points = [(int(lm.x * w), int(lm.y * h)) for lm in mesh_results.multi_face_landmarks[0].landmark]

                    hull = cv2.convexHull(np.array(all_points))
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, hull, 255)

                    roi_blur = cv2.GaussianBlur(roi, (121, 121), 40)
                    roi = np.where(mask[:, :, None] == 255, roi_blur, roi)

                    frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)] = roi
                    blurred_done = True
            
            # Fallback 원형 블러
            if not blurred_done:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = int(max(x2 - x1, y2 - y1) * 0.5)
                sub_x1, sub_y1 = max(0, center_x - radius), max(0, center_y - radius)
                sub_x2, sub_y2 = min(frame.shape[1], center_x + radius), min(frame.shape[0], center_y + radius)
                face_sub_img = frame[sub_y1:sub_y2, sub_x1:sub_x2]
                
                if face_sub_img.size > 0:
                    c_mask = np.zeros(face_sub_img.shape[:2], dtype=np.uint8)
                    cv2.circle(c_mask, (center_x - sub_x1, center_y - sub_y1), radius, 255, -1)
                    f_blur = cv2.GaussianBlur(face_sub_img, (151, 151), 50)
                    frame[sub_y1:sub_y2, sub_x1:sub_x2] = np.where(c_mask[:,:,None] == 255, f_blur, face_sub_img)
    
    out.write(frame)
    cv2.imshow('Auto Face Blur', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

print("✅ 저장이 완료되었습니다!")
cap.release()
out.release()
cv2.destroyAllWindows()