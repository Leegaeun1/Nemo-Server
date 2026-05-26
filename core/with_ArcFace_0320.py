import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import os
import glob

try:
    from insightface.app import FaceAnalysis
    import torch
except ImportError:
    print("❌ insightface 또는 torch 라이브러리가 없습니다.")
    exit()

# ==========================================
# 1. 딥러닝 모델 및 도구 로드
# ==========================================
device = 0 if torch.cuda.is_available() else 'cpu'
print(f" 실행 디바이스: {device}")

# YOLO 얼굴 검출기 (트래킹 용)
try:
    face_detector = YOLO('models/yolov12s-face.pt').to(device)
    #face_detector = YOLO('runs/detect/train6/weights/best.pt').to(device)
except Exception as e:
    print(f" YOLO 모델 로드 실패: {e}")
    face_detector = YOLO('models/yolov12s-face.pt') # CPU 폴백

# MediaPipe Face Mesh (고급 블러 용)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=False, min_detection_confidence=0.3)

# ==========================================
# 2. InsightFace 등록 및 가상 데이터 생성
# ==========================================
print("\n🔄 Test_person 폴더의 정면 사진으로 '가상 다각도' 특징을 추출합니다...")
# 'buffalo_l' 모델이 가장 정확도가 높습니다. (최초 실행 시 다운로드)
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# ctx_id=0은 첫 번째 GPU 사용, DET 해상도는 640
face_analyzer.prepare(ctx_id=0 if device == 0 else -1, det_size=(640, 640))

known_embeddings = [] # 내가 설정한 인물 사진
img_paths = glob.glob("Test_person/person6.png")

if not img_paths:
    print("⚠️ Test_person 폴더에 이미지가 없습니다. 모든 얼굴이 블러 처리됩니다.")

for img_path in img_paths:
    try:
        img = cv2.imread(img_path) 
        if img is None: continue
        
        # 1. 정면 사진에서 얼굴 분석
        faces = face_analyzer.get(img)
        
        if not faces:
            print(f"❌ {os.path.basename(img_path)}에서 얼굴을 찾을 수 없습니다.")
            continue
            
        # 가장 큰 얼굴 하나만 등록 (정면 사진이라 가정)
        face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        
        # 2. [가상 3D 상상] InsightFace의 임베딩은 내부적으로 
        #    얼굴 정렬(Alignment) 및 3D 포즈 보정을 거쳐 
        #    정면 사진 한 장으로도 어느 정도의 측면 변위까지 커버하는 강력한 특징을 추출합니다.
        known_embeddings.append(face.normed_embedding) # 정규화된 임베딩 저장
        print(f"✅ {os.path.basename(img_path)} 등록 완료 (강력한 정면/측면 통합 특징추출)")
        
    except Exception as e:
        print(f"❌ {os.path.basename(img_path)} 처리 중 오류: {e}")

# 코사인 유사도 계산 함수 (InsightFace는 임베딩이 정규화되어 있어 단순 내적으로 계산 가능)
def is_same_person(embed1, embed2, threshold=0.4): # InsightFace 권장 Threshold: 0.35~0.45
    distance = np.dot(embed1, embed2) # 코사인 유사도 (1에 가까울수록 같음!)
    # distance는 유사도이므로 threshold보다 '크면' 같은 사람
    return distance > threshold, distance

# ==========================================
# 3. 영상 설정
# ==========================================
video_path = "videos/video_08.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ 영상을 열 수 없습니다: {video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30 

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('outputs/comp_result8.mp4', fourcc, fps, (width, height)) # 저장 

window_name = 'One-Shot Face Recognition Blur'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# ==========================================
# 4. 신원 확인 캐싱 및 설정
# ==========================================
checked_identities = {} # 사람들 저장
# 임계값: InsightFace 'buffalo_l' 모델 기준. 
# 아는 사람인데 블러되면 이 값을 조금 낮추고(예: 0.35),  -> 만약 다시 만들어달라고 하면 이 값을 조절해서 다시 하기?
# 타인이 안 가려지면 이 값을 조금 높이기 (예: 0.45).
SIMILARITY_THRESHOLD = 0.40

print(f"\n🎬 영상 처리를 시작합니다. 'output_one_shot.mp4'로 저장됩니다.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. YOLO 트래킹으로 빠른 얼굴 검출 및 ID 유지
    results = face_detector.track(frame, persist=True, conf=0.3, imgsz=640, device=device, verbose=False)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, f_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            #cv2.rectangle(frame, (max(0,x1), max(0,y1)), (min(frame.shape[1],x2), min(frame.shape[0],y2)), (0, 255, 0), 2)
            #cv2.putText(frame, f"KNOWN:{f_id}", (max(0,x1), max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ---------------------------------------------------------
            # 신원 확인 로직
            # ---------------------------------------------------------
            if f_id not in checked_identities: # 처음 본 사람
                # 1. 얼굴 영역 크롭 (InsightFace가 랜드마크를 잘 찾도록 상하좌우 여백을 30%씩)
                img_h, img_w = frame.shape[:2]
                pad_w = int((x2 - x1) * 0.3)
                pad_h = int((y2 - y1) * 0.3)
                rx1 = max(0, x1 - pad_w)
                ry1 = max(0, y1 - pad_h)
                rx2 = min(img_w, x2 + pad_w)
                ry2 = min(img_h, y2 + pad_h)
                face_crop = frame[ry1:ry2, rx1:rx2]
                
                # 등록된 데이터가 있고 얼굴이 유효할 때만
                if known_embeddings and face_crop.size > 0:
                    try:
                        # 영상 속 얼굴(여백 포함)에서 특징 추출
                        faces_in_crop = face_analyzer.get(face_crop)
                        
                        if faces_in_crop:
                            # 특징을 무사히 뽑았다면 비교 시작
                            target_face = sorted(faces_in_crop, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                            target_embed = target_face.normed_embedding
                            
                            best_sim = -1.0
                            for known_emb in known_embeddings:
                                same, sim = is_same_person(target_embed, known_emb, SIMILARITY_THRESHOLD)
                                if sim > best_sim: # 가장 유사한 점수 저장
                                    best_sim = sim
                            
                            if best_sim > SIMILARITY_THRESHOLD:
                                checked_identities[f_id] = True  # 아는 사람 확정!
                                print(f"🟢 ID {f_id} 인식 성공! (유사도: {best_sim:.3f})")
                            else:
                                checked_identities[f_id] = False # 모르는 사람 확정!
                                print(f"🔴 ID {f_id} 모르는 사람입니다. (최고 유사도: {best_sim:.3f})")
                        else:
                            # InsightFace가 얼굴을 못 찾았다면?
                            # 캐싱하지 않고 넘어갑니다. (다음 프레임에서 얼굴이 더 잘 보일 때 다시 시도함)
                            continue
                            
                    except Exception as e:
                        continue # 에러 나도 다음 프레임에서 재시도
                else:
                    continue
            # ---------------------------------------------------------
            # [블러 처리 로직]
            # ---------------------------------------------------------
            if checked_identities[f_id]:
                # 아는 사람: 표시만 함 (영상 경계 예외처리 추가)
                cv2.rectangle(frame, (max(0,x1), max(0,y1)), (min(frame.shape[1],x2), min(frame.shape[0],y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"KNOWN:{f_id}", (max(0,x1), max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                continue 

            # 모르는 사람/인식 실패: 블러 처리 (Convex Hull 방식)
            img_h, img_w = frame.shape[:2]
            rx1, ry1 = max(0, x1), max(0, y1)
            rx2, ry2 = min(img_w, x2), min(img_h, y2)
            
            roi = frame[ry1:ry2, rx1:rx2]
            blurred_done = False
            
            if roi.size > 0:
                try:
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

                        frame[ry1:ry2, rx1:rx2] = roi
                        blurred_done = True
                except:
                    pass
            
            # Fallback (원형 블러)
            if not blurred_done:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = int(max(x2 - x1, y2 - y1) * 0.5)
                sub_x1, sub_y1 = max(0, center_x - radius), max(0, center_y - radius)
                sub_x2, sub_y2 = min(frame.shape[1], center_x + radius), min(frame.shape[0], center_y + radius)
                face_sub_img = frame[sub_y1:sub_y2, sub_x1:sub_x2]
                if face_sub_img.size > 0:
                    c_mask = np.zeros(face_sub_img.shape[:2], dtype=np.uint8)
                    cv2.circle(c_mask, (center_x - sub_x1, center_y - sub_y1), radius, 255, -1)
                    k_size = 151
                    if face_sub_img.shape[1] < k_size or face_sub_img.shape[0] < k_size:
                        k_size = min(face_sub_img.shape[1], face_sub_img.shape[0]) // 2 * 2 - 1
                    if k_size >= 1:
                        f_blur = cv2.GaussianBlur(face_sub_img, (k_size, k_size), 50)
                        frame[sub_y1:sub_y2, sub_x1:sub_x2] = np.where(c_mask[:,:,None] == 255, f_blur, face_sub_img)
    
    out.write(frame)
    cv2.imshow(window_name, frame)
    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27: break

print("✅ 저장이 완료되었습니다!")
cap.release()
out.release()
cv2.destroyAllWindows()