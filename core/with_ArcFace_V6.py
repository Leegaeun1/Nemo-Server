import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import os
import glob
import time
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==========================================
# InsightFace 모듈 로드
# ==========================================
try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("❌ insightface 라이브러리가 없습니다. (pip install insightface)")
    exit()

# ==========================================
# 1. 딥러닝 모델 초기화
# ==========================================
print(f"🚀 실행 디바이스: GPU/CPU 자동 할당")

# 1-1. YOLO (트래킹 및 빠른 검출)
try:
    face_detector = YOLO('models/yolov11n-face.pt')
except:
    face_detector = YOLO('yolov8n-face.pt')

# 1-2. MediaPipe (모자이크 및 랜드마크 추출용)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=False, min_detection_confidence=0.3)
static_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)

# 1-3. InsightFace
print("🔄 InsightFace (인식기 포함) 로드 중...")
face_analyzer = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640)) # 서버에 맞게 ctx_id 설정 (GPU면 0)
print("✅ InsightFace 로드 성공")

# 완벽한 코사인 유사도 함수 (L2 정규화 강제 적용)
def is_same_person(embed1, embed2, threshold=0.4):
    e1 = embed1.flatten() / (np.linalg.norm(embed1) + 1e-8)
    e2 = embed2.flatten() / (np.linalg.norm(embed2) + 1e-8)
    distance = np.dot(e1, e2)
    return distance > threshold, distance

# 얼굴 위에 안경 PNG 동적 합성 (크기 일정화, 회전 잘림 방지, 클리핑 적용)
def overlay_glasses(img, glasses_bgra, face_landmarks):
    h, w = img.shape[:2]
    left_eye_indices = [33, 133, 159, 145] # 왼쪽 눈 주변 점들
    right_eye_indices = [263, 362, 386, 374] # 오른쪽 눈 주변 점들
    
    # 1. 랜드마크에서 양쪽 눈 좌표 추출
    lx = int(sum(face_landmarks.landmark[i].x for i in left_eye_indices) / len(left_eye_indices) * w)
    ly = int(sum(face_landmarks.landmark[i].y for i in left_eye_indices) / len(left_eye_indices) * h)
    
    rx = int(sum(face_landmarks.landmark[i].x for i in right_eye_indices) / len(right_eye_indices) * w)
    ry = int(sum(face_landmarks.landmark[i].y for i in right_eye_indices) / len(right_eye_indices) * h)
    dx, dy = rx - lx, ry - ly
    eye_dist = math.sqrt(dx**2 + dy**2)
    angle = -math.degrees(math.atan2(dy, dx))
    
    # ---------------------------------------------------------
    # 알파 채널(투명도)을 검사하여 실제 안경이 있는 픽셀 영역만 타이트하게 잘라냅니다.
    # ---------------------------------------------------------
    alpha_channel = glasses_bgra[:, :, 3]
    coords = cv2.findNonZero(alpha_channel)
    if coords is not None:
        x, y, gw, gh = cv2.boundingRect(coords)
        cropped_glasses = glasses_bgra[y:y+gh, x:x+gw]
    else:
        cropped_glasses = glasses_bgra
        
    # 여백이 제거되었으므로, 실제 안경 폭이 눈 사이 거리의 약 2.4배가 되도록 설정
    scale = (eye_dist * 2.4) / cropped_glasses.shape[1]
    g_w = int(cropped_glasses.shape[1] * scale)
    g_h = int(cropped_glasses.shape[0] * scale)
    if g_w == 0 or g_h == 0: return img
    resized_glasses = cv2.resize(cropped_glasses, (g_w, g_h))
    
    # ---------------------------------------------------------
    # 회전 후 커지는 대각선 길이를 계산하여 캔버스(Bounding Box) 크기를 늘려줍니다.
    # ---------------------------------------------------------
    center = (g_w // 2, g_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos_a = np.abs(M[0, 0])
    sin_a = np.abs(M[0, 1])
    # 회전 후의 새로운 캔버스 가로, 세로 크기 계산
    new_w = int((g_h * sin_a) + (g_w * cos_a))
    new_h = int((g_h * cos_a) + (g_w * sin_a))
    
    # 늘어난 캔버스 크기에 맞춰 회전 중심점 재조정
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 확장된 크기(new_w, new_h)로 회전 적용 -> 잘림 방지!
    rotated_glasses = cv2.warpAffine(resized_glasses, M, (new_w, new_h), 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(0,0,0,0))
    
    # 3. 캔버스 위에 렌더링할 시작 좌표 계산 (눈의 중심 기준)
    cx, cy = (lx + rx) // 2, (ly + ry) // 2
    
    # 확장된 크기(new_w, new_h)를 사용하여 좌표 세팅
    x1 = cx - new_w // 2
    y1 = cy - int(new_h * 0.45) # Y축 기준점 (안경테 두께에 따라 0.4 ~ 0.5 사이 조절 가능)
    x2, y2 = x1 + new_w, y1 + new_h
    
    # ---------------------------------------------------------
    # 경계선 클리핑 (화면 밖으로 나가는 부분 처리)
    # ---------------------------------------------------------
    img_y1 = max(0, y1)
    img_y2 = min(h, y2)
    img_x1 = max(0, x1)
    img_x2 = min(w, x2)

    glass_y1 = max(0, -y1)
    glass_y2 = new_h - max(0, y2 - h)
    glass_x1 = max(0, -x1)
    glass_x2 = new_w - max(0, x2 - w)

    if img_y1 >= img_y2 or img_x1 >= img_x2:
        return img 
        
    result = img.copy()
    GLASSES_OPACITY = 0.80
    # 기존 알파 채널 값에 투명도(OPACITY)를 곱해줍니다.
    alpha_s = (rotated_glasses[glass_y1:glass_y2, glass_x1:glass_x2, 3] / 255.0) * GLASSES_OPACITY
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        result[img_y1:img_y2, img_x1:img_x2, c] = (
            alpha_s * rotated_glasses[glass_y1:glass_y2, glass_x1:glass_x2, c] + 
            alpha_l * result[img_y1:img_y2, img_x1:img_x2, c]
        )
        
    return result

# ==========================================
# 2. 인물 등록 (가상 템플릿 증강)
# ==========================================
print("\n🔄 인물 등록 및 가상 안경 템플릿 생성 시작...")
known_embeddings = []

# 가상 안경 파일 로드
glasses_paths = glob.glob("glasses/*.png")
glasses_list = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in glasses_paths]
glasses_list = [g for g in glasses_list if g is not None and g.shape[2] == 4]
print(f"👓 준비된 가상 안경: {len(glasses_list)}개")


# 합성 결과를 저장할 폴더 생성 (디버깅용)
debug_dir = "augmented_faces"
os.makedirs(debug_dir, exist_ok=True)
print(f"📁 합성 이미지는 '{debug_dir}' 폴더에 저장됩니다.")


img_paths = glob.glob("Test_person/rei3.jpg")

for img_path in img_paths:
    img = cv2.imread(img_path) 
    if img is None: continue
    
    # [Step 1] 원본 맨얼굴 등록
    faces = face_analyzer.get(img)
    if not faces:
        print(f"❌ {os.path.basename(img_path)}에서 얼굴을 찾을 수 없습니다.")
        continue
        
    target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    known_embeddings.append(target_face.embedding)
    print(f"✅ {os.path.basename(img_path)} 원본 등록 완료")

    # [Step 2] 가상 안경 합성 및 저장
    if len(glasses_list) > 0:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_results = static_face_mesh.process(img_rgb)
        
        if mp_results.multi_face_landmarks:
            landmarks = mp_results.multi_face_landmarks[0]
            successful_augs = 0
            
            # MediaPipe 랜드마크 인식 상태 시각화 저장
            debug_img = img.copy()
            mp_drawing.draw_landmarks(
                image=debug_img,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # 눈동자 위치(33, 263)에 빨간 점 찍기
            h, w = img.shape[:2]
            lx, ly = int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h)
            rx, ry = int(landmarks.landmark[263].x * w), int(landmarks.landmark[263].y * h)
            cv2.circle(debug_img, (lx, ly), 5, (0, 0, 255), -1)
            cv2.circle(debug_img, (rx, ry), 5, (0, 0, 255), -1)
            
            debug_save_path = os.path.join(debug_dir, f"debug_mesh_{os.path.basename(img_path)}")
            cv2.imwrite(debug_save_path, debug_img)
            print(f"  🔍 [디버그] MediaPipe 랜드마크 저장 완료: {debug_save_path}")
            
            for i, glasses_bgra in enumerate(glasses_list):
                # 우리의 함수로 안경 씌우기
                augmented_img = overlay_glasses(img, glasses_bgra, landmarks)
                
                # 합성된 결과물을 폴더에 저장!
                save_name = f"aug_{i:02d}_{os.path.basename(img_path)}"
                save_path = os.path.join(debug_dir, save_name)
                cv2.imwrite(save_path, augmented_img)
                
                # 원본 이미지와 픽셀 단위로 동일하면 합성이 실패한 것
                if np.array_equal(img, augmented_img):
                    print(f"  ❌ [디버그] {save_name} 안경 합성 실패 (원본 반환됨)")
                    continue
                
                # InsightFace 임베딩도 추가 등록
                aug_faces = face_analyzer.get(augmented_img)
                if aug_faces:
                    target_aug_face = sorted(aug_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    known_embeddings.append(target_aug_face.embedding)
                    successful_augs += 1
                    
            print(f"✅ {os.path.basename(img_path)} 가상 안경 템플릿 {successful_augs}개 추가 등록 및 저장 완료")
        else:
            # MediaPipe가 얼굴을 전혀 찾지 못한 경우
            print(f"  ❌ [디버그] MediaPipe가 {os.path.basename(img_path)}에서 얼굴을 찾지 못했습니다!")

# ==========================================
# 3. 영상 설정 및 신원 확인 메인 루프
# ==========================================
video_path = os.path.abspath("Test_video/test_video6.mp4")
cap = cv2.VideoCapture(video_path)

out = cv2.VideoWriter('Test_video/output_hybrid_rei5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                      cap.get(cv2.CAP_PROP_FPS) or 30, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 임계값을 0.48로 올려서 타인을 확실히 걸러냅니다.
SIMILARITY_THRESHOLD = 0.40 
print("\n🔍 [1/2 단계] 영상 전체를 스캔하며 신원을 파악 중입니다...")

identity_votes = {}    
VOTE_REQUIREMENT = 3
last_seen_frame = {} # ID가 마지막으로 등장한 프레임 기록
active_known_ids = set() # 현재 '등록된 인물'로 인정받고 있는 ID 목록

frame_tracking_data = [] 
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_idx += 1

    results = face_detector.track(frame, persist=True, conf=0.3, imgsz=640, verbose=False)
    
    current_boxes = []
    current_ids = []
    current_known_status = [] # 이번 프레임의 각 인물별 인증 상태 (True/False)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        current_boxes = results[0].boxes.xyxy.cpu().numpy()
        current_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, f_id in zip(current_boxes, current_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # ID가 1초(30프레임) 이상 사라졌다가 나타나면 이전 투표 기록 싹 초기화
            if f_id in last_seen_frame and (frame_idx - last_seen_frame[f_id] > 30):
                identity_votes[f_id] = 0
                if f_id in active_known_ids:
                    active_known_ids.remove(f_id)
            last_seen_frame[f_id] = frame_idx

            is_known = False
            best_sim = -1.0
            
            # 무조건 continue 하지 않고, 15프레임(0.5초)마다 주기적으로 재검사!
            needs_check = True
            if f_id in active_known_ids:
                if frame_idx % 15 != 0: 
                    needs_check = False # 0.5초 사이에는 검사 생략 (연산량 절약)
                    is_known = True

            if needs_check:
                try:
                    img_h, img_w = frame.shape[:2]
                    pad_w, pad_h = int((x2 - x1) * 0.3), int((y2 - y1) * 0.3)
                    rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                    rx2, ry2 = min(img_w, x2 + pad_w), min(img_h, y2 + pad_h)
                    
                    face_crop = frame[ry1:ry2, rx1:rx2]
                    
                    if face_crop.size > 0:
                        faces_in_crop = face_analyzer.get(face_crop)
                        
                        if faces_in_crop:
                            target_face = sorted(faces_in_crop, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                            target_embed = target_face.embedding
                            
                            for i, known_emb in enumerate(known_embeddings):
                                same, sim = is_same_person(target_embed, known_emb, SIMILARITY_THRESHOLD)
                                if sim > best_sim: 
                                    best_sim = sim
                            
                            # 검사 결과 처리
                            if best_sim > SIMILARITY_THRESHOLD:
                                identity_votes[f_id] = identity_votes.get(f_id, 0) + 1
                                if identity_votes[f_id] >= VOTE_REQUIREMENT:
                                    active_known_ids.add(f_id)
                                    is_known = True
                                    print(f"👍 ID {f_id} 인증 확인!{i}번째 안경 (유사도: {best_sim:.3f})")
                            else:
                                # 핵심: 유사도가 낮으면 엉뚱한 사람이므로 즉시 예외 권한 몰수 및 투표 초기화
                                identity_votes[f_id] = 0
                                if f_id in active_known_ids:
                                    active_known_ids.remove(f_id)
                                    print(f"🔴 ID {f_id} 타인 감지 -> 모자이크 예외 해제!")
                except Exception as e:
                    pass
            
            # 이 프레임에서 이 박스가 모자이크 대상인지 기록
            current_known_status.append(is_known)
                
    # 렌더링을 위해 전체 데이터 기록 (박스 좌표, ID, 상태)
    frame_tracking_data.append((current_boxes, current_ids, current_known_status))

print("✅ 분석 완료! 렌더링을 시작합니다.")

# ==========================================
# [Pass 2] 최종 렌더링 및 비디오 저장 단계
# ==========================================
print("\n🎬 [2/2 단계] 최종 영상을 렌더링하여 저장합니다...")

cap.release()
time.sleep(0.5) 

cap = cv2.VideoCapture(video_path)
frame_idx = 0

if not cap.isOpened():
    print(f"❌ 2단계 영상 재열기 실패! 경로를 확인하세요: {video_path}")

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame_idx >= len(frame_tracking_data): 
        break

    # 저장된 current_known_status를 꺼내서 프레임 단위로 안전하게 렌더링
    boxes, ids, known_statuses = frame_tracking_data[frame_idx]
    
    for box, f_id, is_known in zip(boxes, ids, known_statuses):
        x1, y1, x2, y2 = map(int, box)
        
        if is_known:
            # 등록된 본인: 초록색 박스로 표시하고 모자이크 생략
            cv2.rectangle(frame, (max(0,x1), max(0,y1)), (min(frame.shape[1],x2), min(frame.shape[0],y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"KNOWN:{f_id}", (max(0,x1), max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 타인: 모자이크 처리 수행
            img_h, img_w = frame.shape[:2]
            rx1, ry1 = max(0, x1), max(0, y1)
            rx2, ry2 = min(img_w, x2), min(img_h, y2)
            roi = frame[ry1:ry2, rx1:rx2]
            
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
                    else:
                        raise Exception("MediaPipe fallback")
                except:
                    center_x, center_y = (rx1 + rx2) // 2, (ry1 + ry2) // 2
                    radius = int(max(rx2 - rx1, ry2 - ry1) * 0.5)
                    c_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.circle(c_mask, (roi.shape[1]//2, roi.shape[0]//2), radius, 255, -1)
                    k_size = max(1, min(roi.shape[:2]) // 2 * 2 - 1)
                    f_blur = cv2.GaussianBlur(roi, (k_size, k_size), 50)
                    frame[ry1:ry2, rx1:rx2] = np.where(c_mask[:,:,None] == 255, f_blur, roi)

    out.write(frame)
    frame_idx += 1
    
    if frame_idx % 30 == 0:
        print(f"렌더링 진행 중... ({frame_idx} 프레임 완료)")

print("✅ 모든 처리가 완료되었습니다!")
cap.release()
out.release()
cv2.destroyAllWindows()