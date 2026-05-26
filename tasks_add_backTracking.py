import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import math
import subprocess
import shutil

# ==========================================
# 로컬 테스트용 설정 
# ==========================================
LOCAL_TEST_MODE = True          # True: 로컬 직접 실행 / False: Celery 서버 모드
LOCAL_VIDEO_PATH  = "videos/video_08.mp4"   # 처리할 동영상 경로
LOCAL_FACES_DIR   = "Test_person"   # 등록 얼굴 사진 폴더 (1명)
LOCAL_OUTPUT_PATH = "outputs/test8.mp4"     # 결과물 저장 경로

# ==========================================
# Firebase
# ==========================================
if not LOCAL_TEST_MODE:
    import firebase_admin
    from firebase_admin import credentials, messaging
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            print("Firebase Admin 초기화 완료")
        except Exception as e:
            print(f"Firebase Admin 초기화 실패: {e}")

# ==========================================
# HeatmapEngine
# ==========================================
class HeatmapEngine:
    def __init__(self, frame_width, frame_height, grid_scale=0.1):
        self.grid_w = int(frame_width * grid_scale)
        self.grid_h = int(frame_height * grid_scale)
        self.scale = grid_scale
        self.channels = {}
        self.last_sizes = {}
        self.last_centers = {}
        self.base_decay = 0.96

    def reset_memory(self):
        self.channels.clear()
        self.last_sizes.clear()
        self.last_centers.clear()

    def get_visual_heatmap(self): # 히트맵 시각화용 컬러맵으로 반환
        combined_heat = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for f_id, heatmap in self.channels.items():
            combined_heat = np.maximum(combined_heat, heatmap)
        normalized_heat = np.clip(combined_heat / 10.0 * 255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized_heat, cv2.COLORMAP_JET)
        gray = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(colored_heatmap, colored_heatmap, mask=mask)

    def update_and_recover(self, boxes, ids, confs): # 매 프레임마다 호출됨!
        ''' 1. 기존 모든 채널에 decay(0.96) 적용 → 오래된 기억을 서서히 지움
            2. 새로 들어온 박스마다:
            - 신규 ID이면 → 히트맵에서 같은 위치에 열기가 있는 old_id와 매칭 시도
            - conf가 낮거나 박스가 갑자기 작아지면 → "부분 가려짐"으로 판단
            - 이전 위치/크기와 0.7:0.3 비율로 스무딩 (갑작스러운 움직임 완화)
            3. 히트맵 채널 업데이트 (해당 위치에 +2.0 가중치)
            4. 오랫동안 보이지 않아 heat가 0.5 이하가 된 채널은 삭제'''
        for f_id in list(self.channels.keys()):
            self.channels[f_id] *= self.base_decay

        final_boxes = []
        final_ids = []
        seen_ids = set()

        for box, f_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            hcx, hcy = int(cx * self.scale), int(cy * self.scale)

            matched_id = f_id
            if f_id not in self.channels:
                best_id = None
                best_heat = 0.2
                for old_id, heatmap in self.channels.items():
                    if 0 <= hcy < self.grid_h and 0 <= hcx < self.grid_w:
                        if heatmap[hcy, hcx] > best_heat:
                            best_heat = heatmap[hcy, hcx]
                            best_id = old_id
                if best_id is not None:
                    matched_id = best_id

            if matched_id in seen_ids:
                continue
            seen_ids.add(matched_id)

            is_full_face = True
            if conf < 0.45:
                is_full_face = False
            elif matched_id in self.last_sizes:
                prev_w, prev_h = self.last_sizes[matched_id]
                if w < prev_w * 0.8 or h < prev_h * 0.8:
                    is_full_face = False

            if matched_id in self.last_sizes:
                if is_full_face:
                    prev_w, prev_h = self.last_sizes[matched_id]
                    pcx, pcy = self.last_centers[matched_id]
                    self.last_sizes[matched_id] = (int(prev_w * 0.7 + w * 0.3), int(prev_h * 0.7 + h * 0.3))
                    self.last_centers[matched_id] = (int(pcx * 0.7 + cx * 0.3), int(pcy * 0.7 + cy * 0.3))
            else:
                self.last_sizes[matched_id] = (w, h)
                self.last_centers[matched_id] = (cx, cy)

            curr_cx, curr_cy = self.last_centers[matched_id]
            curr_w, curr_h = self.last_sizes[matched_id]

            if matched_id not in self.channels:
                self.channels[matched_id] = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

            weight = 2.0
            gx1 = max(0, int((curr_cx - curr_w / 2) * self.scale))
            gy1 = max(0, int((curr_cy - curr_h / 2) * self.scale))
            gx2 = min(self.grid_w, int((curr_cx + curr_w / 2) * self.scale))
            gy2 = min(self.grid_h, int((curr_cy + curr_h / 2) * self.scale))

            self.channels[matched_id][gy1:gy2, gx1:gx2] += weight
            self.channels[matched_id] = np.clip(self.channels[matched_id], 0, 10.0)

            final_boxes.append([curr_cx - curr_w // 2, curr_cy - curr_h // 2,
                                 curr_cx + curr_w // 2, curr_cy + curr_h // 2])
            final_ids.append(matched_id)

        missing_ids = set(self.channels.keys()) - seen_ids
        for f_id in missing_ids:
            heatmap = self.channels[f_id]
            if np.max(heatmap) <= 0.5:   # decay 다 됐으면 채널 삭제
                del self.channels[f_id]
                if f_id in self.last_centers: del self.last_centers[f_id]
                if f_id in self.last_sizes: del self.last_sizes[f_id]

        return final_boxes, final_ids


# ==========================================
# 모델 로드
# ==========================================
try:
    from core.net import build_model
except ImportError:
    print("⚠️ AdaFace 'net.py'가 없습니다.")

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError:
    print("⚠️ insightface 라이브러리가 없습니다.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    face_detector = YOLO('models/yolov12s-face.pt').to(device)  # 얼굴 탐지 + 트래킹
except:
    face_detector = YOLO('models/yolov12n-face.pt').to(device)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=False, min_detection_confidence=0.3) # 랜드마크 (현재는 사용안함!)
static_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                          refine_landmarks=False, min_detection_confidence=0.4)

try:
    face_aligner = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'],
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) # insightface 정렬
    face_aligner.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
except Exception as e:
    print(f"face_aligner 로드 에러: {e}")

adaface_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

adaface_model = build_model('ir_50').to(device) # 얼굴 임베딩 추출.
try:
    checkpoint = torch.load('adaface_ir50_ms1mv2.ckpt', map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    adaface_model.load_state_dict(state_dict, strict=False)
    adaface_model.eval()
except:
    print("⚠️ adaface checkpoint를 찾지 못했습니다.")


def is_same_person(embed1, embed2, threshold=0.4): # 두 임베딩의 코사인 유사도 계산! 0.4이상이면 같은 사람으로 판정함.
    e1 = embed1.flatten() / (np.linalg.norm(embed1) + 1e-8)
    e2 = embed2.flatten() / (np.linalg.norm(embed2) + 1e-8)
    distance = np.dot(e1, e2)
    return distance > threshold, distance


def send_push_notification(token, output_path):
    if LOCAL_TEST_MODE:
        return
    print(f"발송 시도 시작! (파일명: {os.path.basename(output_path)})")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    file_size_str = f"{file_size_mb:.1f}MB"

    if not token:
        print("토큰(token)이 비어있습니다!")
        return
    if not firebase_admin._apps:
        print("Firebase Admin이 초기화되지 않았습니다!")
        return

    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title='NEMO 비식별화 완료!',
                body='비식별화 처리가 끝났습니다! 앱으로 들어와서 다운로드해주세요.',
            ),
            data={
                "output_filename": os.path.basename(output_path),
                "file_size": file_size_str,
            },
            token=token,
        )
        response = messaging.send(message)
        print(f"[알림 성공] 서버 응답: {response}")
    except Exception as e:
        print(f"[알림 실패] 실제 에러 발생: {e}")


# ==========================================
# 역방향 패스: 위치 히트맵만 수집 (신원 판단 없음)
# ==========================================
def run_reverse_pass(input_path, total_frames, frame_width, frame_height):
    """
    영상을 역방향으로 읽으며 YOLO 트래킹 → 히트맵 위치 정보 수집.
    반환값: reverse_heatmap_list[frame_idx] = [(cx, cy, w, h), ...]
    신원 판단은 하지 않고 "얼굴이 있었던 위치"만 기록.
    """
    print("🔄 역방향 패스 시작...")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ 역방향 패스: 비디오를 열 수 없습니다.")
        return [[] for _ in range(total_frames)]

    # 전체 프레임을 메모리에 역순으로 읽기
    # (영상이 매우 길면 아래 seek 방식으로 대체 가능)
    frames_reversed = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames_reversed.append(frame)
    cap.release()

    frames_reversed.reverse()  # 역순으로 뒤집기

    heatmap_engine_rev = HeatmapEngine(frame_width, frame_height, grid_scale=0.1)
    prev_frame = None
    scene_change_threshold = 30.0

    # 역방향 결과를 역방향 인덱스 기준으로 저장
    rev_results = []  # rev_results[rev_idx] = [(cx,cy,w,h), ...]

    for rev_idx, frame in enumerate(frames_reversed):
        if prev_frame is not None:
            small_curr = cv2.resize(frame, (64, 64))
            small_prev = cv2.resize(prev_frame, (64, 64))
            diff = cv2.absdiff(small_curr, small_prev)
            if np.mean(diff) > scene_change_threshold:
                heatmap_engine_rev.reset_memory()

        prev_frame = frame.copy()

        results = face_detector.track(frame, persist=True, conf=0.30, imgsz=640,
                                       device=device, verbose=False)
        current_boxes, current_ids, current_confs = [], [], []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            current_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            current_ids   = results[0].boxes.id.int().cpu().tolist()
            current_confs = results[0].boxes.conf.cpu().tolist()

        all_boxes, all_ids = heatmap_engine_rev.update_and_recover(
            current_boxes, current_ids, current_confs)

        face_positions = []
        for box in all_boxes:
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w  = x2 - x1
            h  = y2 - y1
            face_positions.append((cx, cy, w, h))

        rev_results.append(face_positions)

        if rev_idx % 100 == 0:
            print(f"  역방향 {rev_idx}/{total_frames} 프레임 처리 중...")

    # rev_results는 역방향 순서 → 정방향 인덱스로 뒤집기
    rev_results.reverse()

    # 길이를 total_frames에 맞춤
    while len(rev_results) < total_frames:
        rev_results.append([])

    print(f"✅ 역방향 패스 완료 ({len(rev_results)} 프레임)")
    return rev_results


# ==========================================
# 모자이크 적용 헬퍼
# ==========================================
def apply_mosaic(frame, box):
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(img_w, x2), min(img_h, y2)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return

    center_x = (rx1 + rx2) // 2
    center_y = (ry1 + ry2) // 2
    box_w = rx2 - rx1
    box_h = ry2 - ry1
    radius = int(max(box_w, box_h) * 0.6)

    cx_min = max(0, center_x - radius)
    cy_min = max(0, center_y - radius)
    cx_max = min(img_w, center_x + radius)
    cy_max = min(img_h, center_y + radius)
    circle_roi = frame[cy_min:cy_max, cx_min:cx_max]

    if circle_roi.size > 0:
        c_mask = np.zeros(circle_roi.shape[:2], dtype=np.uint8)
        cv2.circle(c_mask, (center_x - cx_min, center_y - cy_min), radius, 255, -1)
        f_blur = cv2.GaussianBlur(circle_roi, (121, 121), 50)
        frame[cy_min:cy_max, cx_min:cx_max] = np.where(
            c_mask[:, :, None] == 255, f_blur, circle_roi)


# ==========================================
# 메인 처리 함수
# ==========================================
def process_video(input_path, output_path, user_faces_dir, device_token=None, user_id=None):
    print(f"🎬 영상 처리 시작: {input_path}")

    # 등록 얼굴 임베딩 로드
    known_embeddings = []
    #face_files = glob.glob(f"{user_faces_dir}/*.*")
    face_files = glob.glob(f"{user_faces_dir}/person6.png")
    print(f"👤 등록된 얼굴 {len(face_files)}개 로드")
    for img_path in face_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = face_aligner.get(img)
        if not faces:
            continue
        target_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                              reverse=True)[0]
        aligned_bgr = face_align.norm_crop(img, landmark=target_face.kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = adaface_model(input_tensor)[0].cpu().numpy()
        known_embeddings.append(embedding)
    print(f"✅ known_embeddings 수집 완료: {len(known_embeddings)}개")
    if len(known_embeddings) == 0:
        print("❌ 등록 얼굴 임베딩이 0개입니다. 사진을 인식 못한 것 같습니다.")
        print(f"   face_files 목록: {face_files}")
        return False
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다.")
        return False

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # ── 역방향 패스 먼저 실행 ──────────────────────────────────
    reverse_positions = run_reverse_pass(input_path, total_frames, frame_width, frame_height)
    # reverse_positions[frame_idx] = [(cx, cy, w, h), ...]

    # ── 정방향 1단계: 트래킹 + 신원 확인 ──────────────────────
    print("🔍 정방향 1단계 진행 중...")

    cap = cv2.VideoCapture(input_path)
    heatmap_engine = HeatmapEngine(frame_width, frame_height, grid_scale=0.1)

    SIMILARITY_THRESHOLD = 0.35
    global_identities = {}   # {track_id: True(등록자)/False(모르는사람)}
    identity_votes    = {}
    VOTE_REQUIREMENT  = 3
    frame_tracking_data = []  # [(boxes, ids), ...]
    prev_frame = None
    scene_change_threshold = 30.0
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if prev_frame is not None:
            small_curr = cv2.resize(frame, (64, 64))
            small_prev = cv2.resize(prev_frame, (64, 64))
            diff = cv2.absdiff(small_curr, small_prev)
            if np.mean(diff) > scene_change_threshold:
                heatmap_engine.reset_memory()

        prev_frame = frame.copy()

        results = face_detector.track(frame, persist=True, conf=0.30, imgsz=640,
                                       device=device, verbose=False)
        current_boxes, current_ids, current_confs = [], [], []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            current_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            current_ids   = results[0].boxes.id.int().cpu().tolist()
            current_confs = results[0].boxes.conf.cpu().tolist()

        all_boxes, all_ids = heatmap_engine.update_and_recover(
            current_boxes, current_ids, current_confs)

        for box, f_id in zip(all_boxes, all_ids):
            if global_identities.get(f_id) == True:
                continue
            try:
                x1, y1, x2, y2 = map(int, box)
                img_h, img_w = frame.shape[:2]
                pad_w = int((x2 - x1) * 0.3)
                pad_h = int((y2 - y1) * 0.3)
                rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                rx2, ry2 = min(img_w, x2 + pad_w), min(img_h, y2 + pad_h)
                face_crop = frame[ry1:ry2, rx1:rx2]

                if face_crop.size > 0:
                    faces_in_crop = face_aligner.get(face_crop)
                    if faces_in_crop:
                        target_face = sorted(faces_in_crop,
                                             key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                                             reverse=True)[0]
                        aligned_bgr = face_align.norm_crop(face_crop, landmark=target_face.kps, image_size=112)
                        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                        input_tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            target_embed = adaface_model(input_tensor)[0].cpu().numpy()

                        best_sim = -1.0
                        for known_emb in known_embeddings:
                            same, sim = is_same_person(target_embed, known_emb, SIMILARITY_THRESHOLD)
                            if sim > best_sim:
                                best_sim = sim

                        if best_sim > SIMILARITY_THRESHOLD:
                            identity_votes[f_id] = identity_votes.get(f_id, 0) + 1
                            if identity_votes[f_id] >= VOTE_REQUIREMENT:
                                global_identities[f_id] = True
                        else:
                            if f_id not in global_identities:
                                global_identities[f_id] = False
            except Exception:
                pass

        frame_tracking_data.append((all_boxes, all_ids))
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  정방향 {frame_idx}/{total_frames} 프레임 처리 중...")

    cap.release()

    # ── 2단계: 렌더링 (정방향 + 역방향 합집합) ────────────────
    print("🎬 2단계 렌더링 시작...")

    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    # 역방향 위치 정보를 픽셀 좌표 박스로 변환해두기
    # reverse_boxes[frame_idx] = [(x1,y1,x2,y2), ...]
    reverse_boxes = []
    for positions in reverse_positions:
        boxes = []
        for (cx, cy, w, h) in positions:
            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = cx + w // 2
            y2 = cy + h // 2
            boxes.append((x1, y1, x2, y2))
        reverse_boxes.append(boxes)

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_idx >= len(frame_tracking_data):
            break

        fwd_boxes, fwd_ids = frame_tracking_data[frame_idx]
        img_h, img_w = frame.shape[:2]

        # 정방향에서 모르는 사람 모자이크
        fwd_mosaic_boxes = []
        for box, f_id in zip(fwd_boxes, fwd_ids):
            if global_identities.get(f_id) != True:
                fwd_mosaic_boxes.append(box)

        # 역방향에서 추가로 잡힌 위치 중 정방향 박스와 겹치지 않는 것만 추가
        # (정방향에서 이미 등록자로 확인된 위치는 역방향에서도 스킵)
        rev_extra_boxes = []
        for rev_box in reverse_boxes[frame_idx]:
            rx1, ry1, rx2, ry2 = rev_box
            rcx = (rx1 + rx2) / 2
            rcy = (ry1 + ry2) / 2

            # 정방향 등록자 박스와 겹치는지 확인
            is_known = False
            for fbox, fid in zip(fwd_boxes, fwd_ids):
                if global_identities.get(fid) == True:
                    fx1, fy1, fx2, fy2 = map(int, fbox)
                    fcx = (fx1 + fx2) / 2
                    fcy = (fy1 + fy2) / 2
                    fw  = fx2 - fx1
                    fh  = fy2 - fy1
                    # 중심점 거리로 같은 얼굴인지 판단 (얼굴 크기 기준)
                    if abs(rcx - fcx) < fw * 0.8 and abs(rcy - fcy) < fh * 0.8:
                        is_known = True
                        break

            # 정방향 모자이크 박스와 이미 겹치는지 확인 (중복 모자이크 방지)
            already_covered = False
            if not is_known:
                for fbox in fwd_mosaic_boxes:
                    fx1, fy1, fx2, fy2 = map(int, fbox)
                    fcx = (fx1 + fx2) / 2
                    fcy = (fy1 + fy2) / 2
                    fw  = fx2 - fx1
                    fh  = fy2 - fy1
                    if abs(rcx - fcx) < fw * 0.8 and abs(rcy - fcy) < fh * 0.8:
                        already_covered = True
                        break

            if not is_known and not already_covered:
                rev_extra_boxes.append(rev_box)

        # 정방향 모자이크 적용
        for box in fwd_mosaic_boxes:
            apply_mosaic(frame, box)

        # 역방향 보강 모자이크 적용
        for box in rev_extra_boxes:
            apply_mosaic(frame, box)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  렌더링 {frame_idx}/{total_frames} 프레임 처리 중...")

    cap.release()
    out.release()

    print("✅ 영상 처리 완료! ffmpeg로 재인코딩 중...")

    temp_path = output_path.replace('.mp4', '_temp.mp4')
    os.rename(output_path, temp_path)

    result = subprocess.run([
        'ffmpeg', '-y',
        '-i', temp_path,
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '23',
        '-movflags', '+faststart',
        output_path
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')

    os.remove(temp_path)

    if result.returncode != 0:
        print(f"❌ ffmpeg 실패: {result.stderr}")
        return False

    print("✅ ffmpeg 재인코딩 완료!")

    if not LOCAL_TEST_MODE:
        if os.path.exists(input_path):
            os.remove(input_path)
        send_push_notification(device_token, output_path)

    return True


# ==========================================
# Celery 백그라운드 작업 (서버 모드)
# ==========================================
if not LOCAL_TEST_MODE:
    try:
        from celery_worker import celery_app

        @celery_app.task(name="process_video")
        def process_video_task(input_path, output_path, device_token, user_id):
            print(f"🎬 [Celery 워커] 영상 처리 시작: {input_path}")
            video_name    = os.path.splitext(os.path.basename(input_path))[0]
            original_name = video_name[37:] if len(video_name) > 37 else video_name
            output_path   = f"outputs/{original_name}_변환.mp4"
            user_faces_dir = f"user_faces/{user_id}"

            success = process_video(
                input_path=input_path,
                output_path=output_path,
                user_faces_dir=user_faces_dir,
                device_token=device_token,
                user_id=user_id,
            )

            if success and os.path.exists(input_path):
                os.remove(input_path)
            if success:
                user_faces_dir_path = f"user_faces/{user_id}"
                if os.path.exists(user_faces_dir_path):
                    shutil.rmtree(user_faces_dir_path)

            return success

    except ImportError:
        print("⚠️ celery_worker를 찾을 수 없습니다. 로컬 모드로만 사용하세요.")


# ==========================================
# 로컬 VSCode 직접 실행
# ==========================================
if __name__ == "__main__":
    if LOCAL_TEST_MODE:
        print("=" * 50)
        print("🖥️  로컬 테스트 모드")
        print(f"   동영상  : {LOCAL_VIDEO_PATH}")
        print(f"   얼굴폴더 : {LOCAL_FACES_DIR}")
        print(f"   출력    : {LOCAL_OUTPUT_PATH}")
        print("=" * 50)

        if not os.path.exists(LOCAL_VIDEO_PATH):
            print(f"❌ 동영상 파일을 찾을 수 없습니다: {LOCAL_VIDEO_PATH}")
        elif not os.path.exists(LOCAL_FACES_DIR):
            print(f"❌ 얼굴 폴더를 찾을 수 없습니다: {LOCAL_FACES_DIR}")
        else:
            success = process_video(
                input_path=LOCAL_VIDEO_PATH,
                output_path=LOCAL_OUTPUT_PATH,
                user_faces_dir=LOCAL_FACES_DIR,
            )
            if success:
                print(f"\n🎉 완료! 결과물: {LOCAL_OUTPUT_PATH}")
            else:
                print("\n❌ 처리 실패")
    else:
        print("서버 모드입니다. Celery 워커로 실행하세요.")