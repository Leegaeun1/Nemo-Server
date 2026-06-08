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
LOCAL_TEST_MODE = True
LOCAL_VIDEO_PATH  = "videos/video_03.mp4"
LOCAL_FACES_DIR   = "data/video_03/registered_face"
LOCAL_OUTPUT_PATH = "outputs/test3.mp4"

# ==========================================
# Firebase (서버 모드 전용)
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
    """
    YOLO 트래커가 ID를 잃거나 튀는 현상을 보완하기 위한 히트맵 기반 안정화 엔진.
    얼굴이 잠깐 가려지거나 신뢰도가 낮아져도 이전 위치/크기를 스무딩하여 유지한다.
    """
 
    def __init__(self, frame_width, frame_height, grid_scale=0.1):
        self.grid_w = int(frame_width * grid_scale) # 히트맵 가로 크기(원본의 10%)
        self.grid_h = int(frame_height * grid_scale) # 히트맵 세로 크기
        self.scale  = grid_scale # 0.1 = 10% 축소 비율
        self.channels     = {}   # {track_id: heatmap(grid_h, grid_w)} : 얼굴별 위치 기록
        self.last_sizes   = {}   # {track_id: (w, h)} : 마지막으로 본 얼굴 크기
        self.last_centers = {}   # {track_id: (cx, cy)} : 마지막으로 본 얼굴 중심
        self.base_decay   = 0.96 # 매 프레임마다 히트맵을 4% 감쇠.(오래된 위치 희석.) => 30프레임 기준 1초뒤 30%됨
 
    def reset_memory(self): # 씬 전환 감지 시 모든 기억 초기화!
        self.channels.clear()
        self.last_sizes.clear()
        self.last_centers.clear()
 
    def get_visual_heatmap(self): # 시각화용!
        combined = np.zeros((self.grid_h, self.grid_w), dtype=np.float32) # 빈 히트맵 생성
        for heatmap in self.channels.values():
            combined = np.maximum(combined, heatmap) # 모든 ID 히트맵을 최댓값으로 합성 -> 최대인부분만 볼수있도록
        norm = np.clip(combined / 10.0 * 255, 0, 255).astype(np.uint8) # 0~255로 정규화
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET) # JET 컬러맵으로 시각화
        gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY) 
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY) # 낮은 값 영역 마스킹 
        return cv2.bitwise_and(colored, colored, mask=mask) # 마스킹 적용해서 반환
 
    def update_and_recover(self, boxes, ids, confs):
        # 1) 모든 채널 decay
        for f_id in list(self.channels.keys()):
            self.channels[f_id] *= self.base_decay # 매 프레임 0.96 곱하기 -> 시간 지날수록 희미..
 
        final_boxes = []
        final_ids   = []
        seen_ids    = set()
 
        for box, f_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1 # 박스 너비, 높이
            cx, cy = x1 + w // 2, y1 + h // 2 # 중심점
            hcx = int(cx * self.scale) # 히트맵 그리드 좌표로 변환
            hcy = int(cy * self.scale)
 
            # 2) 신규 ID → 히트맵에서 같은 위치 old_id 매핑 시도
            matched_id = f_id
            if f_id not in self.channels: # 처음 보는 ID
                best_id   = None # 아직 후보 없음
                best_heat = 0.2 # 최솟값 임계치(히트맵이 최소 0.2는 넘어야 인정.)
                for old_id, heatmap in self.channels.items():
                    if 0 <= hcy < self.grid_h and 0 <= hcx < self.grid_w:
                        if heatmap[hcy, hcx] > best_heat: # 현재 위치와 히트맵 값 비교!
                            best_heat = heatmap[hcy, hcx] # 가장 높은값으로 갱신(가장 뜨겁)
                            best_id   = old_id
                if best_id is not None: # 찾았음
                    matched_id = best_id # 같은 자리의 old_id로 교체 -> ID 연속성 유지.
 
            if matched_id in seen_ids: # 이미 이 프레임에서 처리한 ID이면 스킵!
                continue
            seen_ids.add(matched_id) # 첨보는 ID이면 처리!
 
            # 3) 부분 가려짐 판단
            is_full_face = True
            if conf < 0.45: # 신뢰도 낮으면 가려진것으로 판단!
                is_full_face = False
            elif matched_id in self.last_sizes:
                prev_w, prev_h = self.last_sizes[matched_id]
                if w < prev_w * 0.8 or h < prev_h * 0.8: # 이전보다 20%이상 작아지면 가려짐
                    is_full_face = False
 
            # 4) 위치/크기 스무딩
            if matched_id in self.last_sizes:
                if is_full_face: # 안가려졌음
                    prev_w, prev_h = self.last_sizes[matched_id]
                    pcx, pcy = self.last_centers[matched_id]
                    # 이전 70% 현재30% -> 지수이동평균으로 스무딩
                    self.last_sizes[matched_id]   = (int(prev_w * 0.7 + w * 0.3),
                                                      int(prev_h * 0.7 + h * 0.3))
                    self.last_centers[matched_id] = (int(pcx * 0.7 + cx * 0.3),
                                                      int(pcy * 0.7 + cy * 0.3))
            else: # 처음 보는 ID이면 그냥 저장
                self.last_sizes[matched_id]   = (w, h)
                self.last_centers[matched_id] = (cx, cy)
 
            curr_cx, curr_cy = self.last_centers[matched_id]
            curr_w,  curr_h  = self.last_sizes[matched_id]
 
            # 5) 히트맵 채널 업데이트
            if matched_id not in self.channels:
                self.channels[matched_id] = np.zeros((self.grid_h, self.grid_w), dtype=np.float32) # 새 히트맵 생성
            # 스무딩된 박스 영역에 +2.0 누적, 최대 10.0 
            gx1 = max(0, int((curr_cx - curr_w / 2) * self.scale))
            gy1 = max(0, int((curr_cy - curr_h / 2) * self.scale))
            gx2 = min(self.grid_w, int((curr_cx + curr_w / 2) * self.scale))
            gy2 = min(self.grid_h, int((curr_cy + curr_h / 2) * self.scale))
            self.channels[matched_id][gy1:gy2, gx1:gx2] += 2.0
            self.channels[matched_id] = np.clip(self.channels[matched_id], 0, 10.0)
 
            final_boxes.append([
                curr_cx - curr_w // 2, curr_cy - curr_h // 2,
                curr_cx + curr_w // 2, curr_cy + curr_h // 2
            ])
            final_ids.append(matched_id)
 
        # 6) 오래된 채널 삭제
        for f_id in list(self.channels.keys()):
            if f_id not in seen_ids and np.max(self.channels[f_id]) <= 0.5:
                # 이 프레임에 안보이고 히트맵도 거의 희미해졌으면 삭제!
                del self.channels[f_id]
                self.last_centers.pop(f_id, None)
                self.last_sizes.pop(f_id, None)
 
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
    face_detector = YOLO('models/yolov12s-face.pt').to(device)
except Exception:
    face_detector = YOLO('models/yolov12n-face.pt').to(device)
 
mp_face_mesh  = mp.solutions.face_mesh
face_mesh     = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=False,
                                       min_detection_confidence=0.3)
static_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                          refine_landmarks=False,
                                          min_detection_confidence=0.5)
 
try:
    face_aligner = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'],
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_aligner.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
except Exception as e:
    print(f"face_aligner 로드 에러: {e}")
 
adaface_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
 
adaface_model = build_model('ir_50').to(device)
try:
    checkpoint   = torch.load('adaface_ir50_ms1mv2.ckpt', map_location=device)
    state_dict   = checkpoint.get('state_dict', checkpoint)
    state_dict   = {k.replace('model.', ''): v for k, v in state_dict.items()}
    adaface_model.load_state_dict(state_dict, strict=False)
    adaface_model.eval()
except Exception:
    print("⚠️ adaface checkpoint를 찾지 못했습니다.")
 
 
# ==========================================
# 유틸 함수
# ==========================================
def cosine_similarity(embed1, embed2):
    """두 임베딩 벡터의 코사인 유사도 반환."""
    e1 = embed1.flatten() / (np.linalg.norm(embed1) + 1e-8)
    e2 = embed2.flatten() / (np.linalg.norm(embed2) + 1e-8)
    return float(np.dot(e1, e2)) # 내적
 
 
def is_same_person(embed1, embed2, threshold=0.4):
    sim = cosine_similarity(embed1, embed2)
    return sim > threshold, sim # threshold넘기면 같은사람임
 
 
def extract_embedding(frame, box):
    """
    박스 영역을 크롭 → InsightFace 정렬 → AdaFace 임베딩 추출.
    실패 시 None 반환.
    """
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    # 박스 주변에 30% 패딩 추가!(얼굴 주변 여백 확보)
    pad_w = int((x2 - x1) * 0.3)
    pad_h = int((y2 - y1) * 0.3)
    rx1 = max(0, x1 - pad_w)
    ry1 = max(0, y1 - pad_h)
    rx2 = min(img_w, x2 + pad_w)
    ry2 = min(img_h, y2 + pad_h)
    face_crop = frame[ry1:ry2, rx1:rx2] # 패딩 포함해서 크롭
 
    if face_crop.size == 0:
        return None
 
    faces = face_aligner.get(face_crop) # InsightFace로 얼굴 재검출
    if not faces:
        return None
    # 가장 큰 얼굴 선택
    target = sorted(faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True)[0]
    # 눈/코 랜드마크 기준으로 112x112 정렬 크롭!
    aligned_bgr = face_align.norm_crop(face_crop, landmark=target.kps, image_size=112)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
 
    with torch.no_grad():
        embedding = adaface_model(tensor)[0].cpu().numpy() # 임베딩 벡터 반환
    return embedding
 
 
def check_identity(embedding, known_embeddings, threshold=0.35):
    """
    known_embeddings 중 가장 높은 유사도를 구해 등록자 여부 반환.
    반환값: (is_registered: bool, best_sim: float)
    """
    if embedding is None or len(known_embeddings) == 0:
        return False, -1.0
    best_sim = max(cosine_similarity(embedding, k) for k in known_embeddings)
    return best_sim > threshold, best_sim
 
 
def centers_close(cx1, cy1, cx2, cy2, ref_w, ref_h, ratio=0.8):
    """두 중심점이 얼굴 크기 기준 ratio 이내인지 확인."""
    return abs(cx1 - cx2) < ref_w * ratio and abs(cy1 - cy2) < ref_h * ratio
 
 
def apply_mosaic(frame, box):
    """원형 가우시안 블러 모자이크."""
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(img_w, x2), min(img_h, y2)
    if rx2 <= rx1 or ry2 <= ry1:
        return
 
    cx = (rx1 + rx2) // 2
    cy = (ry1 + ry2) // 2
    bw = rx2 - rx1
    bh = ry2 - ry1
    radius = int(max(bw, bh) * 0.6)
 
    cx_min = max(0, cx - radius)
    cy_min = max(0, cy - radius)
    cx_max = min(img_w, cx + radius)
    cy_max = min(img_h, cy + radius)
    circle_roi = frame[cy_min:cy_max, cx_min:cx_max]
 
    if circle_roi.size == 0:
        return
 
    mask = np.zeros(circle_roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx - cx_min, cy - cy_min), radius, 255, -1)
    blurred = cv2.GaussianBlur(circle_roi, (121, 121), 50)
    frame[cy_min:cy_max, cx_min:cx_max] = np.where(
        mask[:, :, None] == 255, blurred, circle_roi)
 
 
def reset_tracker():
    """
    YOLO 내부 tracker 상태를 완전히 초기화한다.
    predictor 자체를 None으로 밀어서 다음 track() 호출 시 새로 생성되게 한다.
    이렇게 해야 역방향 패스 시작 시 이전 정방향 tracker 상태가 완전히 제거된다.
    """
    try:
        face_detector.predictor = None
    except Exception:
        pass
 
 
def send_push_notification(token, output_path):
    if LOCAL_TEST_MODE:
        return
    if not token or not firebase_admin._apps:
        return
    try:
        file_size_mb  = os.path.getsize(output_path) / (1024 * 1024)
        message = messaging.Message(
            notification=messaging.Notification(
                title='NEMO 비식별화 완료!',
                body='비식별화 처리가 끝났습니다! 앱으로 들어와서 다운로드해주세요.',
            ),
            data={
                "output_filename": os.path.basename(output_path),
                "file_size": f"{file_size_mb:.1f}MB",
            },
            token=token,
        )
        response = messaging.send(message)
        print(f"[알림 성공] {response}")
    except Exception as e:
        print(f"[알림 실패] {e}")
 
 
# ==========================================
# 신원 판단 공용 로직 (정방향 / 역방향 공통)
# ==========================================
def run_identity_pass(
    frames_list,
    frame_width,
    frame_height,
    known_embeddings,
    pass_name="패스",
    similarity_threshold=0.35,
    vote_requirement=3,
    scene_change_threshold=30.0,
    embedding_interval=1,
):
    """
    frames_list 순서대로 프레임을 순회하며 YOLO 트래킹 + 신원 판단을 수행한다.
 
    파라미터:
        embedding_interval : 몇 프레임마다 임베딩을 추출할지 결정.
                             1 = 매 프레임 (정방향 기본값, 가장 정확)
                             3 = 3프레임마다 (역방향 기본값, 속도·정확도 균형)
                             씬 체인지가 감지된 직후 프레임은 interval과 무관하게
                             항상 추출하여 새 씬의 첫 얼굴을 놓치지 않는다.
 
    반환값:
        tracking_data     : List[ (all_boxes, all_ids) ]  — frames_list 순서 기준
        global_identities : dict { track_id -> True(등록자) / False(비등록자) }
 
    역방향 패스일 때는 frames_list 를 이미 뒤집어서 넣으면 된다.
    tracking_data 도 frames_list 순서 그대로 반환되므로,
    역방향으로 넣었다면 호출 측에서 다시 reverse() 해서 정방향 인덱스에 맞춰야 한다.
    """
    print(f"🔄 {pass_name} 시작... "
          f"(총 {len(frames_list)} 프레임, 임베딩 interval={embedding_interval})")
    # 패스마다 새 히트맵 엔진 생성
    heatmap_engine    = HeatmapEngine(frame_width, frame_height, grid_scale=0.1)
    global_identities = {}   # {track_id: True / False}
    identity_votes    = {}   # {track_id: int} : 누적 투표수
    # 각 track_id 별로 마지막으로 임베딩을 시도한 프레임 인덱스 기록
    last_embed_frame  = {}   # {track_id: int}
    tracking_data     = []   # [(all_boxes, all_ids), ...] # 프레임별 트래킹 결과 저장
    prev_frame        = None
    scene_changed     = False  # 씬 체인지 직후 플래그
 
    for idx, frame in enumerate(frames_list):
        # ── 씬 체인지 감지 ──────────────────────────────────
        scene_changed = False
        if prev_frame is not None:
            small_curr = cv2.resize(frame, (64, 64)) # 64x64로 축소해서 빠르게 비교
            small_prev = cv2.resize(prev_frame, (64, 64))
            diff = cv2.absdiff(small_curr, small_prev)
            if np.mean(diff) > scene_change_threshold: # 평균 픽셀 차이 > 30이면 씬 전환
                heatmap_engine.reset_memory() # 초기화
                scene_changed = True   # 씬 전환 직후 -> 무조건 임베딩 추출
        prev_frame = frame.copy()
 
        # ── YOLO 트래킹 ─────────────────────────────────────
        current_boxes, current_ids, current_confs = [], [], []
        try:
            # YOLO 트래킹 , persist=True(프레임 간 ID 유지)
            results = face_detector.track(
                frame, persist=True, conf=0.30, imgsz=640,
                device=device, verbose=False)
            if (results and len(results) > 0
                    and results[0].boxes is not None
                    and results[0].boxes.id is not None):
                current_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() # [x1,y1,x2,y2] 목록
                current_ids   = results[0].boxes.id.int().cpu().tolist() # 트래킹 ID 목록
                current_confs = results[0].boxes.conf.cpu().tolist() # 신뢰도 목록
        except (IndexError, Exception) as e:
            # track() 내부 오류(tracker 상태 불일치 등) -> 해당 프레임 스킵
            if idx % 100 == 0:
                print(f"  [{pass_name}] 프레임 {idx} track() 오류 스킵: {e}")

        # 히트맵으로 보강
        all_boxes, all_ids = heatmap_engine.update_and_recover(
            current_boxes, current_ids, current_confs)
        # ── 신원 판단 ────────────────────────────────────────
        for box, f_id in zip(all_boxes, all_ids):
            # 이미 등록자로 확정된 ID는 스킵
            if global_identities.get(f_id) is True:
                continue
 
            # ── interval 체크 ──────────────────────────────
            # 씬 체인지 직후거나, 이 ID에 대해 interval 이상 지났을 때만 추출
            last_idx = last_embed_frame.get(f_id, -999)
            should_extract = (
                scene_changed                          # 씬 전환 직후는 무조건
                or (idx - last_idx) >= embedding_interval  # interval 도달
            )
            if not should_extract:
                continue
 
            last_embed_frame[f_id] = idx  # 추출 시도 시점 갱신
 
            try:
                embedding = extract_embedding(frame, box)
                is_reg, best_sim = check_identity(
                    embedding, known_embeddings, similarity_threshold)
 
                if is_reg:
                    identity_votes[f_id] = identity_votes.get(f_id, 0) + 1
                    if identity_votes[f_id] >= vote_requirement: # 3번 이상 등록자로 판단되면 확정!
                        global_identities[f_id] = True
                        print(f"  ✅ [{pass_name}] ID {f_id} → 등록자 확정 "
                              f"(프레임 {idx}, sim={best_sim:.3f})")
                else:
                    # 아직 비등록자로만 표시 (나중에 뒤집힐 수 있으므로 False로만 기록)
                    if f_id not in global_identities:
                        global_identities[f_id] = False
 
            except Exception:
                pass
 
        tracking_data.append((all_boxes, all_ids))
 
        if idx % 100 == 0:
            print(f"  [{pass_name}] {idx}/{len(frames_list)} 프레임 완료")
 
    print(f"✅ {pass_name} 완료 — "
          f"등록자 ID: {[k for k,v in global_identities.items() if v]}")
    return tracking_data, global_identities
 
 
# ==========================================
# 두 패스의 신원 정보 병합
# ==========================================
def merge_identities(
    fwd_identities,
    rev_identities,
    fwd_tracking,
    rev_tracking,
    total_frames,
):
    """
    정방향과 역방향 global_identities를 프레임별 위치 기반으로 병합한다.
 
    전략:
    - 역방향에서 등록자로 확정된 ID가,
      어느 프레임에서 정방향 ID와 위치가 겹치면 → 정방향 ID도 등록자로 업데이트.
    - 즉 역방향 정보가 정방향 신원 판단을 보강한다.
    """
    print("🔀 정방향 ↔ 역방향 신원 정보 병합 중...")
 
    merged = dict(fwd_identities)  # 정방향 복사본으로 시작
 
    # 역방향에서 등록자로 확정된 ID 목록
    rev_registered_ids = {rid for rid, val in rev_identities.items() if val is True}
    if not rev_registered_ids:
        print("  역방향 등록자 ID 없음 — 정방향 결과만 사용")
        return merged
 
    for frame_idx in range(total_frames):
        if frame_idx >= len(fwd_tracking) or frame_idx >= len(rev_tracking):
            break
 
        fwd_boxes, fwd_ids = fwd_tracking[frame_idx]
        rev_boxes, rev_ids = rev_tracking[frame_idx]
 
        for rev_box, rev_id in zip(rev_boxes, rev_ids):
            if rev_id not in rev_registered_ids: # 역방향에 X
                continue  # 역방향에서 등록자가 아닌 ID는 무시
 
            rx1, ry1, rx2, ry2 = map(int, rev_box)
            rcx = (rx1 + rx2) / 2
            rcy = (ry1 + ry2) / 2
            rw  = rx2 - rx1
            rh  = ry2 - ry1
 
            # 같은 프레임의 정방향 박스와 위치 비교
            for fwd_box, fwd_id in zip(fwd_boxes, fwd_ids):
                fx1, fy1, fx2, fy2 = map(int, fwd_box)
                fcx = (fx1 + fx2) / 2
                fcy = (fy1 + fy2) / 2
                fw  = fx2 - fx1
                fh  = fy2 - fy1
                ref_w = max(rw, fw)
                ref_h = max(rh, fh)
 
                if centers_close(rcx, rcy, fcx, fcy, ref_w, ref_h, ratio=0.8): # 같은 프레임에서 위치 겹침
                    if merged.get(fwd_id) is not True:
                        merged[fwd_id] = True
                        print(f"  ✅ [병합] 정방향 ID {fwd_id} → 등록자로 업데이트 "
                              f"(역방향 ID {rev_id}, 프레임 {frame_idx})")
                    break  # 이미 매핑됐으면 다음 역방향 박스로
 
    print(f"✅ 병합 완료 — 최종 등록자 ID: {[k for k,v in merged.items() if v]}")
    return merged
 
 
# ==========================================
# 메인 처리 함수
# ==========================================
def process_video(input_path, output_path, user_faces_dir,
                  device_token=None, user_id=None):
    print(f"\n{'='*60}")
    print(f"🎬 영상 처리 시작: {input_path}")
    print(f"{'='*60}\n")
 
    # ── 1) 등록 얼굴 임베딩 수집 ───────────────────────────
    known_embeddings = []
    face_files = glob.glob(f"{user_faces_dir}/person*.*")
    print(f"👤 등록된 얼굴 사진 {len(face_files)}개 로드")
 
    for img_path in face_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = face_aligner.get(img)
        if not faces:
            print(f"  ⚠️ 얼굴 미검출: {img_path}")
            continue
        target = sorted(faces,
                        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                        reverse=True)[0]
        aligned_bgr = face_align.norm_crop(img, landmark=target.kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = adaface_model(tensor)[0].cpu().numpy()
        known_embeddings.append(emb)
        print(f"  ✅ 임베딩 추출 완료: {os.path.basename(img_path)}")
 
    print(f"\n✅ known_embeddings: {len(known_embeddings)}개")
    if len(known_embeddings) == 0:
        print("❌ 등록 얼굴 임베딩 0개 — 처리 중단")
        return False
 
    # ── 2) 영상 정보 파악 ───────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다.")
        return False
 
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # ── 3) 전체 프레임 메모리 로드 ─────────────────────────
    print(f"\n📂 전체 프레임 로드 중... ({total_frames} 프레임)")
    all_frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        all_frames.append(frame)
    cap.release()
 
    total_frames = len(all_frames)
    print(f"✅ {total_frames} 프레임 로드 완료\n")
 
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True)
 
    # ── 4) 정방향 패스: 트래킹 + 신원 판단 ────────────────
    print("\n" + "─"*50)
    print("▶  정방향 패스")
    print("─"*50)
    # YOLO tracker 상태 초기화 (정방향 시작 전)
    reset_tracker()
 
    fwd_tracking, fwd_identities = run_identity_pass(
        frames_list=all_frames,
        frame_width=frame_width,
        frame_height=frame_height,
        known_embeddings=known_embeddings,
        pass_name="정방향",
        embedding_interval=1,   # 매 프레임 추출 (정방향은 최대 정확도)
    )
 
    # ── 5) 역방향 패스: 트래킹 + 신원 판단 ────────────────
    print("\n" + "─"*50)
    print("◀  역방향 패스")
    print("─"*50)
    # YOLO tracker 상태 초기화 (역방향 시작 전)
    reset_tracker()
 
    frames_reversed = list(reversed(all_frames))
 
    rev_tracking_reversed, rev_identities = run_identity_pass(
        frames_list=frames_reversed,
        frame_width=frame_width,
        frame_height=frame_height,
        known_embeddings=known_embeddings,
        pass_name="역방향",
        embedding_interval=3,   # 3프레임마다 임베딩 추출 (속도·정확도 균형)
    )
 
    # 역방향 tracking_data를 정방향 인덱스로 되돌리기
    rev_tracking = list(reversed(rev_tracking_reversed))
 
    # ── 6) 신원 정보 병합 ───────────────────────────────────
    print("\n" + "─"*50)
    merged_identities = merge_identities(
        fwd_identities=fwd_identities,
        rev_identities=rev_identities,
        fwd_tracking=fwd_tracking,
        rev_tracking=rev_tracking,
        total_frames=total_frames,
    )
 
    # ── 7) 렌더링 ───────────────────────────────────────────
    print("\n" + "─"*50)
    print("🎬 렌더링 시작...")
    print("─"*50)
 
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height))
 
    for frame_idx, frame in enumerate(all_frames):
        frame = frame.copy()
        img_h, img_w = frame.shape[:2]
 
        fwd_boxes, fwd_ids = fwd_tracking[frame_idx]
        rev_boxes, rev_ids = rev_tracking[frame_idx]
 
        # ── 정방향: 비등록자 모자이크 박스 수집 ─────────────
        fwd_mosaic_boxes   = []   # 정방향에서 확정된 모자이크 박스
        fwd_registered_boxes = []  # 정방향 등록자 박스 (역방향 비교용)
 
        for box, f_id in zip(fwd_boxes, fwd_ids):
            if merged_identities.get(f_id) is True:
                fwd_registered_boxes.append(box)
            else:
                fwd_mosaic_boxes.append(box)
 
        # ── 역방향 보강: 정방향에서 놓쳤거나 처음부터 가려진 얼굴 ──
        rev_extra_mosaic = []
 
        for rev_box, rev_id in zip(rev_boxes, rev_ids):
            rx1, ry1, rx2, ry2 = map(int, rev_box)
            rcx = (rx1 + rx2) / 2
            rcy = (ry1 + ry2) / 2
            rw  = rx2 - rx1
            rh  = ry2 - ry1
 
            # (a) 정방향 등록자 박스와 겹치면 → 등록자이므로 모자이크 안 함
            overlap_with_registered = False
            for fbox in fwd_registered_boxes:
                fx1, fy1, fx2, fy2 = map(int, fbox)
                fcx = (fx1 + fx2) / 2
                fcy = (fy1 + fy2) / 2
                fw  = fx2 - fx1
                fh  = fy2 - fy1
                if centers_close(rcx, rcy, fcx, fcy,
                                  max(rw, fw), max(rh, fh), ratio=0.8):
                    overlap_with_registered = True
                    break
 
            if overlap_with_registered:
                continue
 
            # (b) 역방향에서 등록자로 확정된 ID인지 확인
            if rev_identities.get(rev_id) is True:
                # 역방향 등록자 → 모자이크 안 함
                continue
 
            # (c) 정방향 모자이크 박스와 이미 겹치면 → 중복 방지
            already_covered = False
            for fbox in fwd_mosaic_boxes:
                fx1, fy1, fx2, fy2 = map(int, fbox)
                fcx = (fx1 + fx2) / 2
                fcy = (fy1 + fy2) / 2
                fw  = fx2 - fx1
                fh  = fy2 - fy1
                if centers_close(rcx, rcy, fcx, fcy,
                                  max(rw, fw), max(rh, fh), ratio=0.8):
                    already_covered = True
                    break
 
            if not already_covered:
                rev_extra_mosaic.append(rev_box)
 
        # ── 모자이크 적용 ────────────────────────────────────
        for box in fwd_mosaic_boxes:
            apply_mosaic(frame, box)
 
        for box in rev_extra_mosaic:
            apply_mosaic(frame, box)
 
        out.write(frame)
 
        if frame_idx % 100 == 0:
            print(f"  렌더링 {frame_idx}/{total_frames} 프레임 완료")
 
    out.release()
    print("✅ 렌더링 완료")
 
    # ── 8) ffmpeg 재인코딩 ─────────────────────────────────
    print("\n🔧 ffmpeg 재인코딩 중...")
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
        print(f"❌ ffmpeg 실패:\n{result.stderr}")
        return False
 
    print("✅ ffmpeg 재인코딩 완료!")

    # ── 9) 썸네일 추출 ─────────────────────────────────────
    thumbnail_path = output_path.replace('.mp4', '.jpg')
    thumb_result = subprocess.run([
        'ffmpeg', '-y',
        '-ss', '1',
        '-i', output_path,
        '-vframes', '1',
        '-q:v', '2',
        thumbnail_path
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')

    if thumb_result.returncode == 0:
        print(f"✅ 썸네일 생성 완료: {thumbnail_path}")
    else:
        print(f"⚠️ 썸네일 생성 실패 (무시하고 계속): {thumb_result.stderr}")

    # ── 10) 마무리 ──────────────────────────────────────────
    if not LOCAL_TEST_MODE:
        if os.path.exists(input_path):
            os.remove(input_path)
        send_push_notification(device_token, output_path)
 
    print(f"\n🎉 최종 결과물: {output_path}")
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
            video_name     = os.path.splitext(os.path.basename(input_path))[0]
            original_name  = video_name[37:] if len(video_name) > 37 else video_name
            output_path    = f"outputs/{original_name}_변환.mp4"
            user_faces_dir = f"user_faces/{user_id}"
 
            success = process_video(
                input_path=input_path,
                output_path=output_path,
                user_faces_dir=user_faces_dir,
                device_token=device_token,
                user_id=user_id,
            )
 
            if success:
                if os.path.exists(input_path):
                    os.remove(input_path)
                user_faces_dir_path = f"user_faces/{user_id}"
 
            return success
 
    except ImportError:
        print("⚠️ celery_worker를 찾을 수 없습니다. 로컬 모드로만 사용하세요.")
 
 
# ==========================================
# 로컬 직접 실행
# ==========================================
if __name__ == "__main__":
    if LOCAL_TEST_MODE:
        print("=" * 60)
        print("🖥️  로컬 테스트 모드")
        print(f"   동영상  : {LOCAL_VIDEO_PATH}")
        print(f"   얼굴폴더 : {LOCAL_FACES_DIR}")
        print(f"   출력    : {LOCAL_OUTPUT_PATH}")
        print("=" * 60)
 
        if not os.path.exists(LOCAL_VIDEO_PATH):
            print(f"❌ 동영상 파일 없음: {LOCAL_VIDEO_PATH}")
        elif not os.path.exists(LOCAL_FACES_DIR):
            print(f"❌ 얼굴 폴더 없음: {LOCAL_FACES_DIR}")
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