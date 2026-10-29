import os
import cv2
from ultralytics import YOLO
import numpy as np
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import math
import subprocess
import shutil
import tempfile

from heatmap_engine import HeatmapEngine

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
    checkpoint   = torch.load('adaface_ir50_ms1mv2.ckpt', map_location=device, weights_only=False)  # torch>=2.6 기본값 대응 (공식 AdaFace ckpt)
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
    YOLO 내부 tracker 상태를 초기화한다.
    역방향 패스 시작 시 이전 정방향 tracker 상태를 제거한다.
    predictor=None 으로 밀면 이후 탐지 박스가 크게 줄어들어 tracker.reset() 만 사용한다.
    """
    try:
        for tr in face_detector.predictor.trackers:
            tr.reset()
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
                title='NEMO De-identification Complete!',
                body='The de-identification process is finished! Please open the app and download it.',
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
    frame_paths,
    frame_width,
    frame_height,
    known_embeddings,
    pass_name="패스",
    similarity_threshold=0.35,
    vote_requirement=3,
    scene_change_threshold=30.0,
    embedding_interval=1,
):
    print(f"🔄 {pass_name} 시작... "
          f"(총 {len(frame_paths)} 프레임, 임베딩 interval={embedding_interval})")
    
    heatmap_engine    = HeatmapEngine(frame_width, frame_height, grid_scale=0.1)
    global_identities = {}   
    identity_votes    = {}   
    last_embed_frame  = {}   
    tracking_data     = []   
    prev_frame_small  = None # 비교용 축소 프레임만 메모리에 유지
    scene_changed     = False  

    for idx, path in enumerate(frame_paths):
        # ── 디스크에서 현재 프레임만 RAM으로 로드 ──
        frame = cv2.imread(path)
        if frame is None:
            continue

        # ── 씬 체인지 감지 ──────────────────────────────────
        scene_changed = False
        small_curr = cv2.resize(frame, (64, 64))
        
        if prev_frame_small is not None:
            diff = cv2.absdiff(small_curr, prev_frame_small)
            if np.mean(diff) > scene_change_threshold: 
                heatmap_engine.reset_memory() 
                scene_changed = True   
        prev_frame_small = small_curr.copy()
 
        # ── YOLO 트래킹 ─────────────────────────────────────
        current_boxes, current_ids, current_confs = [], [], []
        try:
            results = face_detector.track(
                frame, persist=True, conf=0.30, imgsz=640,
                device=device, verbose=False)
            if (results and len(results) > 0
                    and results[0].boxes is not None
                    and results[0].boxes.id is not None):
                current_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() 
                current_ids   = results[0].boxes.id.int().cpu().tolist() 
                current_confs = results[0].boxes.conf.cpu().tolist() 
        except (IndexError, Exception) as e:
            if idx % 100 == 0:
                print(f"  [{pass_name}] 프레임 {idx} track() 오류 스킵: {e}")

        # 히트맵 메모리로 탐지 누락 복원 (복원 박스에도 ID 부여)
        raw_boxes, raw_ids, rec_boxes, rec_ids = heatmap_engine.detect_with_recovery_ids(
            current_boxes, current_ids, current_confs)
        all_boxes = raw_boxes + rec_boxes
        all_ids   = raw_ids + rec_ids
            
        # ── 신원 판단 (실제 탐지된 박스에서만 임베딩 추출) ──────────
        for box, f_id in zip(raw_boxes, raw_ids):
            if global_identities.get(f_id) is True:
                continue
 
            last_idx = last_embed_frame.get(f_id, -999)
            should_extract = (
                scene_changed or (idx - last_idx) >= embedding_interval  
            )
            if not should_extract:
                continue
 
            last_embed_frame[f_id] = idx  
 
            try:
                embedding = extract_embedding(frame, box)
                is_reg, best_sim = check_identity(
                    embedding, known_embeddings, similarity_threshold)
 
                if is_reg:
                    identity_votes[f_id] = identity_votes.get(f_id, 0) + 1
                    if identity_votes[f_id] >= vote_requirement: 
                        global_identities[f_id] = True
                        print(f"  ✅ [{pass_name}] ID {f_id} → 등록자 확정 "
                              f"(프레임 {idx}, sim={best_sim:.3f})")
                else:
                    if f_id not in global_identities:
                        global_identities[f_id] = False
 
            except Exception:
                pass
 
        tracking_data.append((all_boxes, all_ids))
 
        if idx % 100 == 0:
            print(f"  [{pass_name}] {idx}/{len(frame_paths)} 프레임 완료")
 
    print(f"✅ {pass_name} 완료 — 등록자 ID: {[k for k,v in global_identities.items() if v]}")
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
        if img is None: continue
        faces = face_aligner.get(img)
        if not faces: continue
        target = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
        aligned_bgr = face_align.norm_crop(img, landmark=target.kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = adaface_transform(Image.fromarray(aligned_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = adaface_model(tensor)[0].cpu().numpy()
        known_embeddings.append(emb)
    if len(known_embeddings) == 0:
        print("❌ 등록 얼굴 임베딩 0개 — 처리 중단")
        return False
 
    # ── 2) 영상 정보 파악 & 3) 프레임을 디스크에 임시 저장 (메모리 절약) ───
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다.")
        return False
 
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 임시 디렉토리 생성 (자동 삭제를 위해 tempfile 사용)
    temp_dir = tempfile.mkdtemp(prefix="nemo_frames_")
    print(f"\n📂 프레임 디스크 캐싱 중... (RAM 절약 모드 -> {temp_dir})")
    
    frame_paths = []
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    cap.release()
 
    total_frames = len(frame_paths)
    print(f"✅ {total_frames} 프레임 디스크 캐싱 완료\n")
 
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
 
    # ── 4) 정방향 패스 ────────────────
    print("\n" + "─"*50)
    print("▶  정방향 패스")
    print("─"*50)
    reset_tracker()
 
    fwd_tracking, fwd_identities = run_identity_pass(
        frame_paths=frame_paths, # 리스트 대신 경로 전달
        frame_width=frame_width,
        frame_height=frame_height,
        known_embeddings=known_embeddings,
        pass_name="정방향",
        embedding_interval=1,   
    )
 
    # ── 5) 역방향 패스 ────────────────
    print("\n" + "─"*50)
    print("◀  역방향 패스")
    print("─"*50)
    reset_tracker()
 
    # 역방향 처리를 위해 경로 리스트만 뒤집음
    rev_paths = list(reversed(frame_paths))
 
    rev_tracking_reversed, rev_identities = run_identity_pass(
        frame_paths=rev_paths,
        frame_width=frame_width,
        frame_height=frame_height,
        known_embeddings=known_embeddings,
        pass_name="역방향",
        embedding_interval=5,   # 속도 개선 최우선: 역방향 인터벌을 3 -> 5로 대폭 상향
    )
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
 
    # ── 7) 렌더링 (디스크에서 하나씩 꺼내서 바로 영상으로 쓰기) ─────
    print("\n" + "─"*50)
    print("🎬 렌더링 및 모자이크 적용 시작...")
    print("─"*50)
 
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height))
 
    for frame_idx, path in enumerate(frame_paths):
        frame = cv2.imread(path) # 필요한 순간에만 RAM에 로드
        if frame is None: continue

        fwd_boxes, fwd_ids = fwd_tracking[frame_idx]
        rev_boxes, rev_ids = rev_tracking[frame_idx]
 
        fwd_mosaic_boxes   = []   
        fwd_registered_boxes = []  
 
        for box, f_id in zip(fwd_boxes, fwd_ids):
            if merged_identities.get(f_id) is True:
                fwd_registered_boxes.append(box)
            else:
                fwd_mosaic_boxes.append(box)
 
        rev_extra_mosaic = []
        for rev_box, rev_id in zip(rev_boxes, rev_ids):
            rx1, ry1, rx2, ry2 = map(int, rev_box)
            rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            rw, rh  = rx2 - rx1, ry2 - ry1
 
            overlap_with_registered = False
            for fbox in fwd_registered_boxes:
                fx1, fy1, fx2, fy2 = map(int, fbox)
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                fw, fh  = fx2 - fx1, fy2 - fy1
                if centers_close(rcx, rcy, fcx, fcy, max(rw, fw), max(rh, fh), ratio=0.8):
                    overlap_with_registered = True
                    break
 
            if overlap_with_registered or rev_identities.get(rev_id) is True:
                continue
 
            already_covered = False
            for fbox in fwd_mosaic_boxes:
                fx1, fy1, fx2, fy2 = map(int, fbox)
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                fw, fh  = fx2 - fx1, fy2 - fy1
                if centers_close(rcx, rcy, fcx, fcy, max(rw, fw), max(rh, fh), ratio=0.8):
                    already_covered = True
                    break
 
            if not already_covered:
                rev_extra_mosaic.append(rev_box)
 
        # 모자이크 적용
        for box in fwd_mosaic_boxes + rev_extra_mosaic:
            apply_mosaic(frame, box)
 
        out.write(frame)
 
        if frame_idx % 100 == 0:
            print(f"  렌더링 {frame_idx}/{total_frames} 프레임 완료")
 
    out.release()
    print("✅ 렌더링 완료")
    # ── 임시 디렉토리 청소  ──
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("임시 프레임 폴더 정리 완료")
 
    # ── 8) ffmpeg 재인코딩 ─────────────────────────────────
    print("\n🔧 ffmpeg 재인코딩 중...")
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    if shutil.which('ffmpeg') is None:
        print("❌ ffmpeg를 찾을 수 없습니다. 설치 후 PATH를 확인하세요. (렌더링 결과는 재인코딩 없이 유지됩니다)")
        return False

    os.replace(output_path, temp_path)

    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', temp_path, '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
            '-movflags', '+faststart', output_path
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        ok = result.returncode == 0
    except Exception:
        ok = False
        raise
    finally:
        if ok:
            os.remove(temp_path)
        else:
            # 실패 시 원본 렌더링 결과를 원래 이름으로 복구
            if os.path.exists(output_path):
                os.remove(output_path)
            os.replace(temp_path, output_path)

    if not ok:
        print(f"❌ ffmpeg 실패:\n{result.stderr}")
        return False
 
    print("✅ ffmpeg 재인코딩 완료!")

    # ── 9) 썸네일 추출 ─────────────────────────────────────
    thumbnail_path = output_path.replace('.mp4', '.jpg')
    thumb_result = subprocess.run([
        'ffmpeg', '-y', '-ss', '1', '-i', output_path, '-vframes', '1',
        '-q:v', '2', thumbnail_path
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')

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