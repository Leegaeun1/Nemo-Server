"""
measure_identity.py
===================
동일인 판별 정확도 측정 스크립트.

측정 지표
---------
FRR (False Reject Rate) : 등록자가 모자이크된 프레임 비율  -> 낮을수록 좋음
FAR (False Accept Rate) : 미등록자가 통과된 프레임 비율    -> 낮을수록 좋음

등록자 프레임 : 등록자 얼굴 bbox가 라벨링된 프레임
  -> 모자이크 안 됐으면 정상 (TN), 됐으면 오류 (FRR)

미등록자 프레임 : 미등록자 얼굴 bbox가 라벨링된 프레임
  -> 모자이크 됐으면 정상 (TP), 안 됐으면 오류 (FAR)

폴더 구조
---------
videos/
    video_01.mp4 ...
data/
    video_01/
        labels_labelme/           기존 미등록자 GT (compare_all_V2용)
        registered_sample/        등록자 GT (sample_frames.py 출력 폴더)
            frame_XXXX.json       LabelMe 형식
        registered_face/
            face.jpg              등록자 사진 1장
            (또는 face.png, face.jpeg)

사용법
------
1. data/video_XX/registered_face/ 에 등록자 사진 넣기
2. data/video_XX/registered_sample/ 에 등록자 얼굴 bbox 라벨링
   (LabelMe 사용, label명은 아무거나 OK - 박스 있으면 등록자 프레임으로 판단)
3. python measure_identity.py
4. results/identity_eval.csv 와 콘솔 출력으로 결과 확인
"""

import os
import cv2
import json
import glob
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# 설정 
VIDEOS = [f"video_{i:02d}" for i in range(1, 6)]

FACE_MODEL_PATH  = "models/yolov12s-face.pt"
ADAFACE_PATH     = "adaface_ir50_ms1mv2.ckpt"
INSIGHTFACE_NAME = "buffalo_sc"

SIMILARITY_THRESHOLD = 0.35
VOTE_REQUIREMENT     = 3
EMBEDDING_INTERVAL   = 1    # 정방향: 매 프레임
CONF_THRESHOLD       = 0.30

OUTPUT_CSV = "results/identity_eval.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}")

# 모델 로드 
face_detector = YOLO(FACE_MODEL_PATH).to(device)

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    face_aligner = FaceAnalysis(name=INSIGHTFACE_NAME,
                                allowed_modules=['detection'],
                                providers=['CUDAExecutionProvider',
                                           'CPUExecutionProvider'])
    face_aligner.prepare(ctx_id=0 if torch.cuda.is_available() else -1,
                         det_size=(640, 640))
except Exception as e:
    print(f"InsightFace 로드 실패: {e}")
    exit(1)

try:
    import core.net as net
    adaface_model = net.build_model('ir_50')
    ckpt = torch.load(ADAFACE_PATH, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    adaface_model.load_state_dict(state_dict, strict=False)
    adaface_model.eval().to(device)
except Exception as e:
    print(f"AdaFace 로드 실패: {e}")
    exit(1)

adaface_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# HeatmapEngine
class HeatmapEngine:
    """
    YOLO 트래커가 ID를 잃거나 튀는 현상을 보완하기 위한 히트맵 기반 안정화 엔진.
    얼굴이 잠깐 가려지거나 신뢰도가 낮아져도 이전 위치/크기를 스무딩하여 유지한다.
    """

    def __init__(self, frame_width, frame_height, grid_scale=0.1):
        self.grid_w = int(frame_width * grid_scale)
        self.grid_h = int(frame_height * grid_scale)
        self.scale  = grid_scale
        self.channels     = {}   # {track_id: heatmap(grid_h, grid_w)}
        self.last_sizes   = {}   # {track_id: (w, h)}
        self.last_centers = {}   # {track_id: (cx, cy)}
        self.base_decay   = 0.96

    def reset_memory(self):
        self.channels.clear()
        self.last_sizes.clear()
        self.last_centers.clear()

    def update_and_recover(self, boxes, ids, confs):
        # 1) 모든 채널 decay
        for f_id in list(self.channels.keys()):
            self.channels[f_id] *= self.base_decay

        final_boxes = []
        final_ids   = []
        seen_ids    = set()

        for box, f_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            hcx = int(cx * self.scale)
            hcy = int(cy * self.scale)

            # 2) 신규 ID → 히트맵에서 같은 위치 old_id 매핑 시도
            matched_id = f_id
            if f_id not in self.channels:
                best_id   = None
                best_heat = 0.2
                for old_id, heatmap in self.channels.items():
                    if 0 <= hcy < self.grid_h and 0 <= hcx < self.grid_w:
                        if heatmap[hcy, hcx] > best_heat:
                            best_heat = heatmap[hcy, hcx]
                            best_id   = old_id
                if best_id is not None:
                    matched_id = best_id

            if matched_id in seen_ids:
                continue
            seen_ids.add(matched_id)

            # 3) 부분 가려짐 판단 (conf 낮거나 크기가 갑자기 줄어든 경우)
            is_full_face = True
            if conf < 0.45:
                is_full_face = False
            elif matched_id in self.last_sizes:
                prev_w, prev_h = self.last_sizes[matched_id]
                if w < prev_w * 0.8 or h < prev_h * 0.8:
                    is_full_face = False

            # 4) 위치/크기 스무딩
            if matched_id in self.last_sizes:
                if is_full_face:
                    prev_w, prev_h = self.last_sizes[matched_id]
                    pcx, pcy = self.last_centers[matched_id]
                    self.last_sizes[matched_id]   = (int(prev_w * 0.7 + w * 0.3),
                                                      int(prev_h * 0.7 + h * 0.3))
                    self.last_centers[matched_id] = (int(pcx * 0.7 + cx * 0.3),
                                                      int(pcy * 0.7 + cy * 0.3))
                # is_full_face=False이면 last_sizes/centers를 그대로 유지 (이전 값 사용)
            else:
                self.last_sizes[matched_id]   = (w, h)
                self.last_centers[matched_id] = (cx, cy)

            curr_cx, curr_cy = self.last_centers[matched_id]
            curr_w,  curr_h  = self.last_sizes[matched_id]

            # 5) 히트맵 채널 업데이트
            if matched_id not in self.channels:
                self.channels[matched_id] = np.zeros(
                    (self.grid_h, self.grid_w), dtype=np.float32)

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

        # 6) 오래된 채널 삭제 (max <= 0.5이면 제거)
        for f_id in list(self.channels.keys()):
            if f_id not in seen_ids and np.max(self.channels[f_id]) <= 0.5:
                del self.channels[f_id]
                self.last_centers.pop(f_id, None)
                self.last_sizes.pop(f_id, None)

        return final_boxes, final_ids


# 유틸 함수 
def centers_close(cx1, cy1, cx2, cy2, ref_w, ref_h, ratio=0.8):
    """두 중심점이 얼굴 크기 기준 ratio 이내인지 확인."""
    return abs(cx1 - cx2) < ref_w * ratio and abs(cy1 - cy2) < ref_h * ratio


def reset_tracker():
    """
    YOLO 내부 tracker 상태를 완전히 초기화한다.
    역방향 패스 시작 전 호출하여 정방향 tracker 상태를 제거한다.
    """
    try:
        face_detector.predictor = None
    except Exception:
        pass


# 임베딩 추출 
def extract_embedding(frame, box):
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    pad_w = int((x2 - x1) * 0.3)
    pad_h = int((y2 - y1) * 0.3)
    crop = frame[max(0, y1 - pad_h):min(img_h, y2 + pad_h),
                 max(0, x1 - pad_w):min(img_w, x2 + pad_w)]
    if crop.size == 0:
        return None
    faces = face_aligner.get(crop)
    if not faces:
        return None
    f = sorted(faces,
               key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
               reverse=True)[0]
    aligned = face_align.norm_crop(crop, landmark=f.kps, image_size=112)
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    t = adaface_transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = adaface_model(t)[0].cpu().numpy()
    return emb


def cosine_sim(e1, e2):
    a = e1.flatten() / (np.linalg.norm(e1) + 1e-8)
    b = e2.flatten() / (np.linalg.norm(e2) + 1e-8)
    return float(np.dot(a, b))


def check_identity(emb, known_embs):
    if emb is None or not known_embs:
        return False, -1.0
    best = max(cosine_sim(emb, k) for k in known_embs)
    return best > SIMILARITY_THRESHOLD, best

# GT 라벨 로드 
def load_labelme_boxes(label_dir: str) -> dict:
    """
    LabelMe json 폴더 → {frame_key: [[x1,y1,x2,y2], ...]}
    label 이름 무관하게 모든 rectangle shape를 박스로 읽음.
    """
    result = {}
    if not os.path.isdir(label_dir):
        return result
    for jf in sorted(os.listdir(label_dir)):
        if not jf.endswith(".json"):
            continue
        with open(os.path.join(label_dir, jf)) as f:
            lm = json.load(f)
        key = jf.replace(".json", "")
        boxes = []
        for s in lm.get("shapes", []):
            if s.get("shape_type") == "rectangle" and len(s["points"]) >= 2:
                boxes.append([
                    s["points"][0][0], s["points"][0][1],
                    s["points"][1][0], s["points"][1][1],
                ])
        result[key] = boxes
    return result


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def box_detected(gt_box, pred_boxes, thresh=0.3):
    """gt_box가 pred_boxes 중 하나라도 IoU thresh 이상으로 덮이면 True."""
    return any(iou(gt_box, p) >= thresh for p in pred_boxes)


# 등록자 임베딩 로드 
def load_known_embeddings(face_dir: str) -> list:
    exts = ["*.jpg", "*.jpeg", "*.png"]
    imgs = []
    for e in exts:
        imgs += glob.glob(os.path.join(face_dir, e))
    if not imgs:
        return []
    embs = []
    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = face_aligner.get(img)
        if not faces:
            print(f"  ⚠️ 얼굴 미검출: {img_path}")
            continue
        target = sorted(faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                        reverse=True)[0]
        aligned = face_align.norm_crop(img, landmark=target.kps, image_size=112)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        t = adaface_transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = adaface_model(t)[0].cpu().numpy()
        embs.append(emb)
        print(f"  등록자 임베딩 추출: {os.path.basename(img_path)}")
    return embs


# 신원 판단 공용 로직 (V2와 동일, 정방향/역방향 공통)
def run_identity_pass(
    frames_list,
    frame_width,
    frame_height,
    known_embeddings,
    pass_name="패스",
    embedding_interval=1,
    scene_change_threshold=30.0,
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
        global_identities : dict { track_id -> True(등록자) / False(미등록자) }

    역방향 패스일 때는 frames_list 를 이미 뒤집어서 넣으면 된다.
    tracking_data 도 frames_list 순서 그대로 반환되므로,
    역방향으로 넣었다면 호출 측에서 다시 reverse() 해서 정방향 인덱스에 맞춰야 한다.
    """
    print(f"🔄 {pass_name} 시작... "
          f"(총 {len(frames_list)} 프레임, 임베딩 interval={embedding_interval})")

    heatmap_engine    = HeatmapEngine(frame_width, frame_height, grid_scale=0.1)
    global_identities = {}   # {track_id: True / False}
    identity_votes    = {}   # {track_id: int}
    last_embed_frame  = {}   # {track_id: int}
    tracking_data     = []   # [(all_boxes, all_ids), ...]
    prev_frame        = None
    scene_changed     = False

    for idx, frame in enumerate(frames_list):
        # 씬 체인지 감지 
        scene_changed = False
        if prev_frame is not None:
            small_curr = cv2.resize(frame, (64, 64))
            small_prev = cv2.resize(prev_frame, (64, 64))
            diff = cv2.absdiff(small_curr, small_prev)
            if np.mean(diff) > scene_change_threshold:
                heatmap_engine.reset_memory()
                scene_changed = True
        prev_frame = frame.copy()

        # YOLO 트래킹 
        current_boxes, current_ids, current_confs = [], [], []
        try:
            results = face_detector.track(
                frame, persist=True, conf=CONF_THRESHOLD, imgsz=640,
                device=device, verbose=False)
            if (results and len(results) > 0
                    and results[0].boxes is not None
                    and results[0].boxes.id is not None):
                current_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
                current_ids   = results[0].boxes.id.int().cpu().tolist()
                current_confs = results[0].boxes.conf.cpu().tolist()
        except Exception as e:
            if idx % 100 == 0:
                print(f"  [{pass_name}] 프레임 {idx} track() 오류 스킵: {e}")

        all_boxes, all_ids = heatmap_engine.update_and_recover(
            current_boxes, current_ids, current_confs)

        # 신원 판단 
        for box, f_id in zip(all_boxes, all_ids):
            if global_identities.get(f_id) is True:
                continue

            last_idx = last_embed_frame.get(f_id, -999)
            should_extract = (
                scene_changed
                or (idx - last_idx) >= embedding_interval
            )
            if not should_extract:
                continue

            last_embed_frame[f_id] = idx

            try:
                embedding = extract_embedding(frame, box)
                is_reg, best_sim = check_identity(embedding, known_embeddings)

                if is_reg:
                    identity_votes[f_id] = identity_votes.get(f_id, 0) + 1
                    print(f"  ✅ [{pass_name}] ID {f_id} sim={best_sim:.3f} "
                          f"투표={identity_votes[f_id]}")
                    if identity_votes[f_id] >= VOTE_REQUIREMENT:
                        global_identities[f_id] = True
                        print(f"  ✅ [{pass_name}] ID {f_id} → 등록자 확정 "
                              f"(프레임 {idx})")
                else:
                    #print(f"      [{pass_name}] ID {f_id} sim={best_sim:.3f} → 미등록")
                    if f_id not in global_identities:
                        global_identities[f_id] = False

            except Exception:
                pass

        tracking_data.append((all_boxes, all_ids))

        if idx % 100 == 0:
            print(f"  [{pass_name}] {idx}/{len(frames_list)} 프레임 완료")

    print(f"✅ {pass_name} 완료 — "
          f"등록자 ID: {[k for k, v in global_identities.items() if v]}")
    return tracking_data, global_identities


# 두 패스의 신원 정보 병합
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

    merged = dict(fwd_identities)

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
            if rev_id not in rev_registered_ids:
                continue

            rx1, ry1, rx2, ry2 = map(int, rev_box)
            rcx = (rx1 + rx2) / 2
            rcy = (ry1 + ry2) / 2
            rw  = rx2 - rx1
            rh  = ry2 - ry1

            for fwd_box, fwd_id in zip(fwd_boxes, fwd_ids):
                fx1, fy1, fx2, fy2 = map(int, fwd_box)
                fcx = (fx1 + fx2) / 2
                fcy = (fy1 + fy2) / 2
                fw  = fx2 - fx1
                fh  = fy2 - fy1
                ref_w = max(rw, fw)
                ref_h = max(rh, fh)

                if centers_close(rcx, rcy, fcx, fcy, ref_w, ref_h, ratio=0.8):
                    if merged.get(fwd_id) is not True:
                        merged[fwd_id] = True
                        print(f"  ✅ [병합] 정방향 ID {fwd_id} → 등록자로 업데이트 "
                              f"(역방향 ID {rev_id}, 프레임 {frame_idx})")
                    break

    print(f"✅ 병합 완료 — 최종 등록자 ID: {[k for k, v in merged.items() if v]}")
    return merged


#  프레임별 모자이크 박스 수집
def get_mosaic_boxes(frame_idx, fwd_tracking, rev_tracking, merged_ids, rev_identities):
    """
    해당 프레임에서 모자이크 적용 대상 박스 목록을 반환한다.
    V2 렌더링 로직과 동일하게 centers_close() 기반으로 중복/등록자 여부를 판단한다.
    """
    fwd_boxes, fwd_ids = fwd_tracking[frame_idx]
    rev_boxes, rev_ids = rev_tracking[frame_idx]

    # 정방향: 등록자/비등록자 분리
    fwd_mosaic_boxes     = []
    fwd_registered_boxes = []

    for box, f_id in zip(fwd_boxes, fwd_ids):
        if merged_ids.get(f_id) is True:
            fwd_registered_boxes.append(box)
        else:
            fwd_mosaic_boxes.append(box)

    # 역방향 보강: 정방향에서 놓쳤거나 처음부터 가려진 얼굴
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

        # (b) 역방향에서도 등록자로 확정된 ID인지 확인
        if rev_identities.get(rev_id) is True:
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

    # 최종 모자이크 박스: 정방향 비등록 + 역방향 보강
    all_mosaic = fwd_mosaic_boxes + rev_extra_mosaic
    return [[float(v) for v in b] for b in all_mosaic]


# 영상 1개 측정 
def evaluate_video(video_name: str) -> dict | None:
    video_path      = f"videos/{video_name}.mp4"
    face_dir        = f"data/{video_name}/registered_face"
    reg_label_dir   = f"data/{video_name}/registered_sample"   # 등록자 GT
    unreg_label_dir = f"data/{video_name}/labels_labelme"      # 미등록자 GT

    # 필수 파일 확인
    if not os.path.exists(video_path):
        print(f"[{video_name}] 영상 없음, 스킵")
        return None
    if not os.path.isdir(face_dir) or not glob.glob(os.path.join(face_dir, "*.*")):
        print(f"[{video_name}] 등록자 사진 없음 ({face_dir}), 스킵")
        return None
    if not os.path.isdir(reg_label_dir):
        print(f"[{video_name}] 등록자 라벨 없음 ({reg_label_dir}), 스킵")
        return None

    print(f"\n{'='*55}\n  📹 {video_name}\n{'='*55}")

    # 등록자 임베딩 준비
    known_embs = load_known_embeddings(face_dir)
    if not known_embs:
        print(f"[{video_name}] 등록자 임베딩 추출 실패, 스킵")
        return None

    # GT 라벨 로드
    reg_gt   = load_labelme_boxes(reg_label_dir)
    unreg_gt = load_labelme_boxes(unreg_label_dir)

    # 프레임 로드
    cap = cv2.VideoCapture(video_path)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(frm)
    cap.release()
    total = len(frames)
    print(f"  총 {total}프레임 로드 완료")

    # 정방향 패스 
    print("\n" + "─" * 50)
    print("▶  정방향 패스")
    print("─" * 50)
    reset_tracker()
    fwd_tracking, fwd_identities = run_identity_pass(
        frames_list=frames,
        frame_width=fw,
        frame_height=fh,
        known_embeddings=known_embs,
        pass_name="정방향",
        embedding_interval=EMBEDDING_INTERVAL,
    )

    # 역방향 패스
    print("\n" + "─" * 50)
    print("◀  역방향 패스")
    print("─" * 50)
    reset_tracker()
    rev_tracking_rev, rev_identities = run_identity_pass(
        frames_list=list(reversed(frames)),
        frame_width=fw,
        frame_height=fh,
        known_embeddings=known_embs,
        pass_name="역방향",
        embedding_interval=3,
    )
    rev_tracking = list(reversed(rev_tracking_rev))

    # 신원 정보 병합 
    print("\n" + "─" * 50)
    merged_ids = merge_identities(
        fwd_identities=fwd_identities,
        rev_identities=rev_identities,
        fwd_tracking=fwd_tracking,
        rev_tracking=rev_tracking,
        total_frames=total,
    )

    # 디버그: merged_ids 상태 확인 
    reg_ids_list   = [k for k, v in merged_ids.items() if v is True]
    unreg_ids_list = [k for k, v in merged_ids.items() if v is False]
    print(f"\n  [디버그] merged_ids 총 {len(merged_ids)}개")
    print(f"    등록자(True) : {len(reg_ids_list)}개  ID={reg_ids_list}")
    print(f"    미등록(False): {len(unreg_ids_list)}개  ID={unreg_ids_list[:10]}")

    reg_keys   = sorted(reg_gt.keys())[:3]
    unreg_keys = sorted(unreg_gt.keys())[:3]
    print(f"    등록자 GT 키 샘플  : {reg_keys}")
    print(f"    미등록자 GT 키 샘플: {unreg_keys}")

    for si in [0, total // 4, total // 2]:
        mb = get_mosaic_boxes(si, fwd_tracking, rev_tracking, merged_ids, rev_identities)
        fb, fi = fwd_tracking[si]
        key = f"frame_{si:04d}"
        unreg_boxes = unreg_gt.get(key, [])
        print(f"    frame_{si:04d}: 전체={len(fb)}개 모자이크={len(mb)}개 "
              f"IDs={fi}  미등록GT={len(unreg_boxes)}개")

    # FRR 계산 (등록자 얼굴이 모자이크됐는지) 
    frr_total = 0
    frr_wrong = 0

    for f_idx in range(total):
        key = f"frame_{f_idx:04d}"
        gt_boxes = reg_gt.get(key, [])
        if not gt_boxes:
            continue
        mosaic_boxes = get_mosaic_boxes(
            f_idx, fwd_tracking, rev_tracking, merged_ids, rev_identities)
        for gt in gt_boxes:
            frr_total += 1
            if box_detected(gt, mosaic_boxes):
                frr_wrong += 1

    # FAR 계산 (미등록자 얼굴이 모자이크 안 됐는지) 
    reg_gt_all_frames = {}
    for key, boxes in reg_gt.items():
        try:
            f_idx = int(key.replace("frame_", ""))
            reg_gt_all_frames[f_idx] = boxes
        except ValueError:
            pass

    far_total = 0
    far_wrong = 0

    for f_idx in range(total):
        key = f"frame_{f_idx:04d}"
        gt_boxes = unreg_gt.get(key, [])
        if not gt_boxes:
            continue

        mosaic_boxes   = get_mosaic_boxes(
            f_idx, fwd_tracking, rev_tracking, merged_ids, rev_identities)
        reg_boxes_this = reg_gt_all_frames.get(f_idx, [])

        for gt in gt_boxes:
            # 등록자 위치와 겹치는 GT는 스킵 (등록자 박스가 섞인 경우)
            fwd_boxes, fwd_ids = fwd_tracking[f_idx]
            reg_pred_boxes = [b for b, i in zip(fwd_boxes, fwd_ids)
                            if merged_ids.get(i) is True]
            if any(iou(gt, rb) >= 0.3 for rb in reg_pred_boxes):
                continue
            if reg_boxes_this and any(iou(gt, rb) >= 0.3 for rb in reg_boxes_this):
                continue
            far_total += 1
            if not box_detected(gt, mosaic_boxes):
                far_wrong += 1

    frr = frr_wrong / frr_total if frr_total > 0 else None
    far = far_wrong / far_total if far_total > 0 else None

    print(f"\n  결과:")
    if frr is not None:
        print(f"  등록자 GT  {frr_total}개  →  FRR: {frr * 100:.1f}%")
    else:
        print("  등록자 라벨 없음")
    if far is not None:
        print(f"  미등록자 GT {far_total}개  →  FAR: {far * 100:.1f}%")
    else:
        print("  미등록자 라벨 없음")

    return {
        "video"    : video_name,
        "reg_gt"   : frr_total,
        "unreg_gt" : far_total,
        "frr_wrong": frr_wrong,
        "far_wrong": far_wrong,
        "FRR(%)"   : round(frr * 100, 2) if frr is not None else "-",
        "FAR(%)"   : round(far * 100, 2) if far is not None else "-",
    }


# 전체 실행 및 리포트
def print_report(results: list):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  동일인 판별 정확도 리포트")
    print(f"  임계값: {SIMILARITY_THRESHOLD}  /  투표 조건: {VOTE_REQUIREMENT}회")
    print(sep)
    print(f"  {'영상':<12} {'등록자GT':>8} {'FRR(%)':>8} {'미등록GT':>9} {'FAR(%)':>8}")
    print(f"  {'-'*55}")

    frr_vals, far_vals = [], []
    for r in results:
        frr_str = f"{r['FRR(%)']:>7.2f}%" if r['FRR(%)'] != "-" else "     -"
        far_str = f"{r['FAR(%)']:>7.2f}%" if r['FAR(%)'] != "-" else "     -"
        print(f"  {r['video']:<12} {r['reg_gt']:>8} {frr_str} "
              f"{r['unreg_gt']:>9} {far_str}")
        if r['FRR(%)'] != "-":
            frr_vals.append(r['FRR(%)'])
        if r['FAR(%)'] != "-":
            far_vals.append(r['FAR(%)'])

    print(f"  {'-'*55}")
    avg_frr = f"{sum(frr_vals) / len(frr_vals):.2f}%" if frr_vals else "-"
    avg_far = f"{sum(far_vals) / len(far_vals):.2f}%" if far_vals else "-"
    print(f"  {'평균':<12} {'':>8} {avg_frr:>8} {'':>9} {avg_far:>8}")
    print(sep)
    print(f"\n  FRR: 등록자가 모자이크된 비율 (낮을수록 좋음)")
    print(f"  FAR: 미등록자가 통과된 비율   (낮을수록 좋음)")
    print(sep)

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📊 CSV 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    all_results = []
    for vname in VIDEOS:
        r = evaluate_video(vname)
        if r:
            all_results.append(r)

    if all_results:
        print_report(all_results)
    else:
        print("❌ 처리된 영상이 없습니다.")
        print("   data/video_XX/registered_face/ 에 등록자 사진을 넣고")
        print("   data/video_XX/registered_sample/ 에 등록자 라벨을 만들어주세요.")