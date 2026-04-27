import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import mediapipe as mp
from insightface.app import FaceAnalysis
import torchvision.models as models

# ==========================================
# 1. 모델 구조 정의 (학습 시와 동일)
# ==========================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(torch.mean(x, dim=[2, 3], keepdim=True))))
        max_out = self.fc2(self.relu(self.fc1(torch.amax(x, dim=[2, 3], keepdim=True))))
        x = x * self.sigmoid_channel(avg_out + max_out)
        avg_p = torch.mean(x, dim=1, keepdim=True)
        max_p = torch.amax(x, dim=1, keepdim=True)
        x = x * self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_p, max_p], dim=1)))
        return x

class SPP(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        self.pools = nn.ModuleList([nn.MaxPool2d(k, stride=1, padding=k//2) for k in pool_sizes])
    def forward(self, x):
        return torch.cat([x] + [pool(x) for pool in self.pools], dim=1)

class YOLOFaceDetector(nn.Module):
    def __init__(self, num_classes=1, num_anchors=3):
        super(YOLOFaceDetector, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.out_channels = num_anchors * (5 + num_classes + 1)
        mobilenet = models.mobilenet_v2(pretrained=False).features
        self.stage1 = mobilenet[:7]
        self.stage2 = mobilenet[7:14]
        self.stage3 = mobilenet[14:]
        self.conv_p5 = nn.Conv2d(1280, 512, 1); self.cbam_p5 = CBAM(512); self.spp = SPP()
        self.conv_spp = nn.Conv2d(512 * 4, 512, 1); self.up_p5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_p4_1 = nn.Conv2d(96, 256, 1)
        self.conv_p4_2 = nn.Sequential(nn.Conv2d(512 + 256, 256, 3, padding=1), nn.ReLU())
        self.cbam_p4 = CBAM(256); self.up_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_p3_1 = nn.Conv2d(32, 128, 1)
        self.conv_p3_2 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1), nn.ReLU())
        self.cbam_p3 = CBAM(128)
        self.head_p5 = self._make_head(512); self.head_p4 = self._make_head(256); self.head_p3 = self._make_head(128)

    def _make_head(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, self.out_channels, 1)
        )

    def forward(self, x):
        p3_feat = self.stage1(x); p4_feat = self.stage2(p3_feat); p5_feat = self.stage3(p4_feat)
        p5 = self.conv_spp(self.spp(self.cbam_p5(self.conv_p5(p5_feat))))
        p5_up = self.up_p5(p5); p4 = self.cbam_p4(self.conv_p4_2(torch.cat([p5_up, self.conv_p4_1(p4_feat)], dim=1)))
        p4_up = self.up_p4(p4); p3 = self.cbam_p3(self.conv_p3_2(torch.cat([p4_up, self.conv_p3_1(p3_feat)], dim=1)))
        def reshape_output(out):
            batch, _, h, w = out.shape
            return out.permute(0, 2, 3, 1).contiguous().view(batch, h, w, self.num_anchors, -1)
        return [reshape_output(self.head_p3(p3)), reshape_output(self.head_p4(p4)), reshape_output(self.head_p5(p5))]

# ==========================================
# 2. 유틸리티 (디코딩 & 트래커)
# ==========================================
class SimpleTracker:
    def __init__(self, dist_thresh=120, max_lost=30):
        self.next_id = 1
        self.tracks = {}      # {id: last_center_pt}
        self.lost_counts = {} # {id: missed_frames_count}
        self.dist_thresh = dist_thresh # 거리가 멀어도 따라가도록 임계값 상향
        self.max_lost = max_lost       # 30프레임(약 1초) 동안 안 보여도 ID 유지

    def update(self, bboxes):
        new_tracks = {}
        ids = []
        
        # 현재 프레임의 모든 박스에 대해 기존 ID 매칭 시도
        for box in bboxes:
            cx, cy = box[0] + box[2]//2, box[1] + box[3]//2
            matched_id = -1
            min_dist = self.dist_thresh
            
            for tid, last_pt in self.tracks.items():
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(last_pt))
                if dist < min_dist:
                    matched_id = tid
                    min_dist = dist
            
            if matched_id != -1:
                # 기존 ID 매칭 성공
                new_tracks[matched_id] = (cx, cy)
                self.lost_counts[matched_id] = 0 # 실종 카운트 초기화
                ids.append(matched_id)
            else:
                # 새로운 ID 부여
                new_tracks[self.next_id] = (cx, cy)
                self.lost_counts[self.next_id] = 0
                ids.append(self.next_id)
                self.next_id += 1
        
        # 이번 프레임에서 매칭되지 않은 기존 트랙들도 '기억' 유지
        for tid in list(self.tracks.keys()):
            if tid not in new_tracks:
                self.lost_counts[tid] += 1
                if self.lost_counts[tid] <= self.max_lost:
                    # 아직 삭제하지 않고 마지막 위치 유지
                    new_tracks[tid] = self.tracks[tid]
        
        self.tracks = new_tracks
        return ids

def decode_prediction(preds, img_w, img_h, device):
    anchors = [[[10,13],[16,30],[33,23]], [[30,61],[62,45],[59,119]], [[116,90],[156,198],[373,326]]]
    strides = [8, 16, 32]
    boxes, scores = [], []
    for i, pred in enumerate(preds):
        stride = strides[i]
        anchor = torch.tensor(anchors[i]).to(device)
        p = pred.clone()
        p[..., 0:2] = torch.sigmoid(p[..., 0:2]); p[..., 4:] = torch.sigmoid(p[..., 4:])
        mask = p[..., 4] > 0.45
        if mask.any():
            for batch, y, x, a in mask.nonzero():
                val = p[batch, y, x, a]
                cx = (val[0] + x) * stride * (img_w / 640)
                cy = (val[1] + y) * stride * (img_h / 640)
                w = torch.exp(val[2]) * anchor[a][0] * (img_w / 640)
                h = torch.exp(val[3]) * anchor[a][1] * (img_h / 640)
                boxes.append([int(cx-w/2), int(cy-h/2), int(w), int(h)])
                scores.append(val[4].item())
    if not boxes: return [], []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.45)
    if len(idxs) == 0: return [], []
    return [boxes[i] for i in idxs.flatten()], [scores[i] for i in idxs.flatten()]

# ==========================================
# 3. 메인 실행 환경 설정
# ==========================================
SIMILARITY_THRESHOLD = 0.40 # 인식 임계값

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = YOLOFaceDetector().to(device)
detector.load_state_dict(torch.load('checkpoints/best_yolo_face_torch.pth', map_location=device))
detector.eval()

analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

# 0.10.11 버전에서 solutions 정상 작동 확인됨
mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

tracker = SimpleTracker()

# 인물 등록 (Test_person 폴더)
known_embs = []
print("🔄 인물 등록 중...")
for p in glob.glob("Test_person/*.*"):
    img = cv2.imread(p)
    if img is not None:
        faces = analyzer.get(img)
        if faces:
            known_embs.append(faces[0].normed_embedding)
            print(f"✅ {os.path.basename(p)} 등록 완료")

# 영상 처리
cap = cv2.VideoCapture("Test_video/input_video1.mp4")
out = cv2.VideoWriter('Test_video/final_recognition_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                       cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

checked_ids = {} # {id: is_known_bool}

print("\n🎬 영상 처리를 시작합니다...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Detection
    input_img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640))
    input_tensor = torch.from_numpy(input_img).permute(2,0,1).unsqueeze(0).float().div(255).to(device)
    with torch.no_grad():
        preds = detector(input_tensor)
    
    boxes, scores = decode_prediction(preds, frame.shape[1], frame.shape[0], device)
    ids = tracker.update(boxes)

    # 2. Recognition & Logic
    for box, f_id in zip(boxes, ids):
        x, y, w, h = box
        # 화면 경계 처리
        img_h, img_w = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 신원 확인 (ID당 최초 1회만 수행하여 부하 감소)
        if f_id not in checked_ids:
            # [핵심 수정] 기존에 잘 작동하던 30% 가변 여백 로직 적용
            pad_w = int(w * 0.3)
            pad_h = int(h * 0.3)
            rx1 = max(0, x - pad_w)
            ry1 = max(0, y - pad_h)
            rx2 = min(img_w, x + w + pad_w)
            ry2 = min(img_h, y + h + pad_h)
            
            face_crop = frame[ry1:ry2, rx1:rx2]
            
            if face_crop.size > 0:
                # InsightFace로 특징 추출
                faces_in_crop = analyzer.get(face_crop)
                
                if faces_in_crop:
                    # 가장 큰 얼굴 선택
                    target_face = sorted(faces_in_crop, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    target_embed = target_face.normed_embedding
                    
                    best_sim = -1.0
                    for known_emb in known_embs:
                        sim = np.dot(target_embed, known_emb)
                        if sim > best_sim:
                            best_sim = sim
                    
                    # 결과 판정
                    if best_sim > SIMILARITY_THRESHOLD:
                        checked_ids[f_id] = True
                        print(f"🟢 ID {f_id} 인식 성공! (유사도: {best_sim:.3f})")
                    else:
                        checked_ids[f_id] = False
                        print(f"🔴 ID {f_id} 모르는 사람입니다. (최고 유사도: {best_sim:.3f})")
                else:
                    # 얼굴을 못 찾은 경우 로그 출력 후 다음 프레임에서 재시도하게 둠
                    print(f"⚠️ ID {f_id}: 얼굴 특징 추출 실패 (이미지가 너무 작거나 흐림)")
                    continue

        # 결과 적용 (블러/박스)
        if checked_ids.get(f_id, False):
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"KNOWN:{f_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 모르는 사람 블러 로직 (기존과 동일)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                res = mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                if res.multi_face_landmarks:
                    pts = np.array([(int(l.x*roi.shape[1]), int(l.y*roi.shape[0])) for l in res.multi_face_landmarks[0].landmark])
                    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
                    blur = cv2.GaussianBlur(roi, (91, 91), 30)
                    frame[y1:y2, x1:x2] = np.where(mask[...,None]==255, blur, roi)

    out.write(frame)
    cv2.imshow('Final System', frame)
    if cv2.waitKey(1) == 27: break

cap.release(); out.release(); cv2.destroyAllWindows()
print("\n✅ 모든 처리가 완료되었습니다!")