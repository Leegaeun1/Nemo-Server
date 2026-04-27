import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

# ==========================================
# 1. Attention & SPP 모듈 정의 - 모델이 어디에 집중할지를 결정!
# ==========================================
class CBAM(nn.Module):
    '''
    [어텐션 모듈] 가려진 얼굴에서 '눈'이나 '이마'처럼 노출된 중요한 특징을 
    모델이 더 강하게 인식하도록 채널(무엇)과 공간(어디) 차원에서 가중치를 부여합니다.
    '''
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention - 어떤 특징(색상, 질감 등)이 중요한가?
        # 1x1 컨볼루션 사용 -> 채널 수를 줄였다가 다시 늘리며 중요도 계산
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # Conv2d(입력 이미지의 채널 수(RGB는 3),출력 특징 맵의 채널 수 (필터 개수),필터의 크기(3x3이면 3),stride(필터가 이동하는 간격),padding(입력 데이터 가장자리에 추가하는 0의 폭))
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid() # 0~1사이의 가중치로 변환
        
        # Spatial Attention - 이미지의 어느 위치에 얼굴이 있는가?
        # 채널 평균과 최대값을 합쳐서 7x7 필터로 훑으며 위치 정보를 파악합니다.
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 1. 채널 어텐션 적용
        # 공간 정보를 압축(GAP, GMP)해서 채널별 중요도(가중치)를 구합니다.
        avg_out = self.fc2(self.relu(self.fc1(torch.mean(x, dim=[2, 3], keepdim=True)))) # 모든 채널의 평균
        max_out = self.fc2(self.relu(self.fc1(torch.amax(x, dim=[2, 3], keepdim=True)))) # 가장 강한 특징을 봄
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn # 중요한 채널은 키우고, 불필요한 채널은 죽입니다.
        
        # 2. 공간 어텐션 적용
        # 채널 정보를 압축해서 '중요한 영역'이 어디인지 가중치를 구합니다.
        avg_pool = torch.mean(x, dim=1, keepdim=True) # 모든 채널의 평균
        max_pool = torch.amax(x, dim=1, keepdim=True)
        # "특정 지점의 값이 주변 값들과 비교했을 때 의미 있는 패턴(예: 둥근 얼굴의 윤곽, 눈의 배치 등)을 형성하는가?"를 체크 ->0(무시)~1(중요)로 변환
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_pool, max_pool], dim=1))) 
        return x * spatial_attn # 얼굴이 있는 위치의 특징을 더 강조합니다.

class SPP(nn.Module):
    '''
    [공간 피라미드 풀링] 다양한 크기의 필터(5, 9, 13)를 거쳐 특징을 합칩니다.
    얼굴이 화면에 아주 작게 나오거나 아주 크게 나와도 놓치지 않게 해줍니다.
    '''
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        # 서로 다른 크기(5x5, 9x9, 13x13)의 맥스풀링 레이어를 준비합니다.
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2) for k in pool_sizes])

    def forward(self, x):
        # 원본과 각각의 풀링 결과를 옆으로(채널 방향으로) 길게 붙입니다.
        # 결과적으로 4배의 채널이 생성되며, 전역적인 맥락 정보를 얻게 됩니다.
        return torch.cat([x] + [pool(x) for pool in self.pools], dim=1)

# ==========================================
# 2. YOLO 모델 구축 - "얼굴 탐지기 설계도"
# ==========================================
class YOLOFaceDetector(nn.Module):
    def __init__(self, num_classes=1, num_anchors=3):
        super(YOLOFaceDetector, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 출력: 앵커당 [x, y, w, h, objectness, class, occlusion] 총 7개 값
        self.out_channels = num_anchors * (5 + num_classes + 1) # [x, y, w, h, obj, cls, occ]
        
        # [Step 1: Backbone] 이미지를 해석하는 뇌 (MobileNetV2)
        # ImageNet이라는 거대 데이터셋으로 '이미 사물을 보는 법'을 배운 MobileNetV2를 가져옵니다.
        # pretrained=True가 바로 '전이학습'의 시작입니다.
        mobilenet = models.mobilenet_v2(pretrained=True).features

        # 욜로는 다양한 크기를 탐지해야 하므로 백본을 3단계(P3, P4, P5)로 쪼개서 특징을 추출합니다.
        self.stage1 = mobilenet[:7]   # P3 (stride 8) - out: 32 channels # 1/8 작은 특징 (엣지, 질감)
        self.stage2 = mobilenet[7:14] # P4 (stride 16) - out: 96 channels # 1/16 중간 특징 (형태)
        self.stage3 = mobilenet[14:]  # P5 (stride 32) - out: 1280 channels # 1/32 고차원 특징 (의미 정보)

        # [2단계: Neck - 지식 가공하기]
        # 백본에서 나온 정보를 섞어서 얼굴 탐지에 최적화된 형태로 바꿉니다. (FPN 구조)
        # 여기에 우리가 추가한 CBAM(어텐션)이 들어가서 '가려짐'에 강해지도록 보정합니다.
        self.conv_p5 = nn.Conv2d(1280, 512, 1) # 채널 다이어트 (1280 -> 512)
        self.cbam_p5 = CBAM(512)
        self.spp = SPP()
        self.conv_spp = nn.Conv2d(512 * 4, 512, 1) # SPP로 늘어난 채널을 다시 512로 조정
        
        self.up_p5 = nn.Upsample(scale_factor=2, mode='nearest') # 1/32 -> 1/16로 확대
        self.conv_p4_1 = nn.Conv2d(96, 256, 1)
        self.conv_p4_2 = nn.Sequential(nn.Conv2d(512 + 256, 256, 3, padding=1), nn.ReLU())
        self.cbam_p4 = CBAM(256)
        
        self.up_p4 = nn.Upsample(scale_factor=2, mode='nearest') # 1/16 -> 1/8로 확대
        self.conv_p3_1 = nn.Conv2d(32, 128, 1)
        self.conv_p3_2 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1), nn.ReLU())
        self.cbam_p3 = CBAM(128)

        # [Step 3: Head] 최종 결과(박스 좌표 등)를 숫자로 뱉어냄
        # 최종적으로 박스 위치(x, y, w, h), 점수(obj), 클래스(cls), 가려짐(occ)을 출력합니다.
        self.head_p5 = self._make_head(512)
        self.head_p4 = self._make_head(256)
        self.head_p3 = self._make_head(128)

    def _make_head(self, in_channels):
        """박스를 예측하는 3개 층의 컨볼루션 블록 생성"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, self.out_channels, 1)
        )

    def forward(self, x):
        # 1. 백본 통과 (특징 추출)
        p3_feat = self.stage1(x)
        p4_feat = self.stage2(p3_feat)
        p5_feat = self.stage3(p4_feat)

        # 2. 목(Neck) 통과 (특징 융합)
        # P5 (가장 작은 맵) 처리
        p5 = self.conv_p5(p5_feat)
        p5 = self.cbam_p5(p5)
        p5 = self.conv_spp(self.spp(p5))
        
        # P4 처리 (P5 정보를 위로 올려서 합침)
        p5_up = self.up_p5(p5)
        p4 = self.conv_p4_1(p4_feat)
        p4 = self.cbam_p4(self.conv_p4_2(torch.cat([p5_up, p4], dim=1)))
        
        # P3 처리 (P4 정보를 위로 올려서 합침)
        p4_up = self.up_p4(p4)
        p3 = self.conv_p3_1(p3_feat)
        p3 = self.cbam_p3(self.conv_p3_2(torch.cat([p4_up, p3], dim=1)))

        # 3. 헤드 통과 (결과값 텐서 모양 정리)
        # [Batch, 21, H, W] -> [Batch, H, W, 3, 7] 순서로 변경 (YOLO 계산 편의성)
        out_p5 = self.head_p5(p5).permute(0, 2, 3, 1).contiguous().view(x.size(0), p5.size(2), p5.size(3), self.num_anchors, -1)
        out_p4 = self.head_p4(p4).permute(0, 2, 3, 1).contiguous().view(x.size(0), p4.size(2), p4.size(3), self.num_anchors, -1)
        out_p3 = self.head_p3(p3).permute(0, 2, 3, 1).contiguous().view(x.size(0), p3.size(2), p3.size(3), self.num_anchors, -1)

        return [out_p3, out_p4, out_p5]

    def freeze_backbone(self):
        """백본의 가중치를 고정(Freeze)하여 이미 배운 지식이 변하지 않게 보호합니다."""
        for param in self.stage1.parameters(): param.requires_grad = False
        for param in self.stage2.parameters(): param.requires_grad = False
        for param in self.stage3.parameters(): param.requires_grad = False

    def unfreeze_backbone(self):
        """백본 가중치 해제 (미세 조정용)"""
        for param in self.stage1.parameters(): param.requires_grad = True
        for param in self.stage2.parameters(): param.requires_grad = True
        for param in self.stage3.parameters(): param.requires_grad = True

# ==========================================
# 3. 데이터 로더 - 모델이 학습할 수 있도록 정답을 욜로 형식으로 변환
# ==========================================
class YOLODataset(Dataset):
    """
    이미지와 .txt 라벨을 읽어서 모델이 이해할 수 있는 텐서로 바꿉니다.
    핵심: 얼굴 크기에 가장 잘 맞는 앵커 박스(Anchor Box)를 골라 정답을 기록합니다.
    """
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
        self.label_dir = label_dir
        self.img_size = img_size
        
        # 미리 정의된 표준 얼굴 박스 크기 (YOLOv5 기준)
        self.anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],      # P3용 (작은 물체용)
            [[30, 61], [62, 45], [59, 119]],     # P4용 (중간 물체용)
            [[116, 90], [156, 198], [373, 326]]  # P5용 (큰 물체용)
        ], dtype=np.float32)
        self.strides = np.array([8, 16, 32]) # 각 특징맵의 축소 배수
        self.grid_sizes = [self.img_size // s for s in self.strides]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 1. 이미지 로드 (PyTorch는 Channel First: C, H, W)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0 # 0~1 정규화
        img = torch.from_numpy(img).permute(2, 0, 1) # (H,W,C) -> (C,H,W)
        
        # 2. 정답(Targets)을 담을 빈 그릇 만들기
        # 각 층마다 [Grid_H, Grid_W, Anchors, 7개정보] 형태
        targets = [torch.zeros((gs, gs, 3, 7)) for gs in self.grid_sizes]
        
        # 3. 라벨 읽기 및 정답 기록
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = map(float, parts[:5])
                        occ = 1.0 if '_extreme_occ' in img_path else 0.0 # 파일명에 따라 가려짐 마킹
                        
                        box_w, box_h = w * self.img_size, h * self.img_size
                        
                        # 4. 어떤 앵커 박스가 실제 얼굴과 가장 잘 어울리는지 찾기 (IoU)
                        best_iou, best_s, best_a = 0, -1, -1
                        for s_idx in range(3):
                            for a_idx in range(3):
                                aw, ah = self.anchors[s_idx, a_idx]
                                # 단순 교차 영역 계산 (박스 중심이 같다고 가정)
                                inter = min(box_w, aw) * min(box_h, ah)
                                union = box_w * box_h + aw * ah - inter
                                iou = inter / (union + 1e-6)
                                if iou > best_iou:
                                    best_iou, best_s, best_a = iou, s_idx, a_idx
                        
                        # 5. 찾은 위치의 그리드에 정답 정보 채워넣기
                        if best_s != -1:
                            stride = self.strides[best_s]
                            grid_x, grid_y = int(cx * self.img_size // stride), int(cy * self.img_size // stride)
                            
                            if 0 <= grid_x < self.grid_sizes[best_s] and 0 <= grid_y < self.grid_sizes[best_s]:
                                # 좌표값 오프셋 계산 (0~1 사이값)
                                tx = (cx * self.img_size) / stride - grid_x
                                ty = (cy * self.img_size) / stride - grid_y
                                aw, ah = self.anchors[best_s, best_a]
                                # 너비와 높이는 로그 스케일로 정규화 (YOLO 공식)
                                tw = np.log(box_w / aw + 1e-16)
                                th = np.log(box_h / ah + 1e-16)
                                
                                targets[best_s][grid_y, grid_x, best_a] = torch.tensor([tx, ty, tw, th, 1.0, cls, occ])
                                
        return img, targets

# ==========================================
# 5. 손실 함수 - "예측과 실제의 차이를 점수로 매기기"
# ==========================================
class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum') # 좌표 계산용
        self.bce = nn.BCEWithLogitsLoss(reduction='sum') # 유무/분류 계산용

    def forward(self, preds, targets):
        tot_loss = 0
        for i in range(3): # P3, P4, P5 각 층마다 계산
            pred = preds[i]
            target = targets[i].to(pred.device)
            
            # 마스크: 객체가 있는 곳(obj)과 없는 곳(noobj)을 나눔
            obj_mask = target[..., 4] == 1
            noobj_mask = target[..., 4] == 0

            # 1. 객체가 있는 칸의 손실 (좌표 + 존재감 + 클래스 + 가려짐)
            if obj_mask.sum() > 0:
                # 좌표 Loss (x, y, w, h)
                coord_loss = self.mse(pred[obj_mask][:, :4], target[obj_mask][:, :4]) * 5.0
                # Objectness Loss (객체 있음)
                obj_loss = self.bce(pred[obj_mask][:, 4], target[obj_mask][:, 4]) * 1.0
                # Class & Occlusion Loss
                cls_loss = self.bce(pred[obj_mask][:, 5], target[obj_mask][:, 5]) * 1.0
                occ_loss = self.bce(pred[obj_mask][:, 6], target[obj_mask][:, 6]) * 0.5 # 가려짐 가중치
            else:
                coord_loss, obj_loss, cls_loss, occ_loss = 0, 0, 0, 0

            # 2. 객체가 없는 칸(배경)의 손실 (가짜를 잘 걸러내는지)
            noobj_loss = self.bce(pred[noobj_mask][:, 4], target[noobj_mask][:, 4]) * 0.5
            
            tot_loss += (coord_loss + obj_loss + noobj_loss + cls_loss + occ_loss)
            
        return tot_loss / preds[0].size(0) # 배치의 평균 손실 반환

# ==========================================
# 5. 메인 학습 루프
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 사용 중인 디바이스: {device}")

    # 데이터셋 준비
    train_dataset = YOLODataset('dataset_extreme_occ/images/train', 'dataset_extreme_occ/labels/train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # 모델 및 Loss 세팅
    model = YOLOFaceDetector().to(device)
    criterion = YOLOLoss()
    
    # ====== [Stage 1: Backbone 동결] ======
    # 이유: 새로 만든 Neck과 Head는 초기값이 무작위라 처음에 크게 흔들립니다.
    # 이때 백본까지 같이 학습하면, 백본이 가진 좋은 지식(사물 인식 능력)이 망가집니다.
    # 그래서 백본은 딱딱하게 굳혀두고(Freeze), 뒤쪽의 새로운 레이어들만 먼저 학습시킵니다.
    print("\n[Stage 1] Backbone 동결 후 Head 학습 시작...")
    model.freeze_backbone()

    # filter(lambda p: p.requires_grad, ...) -> 고정되지 않은 파라미터만 최적화함
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10): # 예시로 10 에포크만
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            preds = model(imgs)
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    # ====== [Stage 2: 전체 해제 후 미세조정] ======
    # 이유: 이제 뒷부분이 어느 정도 얼굴을 찾을 줄 알게 되었습니다. 
    # 이제 백본의 잠금을 해제(Unfreeze)하고, 백본도 우리 데이터(가려진 얼굴)에 맞춰 
    # 아주 조금씩 변하게 합니다. 이때는 지식이 급격히 변하면 안 되므로 학습률(lr)을 1/10로 낮춥니다.
    print("\n[Stage 2] 전체 가중치 Fine-tuning 시작...")
    model.unfreeze_backbone()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4) # lr 낮춤
    
    for epoch in range(50):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50")
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            
    # 가중치 저장
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/best_yolo_face_torch.pth')
    print("\n✅ 파이토치 모델 학습 및 저장 완료!")

if __name__ == '__main__':
    train()