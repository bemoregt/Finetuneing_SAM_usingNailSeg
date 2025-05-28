import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from segment_anything import sam_model_registry
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import urllib.request
import zipfile
import glob

# FFT 기반 셀프어텐션 클래스 정의
class FFTSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 쿼리, 키, 밸류 프로젝션 레이어
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 쿼리, 키, 밸류 프로젝션
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # FFT 기반 어텐션 계산
        q_fft = torch.fft.rfft(q, dim=-1)
        k_fft = torch.fft.rfft(k, dim=-1)
        
        # 복소수 곱셈 (컨볼루션에 해당)
        res = q_fft * k_fft.conj()
        
        # 역변환
        attn_output = torch.fft.irfft(res, dim=-1, n=self.head_dim)
        
        # 결과와 밸류의 곱
        output = attn_output * v
        
        # 원래 형태로 변환
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output

# SAM 모델 수정하는 함수
def replace_attention_with_fft(model):
    """SAM 모델의 셀프어텐션 레이어를 FFT 버전으로 교체"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            # MultiheadAttention의 embed_dim 가져오기
            embed_dim = module.embed_dim
            # FFT 어텐션으로 교체
            setattr(model, name, FFTSelfAttention(dim=embed_dim, num_heads=module.num_heads, dropout=module.dropout))
        else:
            replace_attention_with_fft(module)
    
    return model

# SAM 모델 다운로드 함수
def download_sam_model(model_type="vit_b", save_dir="./sam_checkpoints"):
    """SAM 모델 체크포인트를 다운로드하는 함수"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    checkpoint_path = os.path.join(save_dir, f"sam_{model_type}.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"SAM {model_type} 모델 다운로드 중...")
        urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
    
    return checkpoint_path

# 네일 세그멘테이션 데이터셋 클래스
class NailSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(1024, 1024)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        
        # 이미지와 마스크 경로 설정
        self.images_dir = os.path.join(data_dir, "trainset_nails_segmentation/images")
        self.labels_dir = os.path.join(data_dir, "trainset_nails_segmentation/labels")
        
        print(f"이미지 디렉토리: {self.images_dir}")
        print(f"마스크 디렉토리: {self.labels_dir}")
        
        # 모든 이미지 파일 찾기
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for file_path in glob.glob(os.path.join(self.images_dir, ext)):
                # labels 폴더 내부의 파일은 제외
                if 'labels' not in file_path:
                    base_name = os.path.basename(file_path)
                    mask_path = os.path.join(self.labels_dir, base_name)
                    if os.path.exists(mask_path):
                        self.image_files.append((file_path, mask_path))
        
        print(f"총 {len(self.image_files)}개의 이미지-마스크 쌍을 찾았습니다.")
        for i in range(min(5, len(self.image_files))):
            print(f"파일 샘플 {i+1}: {self.image_files[i]}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_files[idx]
        
        # 이미지와 마스크 로드
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 그레이스케일로 로드
        
        # 이미지와 마스크 리사이즈
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)
        
        # 마스크를 이진 마스크로 변환 (임계값 127)
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8)  # 이진 마스크로 변환
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # 마스크를 텐서로 변환
        mask = torch.from_numpy(mask_np).long()
        
        return image, mask

# 커스텀 콜레이트 함수 (배치 생성)
def custom_collate_fn(batch):
    images = []
    masks = []
    
    for image, mask in batch:
        images.append(image)
        masks.append(mask)
    
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)
    
    return images, masks

# SAM 모델 수정 (학습을 위한 헤드 추가)
class ModifiedSAM(nn.Module):
    def __init__(self, sam_model, num_classes=2):  # 네일 세그멘테이션은 이진 분류(배경=0, 네일=1)
        super().__init__()
        self.sam = sam_model
        
        # SAM 출력 차원 가져오기 (임베딩 차원)
        self.embedding_dim = 256
        
        # 세그멘테이션 헤드 추가 (이진 분류)
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # 이미지 인코더 실행
        features = self.sam.image_encoder(x)  # [B, 256, H/16, W/16]
        
        # 특징맵 업샘플링 (원본 이미지 크기로)
        features = F.interpolate(features, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 세그멘테이션 헤드
        logits = self.seg_head(features)  # [B, num_classes, H, W]
        
        return logits

# 학습 함수
def train_sam_with_fft(model, dataloader, optimizer, criterion, device, num_epochs=5):
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순방향 전파
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return model

# 모델 평가 함수
def evaluate_model(model, dataloader, device):
    model.eval()
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # 순방향 전파
            outputs = model(images)
            
            # 클래스별 소프트맥스 확률
            probs = F.softmax(outputs, dim=1)
            
            # 가장 높은 확률을 가진 클래스 선택
            preds = torch.argmax(probs, dim=1)
            
            # Dice 점수 계산 (이진 분류의 경우 클래스 1에 대해서만)
            pred_mask = (preds == 1).float()
            true_mask = (masks == 1).float()
            
            intersection = (pred_mask * true_mask).sum()
            union = pred_mask.sum() + true_mask.sum()
            
            dice_score = (2.0 * intersection) / (union + 1e-6)  # 0으로 나누기 방지
            
            total_dice += dice_score.item()
            num_samples += 1
    
    avg_dice = total_dice / num_samples if num_samples > 0 else 0
    print(f"평균 Dice 점수: {avg_dice:.4f}")
    
    return avg_dice

# 메인 실행 함수
def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"사용 디바이스: {device}")
    
    # 데이터 디렉토리 설정 (이미지에서 확인한 정확한 경로)
    data_dir = "/Users/m1_4k/Pictures/nail_seg"  # 이미지에 맞게 수정
    
    # SAM 모델 체크포인트 다운로드
    print("SAM 모델 준비 중...")
    checkpoint_path = download_sam_model()
    
    # SAM 모델 불러오기
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    
    # 셀프어텐션을 FFT 버전으로 교체
    print("SAM 모델의 셀프어텐션을 FFT 버전으로 교체 중...")
    sam = replace_attention_with_fft(sam)
    
    # SAM 모델 수정 (세그멘테이션 헤드 추가, 네일 세그멘테이션은 이진 분류)
    model = ModifiedSAM(sam, num_classes=2)
    
    # 데이터 변환 설정
    target_size = (1024, 1024)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 네일 세그멘테이션 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = NailSegmentationDataset(
        data_dir=data_dir,
        transform=transform,
        target_size=target_size
    )
    
    # 데이터셋이 비어있으면 종료
    if len(dataset) == 0:
        print("데이터셋이 비어있습니다! 경로를 확인하세요.")
        return
    
    # 데이터셋 분할 (학습:검증 = 8:2)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    
    # 너무 작은 데이터셋인 경우 처리
    if split == 0 and dataset_size > 0:
        split = 1  # 최소 1개는 검증용으로 사용
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"학습 데이터 크기: {len(train_dataset)}, 검증 데이터 크기: {len(val_dataset)}")
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1,  # 메모리 제한으로 배치 크기 1
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # 학습 실행
    print("FFT 기반 SAM 모델 학습 시작...")
    trained_model = train_sam_with_fft(model, train_dataloader, optimizer, criterion, device, num_epochs=5)
    
    # 모델 평가
    print("모델 평가 중...")
    dice_score = evaluate_model(trained_model, val_dataloader, device)
    
    # 모델 저장
    save_path = "fft_sam_nails_segmentation.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"모델 학습 및 평가 완료! 저장 경로: {save_path}")
    print(f"최종 Dice 점수: {dice_score:.4f}")

if __name__ == "__main__":
    main()
