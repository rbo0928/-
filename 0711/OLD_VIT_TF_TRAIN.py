import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import math
import time
import torchvision.transforms.functional as TF
from tqdm import tqdm # [NEW] 匯入 tqdm


# --- 模組 1: 影像編碼器 (Vision Transformer 風格) ---
# 根據報告，我們需要一個能處理 640x200 影像的編碼器。
# 我們將實作一個簡化版的 ViT。
# 圖片裁切參數
CROP_TOP_PIXELS = 280
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

class CustomTopCrop:
    def __init__(self, top_pixels):
        self.top_pixels = top_pixels
    def __call__(self, img):
        return TF.crop(img, self.top_pixels, 0, ORIGINAL_HEIGHT - self.top_pixels, ORIGINAL_WIDTH)

class PatchEmbedding(nn.Module):
    """
    將影像轉換為 Patch Embeddings。
    報告建議：影像大小 640x200，Patch 大小 16x16。
    """
    def __init__(self, img_size=(200, 640), patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # 注意：影像維度順序為 (H, W)，所以 n_patches 的計算應對應
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # 一個卷積層即可實現 Patching 和 Embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x 的形狀: (batch_size, in_channels, H, W)
        x = self.proj(x)  # 形狀: (batch_size, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2) # 形狀: (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2) # 形狀: (batch_size, n_patches, embed_dim)
        return x

class ImageEncoder(nn.Module):
    """
    影像編碼器，使用 Transformer Encoder 來處理 Patches。
    報告建議：隱藏維度 768，12 層 Transformer，12 個注意力頭 (ViT-Base)。
    此處為簡化示範，使用 6 層和 8 個頭。
    """
    def __init__(self, img_size=(200, 640), patch_size=16, in_channels=3,
                 embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches

        # 加入 [CLS] token，用於匯總整個影像的資訊
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 加入位置編碼，讓模型知道每個 patch 的相對位置
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True # 輸入形狀為 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x 的形狀: (batch_size, C, H, W)
        B = x.shape[0]
        
        x = self.patch_embed(x) # (B, n_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, n_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        x = self.norm(x)
        return x[:, 0] # (B, embed_dim)

# --- 模組 2: 速度編碼器 ---
class SpeedEncoder(nn.Module):
    """將純量速度值編碼為高維向量。"""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )

    def forward(self, speed):
        return self.encoder(speed)

# --- 模組 3: 預測頭 ---
class PredictionHead(nn.Module):
    """根據最終的特徵向量，預測左右輪的速度（2個連續值）。"""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 2) # 輸出2個值：左輪速度、右輪速度
        )

    def forward(self, x):
        return self.head(x)

# --- 總模型: 多模態時序 Transformer ---
class PositionalEncoding(nn.Module):
    """為序列加入時序位置資訊。"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # batch_first=True
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: 形狀 (batch, seq_len, embedding_dim) """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultimodalTemporalTransformer(nn.Module):
    """融合影像和速度數據，並透過時序 Transformer 進行預測的主模型。"""
    def __init__(self, seq_len=20, img_size=(200, 640), embed_dim=768, depth=6, num_heads=8, temporal_depth=4, temporal_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.image_encoder = ImageEncoder(img_size=img_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.speed_encoder = SpeedEncoder(embed_dim=embed_dim)
        self.temporal_pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)

        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=temporal_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=temporal_depth)
        self.prediction_head = PredictionHead(embed_dim)

    def forward(self, image_sequence, speed_sequence):
        # image_sequence: (B, T, C, H, W)
        # speed_sequence: (B, T)
        batch_size, seq_len, _, _, _ = image_sequence.shape
        
        # 將序列維度與批次維度合併，以進行批次化的特徵提取
        image_sequence_flat = image_sequence.view(batch_size * seq_len, *image_sequence.shape[2:])
        speed_sequence_flat = speed_sequence.view(batch_size * seq_len, 1)

        img_feat_flat = self.image_encoder(image_sequence_flat)
        speed_feat_flat = self.speed_encoder(speed_sequence_flat)
        
        fused_feat_flat = img_feat_flat + speed_feat_flat
        
        # 將特徵還原為序列形狀 (B, T, embed_dim)
        fused_sequence = fused_feat_flat.view(batch_size, seq_len, self.embed_dim)

        sequence_with_pos = self.temporal_pos_encoder(fused_sequence)
        temporal_output = self.temporal_transformer(sequence_with_pos)
        
        final_feature = temporal_output[:, -1, :] # 取序列最後一個時間步的輸出來預測
        predicted_speeds = self.prediction_head(final_feature)
        return predicted_speeds


# --- 資料集與資料加載器 ---
class DrivingDataset(Dataset):
    """自動駕駛時序資料集"""
    def __init__(self, csv_path, root_dir, seq_len=20, transform=None):
        self.dataframe = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        
        # 為了確保每個序列都有完整的 seq_len 幀，資料集長度要減去 seq_len-1
        self.data_len = len(self.dataframe) - (self.seq_len - 1)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # 獲取從 idx 到 idx + seq_len 的數據片段
        sequence_df = self.dataframe.iloc[idx : idx + self.seq_len]
        
        # 讀取影像序列和速度序列
        image_sequence = []
        speed_sequence = []
        
        for _, row in sequence_df.iterrows():
            img_name = os.path.join(self.root_dir, row['img_path'])
            image = Image.open(img_name).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            image_sequence.append(image)
            
            # 獲取速度
            speed_sequence.append(row['speed_signed'])
            
        # 將列表堆疊成 Tensor
        image_tensor = torch.stack(image_sequence)
        speed_tensor = torch.tensor(speed_sequence, dtype=torch.float32)
        
        # 目標是序列中最後一幀的左右輪速度
        target_row = sequence_df.iloc[-1]
        target = torch.tensor([target_row['lwheel'], target_row['rwheel']], dtype=torch.float32)
        
        return image_tensor, speed_tensor, target

# --- 訓練主程式 ---
if __name__ == '__main__':
    # --- 超參數設定 ---
    DATA_ROOT = "./2025_07_14/1/"
    IMG_DIR = os.path.join(DATA_ROOT, 'recorded_images')
    TRAIN_CSV = os.path.join(DATA_ROOT, "train_data.csv")
    VAL_CSV = os.path.join(DATA_ROOT, "val_data.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "test_data.csv") # [NEW] 測試集路徑
    SAVE_PATH = "./" # 模型儲存路徑
    
    # 模型參數 (針對 8GB VRAM 的建議 - 提升版)
    SEQ_LENGTH = 20
    IMG_H, IMG_W = 200, 640
    EMBED_DIM = 512      # 提升嵌入維度
    DEPTH = 6            # 影像編碼器深度 (維持不變)
    NUM_HEADS = 8        # 提升影像編碼器注意力頭 (需為 EMBED_DIM 的因數)
    TEMPORAL_DEPTH = 4   # 時序 Transformer 深度 (維持不變)
    TEMPORAL_HEADS = 8   # 提升時序 Transformer 注意力頭 (需為 EMBED_DIM 的因數)

    # 訓練參數
    NUM_EPOCHS = 10
    BATCH_SIZE = 4 # 為適應更大的模型，稍微降低 Batch Size
    LEARNING_RATE = 1e-4
    
    # --- 資料轉換 ---
    # 定義影像的轉換流程
    data_transforms = transforms.Compose([
        CustomTopCrop(CROP_TOP_PIXELS),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomAffine(degrees=7, translate=(0.07, 0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 建立資料集與資料加載器 ---
    print("正在加載資料...")
    # 檢查檔案是否存在
    if not all(os.path.exists(p) for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]):
        print("錯誤: 找不到 train_data.csv, val_data.csv 或 test_data.csv。")
        print(f"請確認 '{DATA_ROOT}' 路徑下有正確的資料檔案。")
        print("將跳過執行過程。")
        exit()
    
    os.makedirs(SAVE_PATH, exist_ok=True) # 確保儲存路徑存在

    train_dataset = DrivingDataset(csv_path=TRAIN_CSV, root_dir=IMG_DIR, seq_len=SEQ_LENGTH, transform=data_transforms)
    val_dataset = DrivingDataset(csv_path=VAL_CSV, root_dir=IMG_DIR, seq_len=SEQ_LENGTH, transform=data_transforms)
    test_dataset = DrivingDataset(csv_path=TEST_CSV, root_dir=IMG_DIR, seq_len=SEQ_LENGTH, transform=data_transforms) # [NEW] 建立測試集

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True) # [NEW] 建立測試資料加載器
    
    print(f"訓練資料集大小: {len(train_dataset)} | 驗證資料集大小: {len(val_dataset)} | 測試資料集大小: {len(test_dataset)}")

    # --- 初始化模型、損失函數、優化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    model = MultimodalTemporalTransformer(
        seq_len=SEQ_LENGTH, img_size=(IMG_H, IMG_W), embed_dim=EMBED_DIM,
        depth=DEPTH, num_heads=NUM_HEADS,
        temporal_depth=TEMPORAL_DEPTH, temporal_heads=TEMPORAL_HEADS
    ).to(device)

    print(f"模型總參數數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # --- 訓練與驗證迴圈 ---
    best_val_loss = float('inf') # 初始化最佳驗證損失

    for epoch in range(NUM_EPOCHS):
        # --- 訓練 ---
        model.train()
        train_loss = 0.0
        # [NEW] 加入 tqdm 進度條
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [訓練]", leave=False)
        for images, speeds, targets in train_progress_bar:
            images = images.to(device)
            speeds = speeds.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                predictions = model(images, speeds)
                loss = criterion(predictions, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # --- 驗證 ---
        model.eval()
        val_loss = 0.0
        # [NEW] 加入 tqdm 進度條
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [驗證]", leave=False)
        with torch.no_grad():
            for images, speeds, targets in val_progress_bar:
                images = images.to(device)
                speeds = speeds.to(device)
                targets = targets.to(device)
                
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    predictions = model(images, speeds)
                    loss = criterion(predictions, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | "
              f"訓練損失: {avg_train_loss:.4f} | "
              f"驗證損失: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(SAVE_PATH, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"找到新的最佳模型，驗證損失為 {best_val_loss:.4f}，已儲存至 {model_save_path}")

    print("\n訓練完成！")

    # --- [NEW] 測試最佳模型 ---
    print("\n--- 開始測試最佳模型 ---")
    # 載入最佳模型權重
    best_model_path = os.path.join(SAVE_PATH, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"已成功載入模型: {best_model_path}")
    else:
        print("錯誤: 找不到最佳模型檔案 'best_model.pth'，將使用最終模型進行測試。")

    model.eval()
    test_loss = 0.0
    test_progress_bar = tqdm(test_loader, desc="[測試]", leave=True)
    with torch.no_grad():
        for images, speeds, targets in test_progress_bar:
            images = images.to(device)
            speeds = speeds.to(device)
            targets = targets.to(device)
            
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                predictions = model(images, speeds)
                loss = criterion(predictions, targets)
            
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print("-" * 50)
    print(f"最終測試損失 (MSE): {avg_test_loss:.4f}")
    print("-" * 50)

