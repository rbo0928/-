import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torchvision.transforms as transforms
import torchvision.models as models
import math
import pandas as pd
from PIL import Image
import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import random
import torchvision.transforms.functional as TF

# --- 優化後的參數設定 ---
DATA_DIR = '2025_08_14/'
TRAIN_DIR = os.path.join(DATA_DIR, '1')
VAL_DIR = os.path.join(DATA_DIR, '2')
TEST_DIR = os.path.join(DATA_DIR, '3')
TRAIN_CSV_PATH = os.path.join(TRAIN_DIR, 'log.csv')
VAL_CSV_PATH = os.path.join(VAL_DIR, 'log.csv')
TEST_CSV_PATH = os.path.join(TEST_DIR, 'log.csv')
BEST_MODEL_SAVE_PATH = 'best_transformer_driver_model.pth'

# 預覽開關
SHOW_PREVIEW = False
# 重複訓練開關
RETRAIN = True

# 優化後的模型與訓練參數
SEQUENCE_LENGTH = 15  # 減少序列長度以節省記憶體
BATCH_SIZE = 6        # 針對8G顯卡優化
EPOCHS = 80
LEARNING_RATE = 3e-4  # 提高初始學習率
IMG_HEIGHT = 224      # 降低解析度以節省記憶體
IMG_WIDTH = 224

# # Transformer 模型參數
# D_MODEL = 512
# N_HEAD = 8
# N_LAYERS = 3
# DROPOUT = 0.3

# # Transformer 模型參數2
# D_MODEL = 768
# N_HEAD = 12
# N_LAYERS = 6
# DROPOUT = 0.3

# Transformer 模型參數3
D_MODEL = 512
N_HEAD = 8
N_LAYERS = 4
DROPOUT = 0.15

# 圖片裁切參數
CROP_TOP_PIXELS = 260
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

# --- 1. 增強的資料集與轉換 ---
class CustomTopCrop:
    def __init__(self, top_pixels):
        self.top_pixels = top_pixels
    def __call__(self, img):
        return TF.crop(img, self.top_pixels, 0, ORIGINAL_HEIGHT - self.top_pixels, ORIGINAL_WIDTH)

class DrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir, sequence_length, transform=None, is_training=False):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.annotations) - self.sequence_length + 1

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.sequence_length
        
        target_row = self.annotations.iloc[end_index - 1]
        l_speed = target_row['lwheel']
        r_speed = target_row['rwheel']
        targets = torch.tensor([l_speed, r_speed], dtype=torch.float32)

        # 增強的資料增強策略
        apply_straight_aug = False
        apply_speed_aug = False
        shift_px = 0
        speed_noise = 0
        
        if self.is_training:
            is_straight = abs(l_speed - r_speed) < 1.5
            
            # 原有的直線增強
            if is_straight and random.random() < 0.4:
                apply_straight_aug = True
                shift_direction = random.choice([-1, 1])
                shift_px = random.randint(15, 40) * shift_direction
                correction_strength = random.uniform(1.8, 2.8)
                if shift_px < 0:
                    targets = torch.tensor([l_speed + correction_strength, r_speed - correction_strength], dtype=torch.float32)
                else:
                    targets = torch.tensor([l_speed - correction_strength, r_speed + correction_strength], dtype=torch.float32)
            
            # 新增速度雜訊增強
            elif random.random() < 0.2:
                apply_speed_aug = True
                speed_noise = random.uniform(-0.3, 0.3)
                targets = torch.tensor([l_speed + speed_noise, r_speed + speed_noise], dtype=torch.float32)

        sequence_images = []
        for i in range(start_index, end_index):
            img_name = self.annotations.iloc[i]['img_path']
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            
            if apply_straight_aug:
                image = image.transform(image.size, Image.AFFINE, (1, 0, -shift_px, 0, 1, 0))
            
            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)

        images_tensor = torch.stack(sequence_images)
        return images_tensor, targets

# --- 2. 優化的 Transformer 模型架構 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VisionTransformerDriver(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dropout, num_classes=2):
        super(VisionTransformerDriver, self).__init__()
        
        # 使用更輕量的 CNN backbone
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            efficientnet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(1280),
            nn.Dropout(0.3)
        )
        
        # 特徵投影層
        self.feature_proj = nn.Sequential(
            nn.Linear(1280, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer 編碼器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, 
            dim_feedforward=d_model * 2,
            dropout=dropout, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 分類 token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 輸出層
        self.output_fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.d_model = d_model
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # CNN 特徵提取
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x)
        features = self.feature_proj(features)
        features = features.view(batch_size, seq_len, self.d_model)
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, features), dim=1)
        
        # Transformer 處理
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        transformer_output = self.transformer_encoder(x)
        cls_output = transformer_output[:, 0, :]
        
        # 最終預測
        out = self.output_fc(cls_output)
        return out

# --- 改進的訓練函式 ---
def run_epoch(model, dataloader, criterion, optimizer, device, is_training, epoch_desc=""):
    model.train() if is_training else model.eval()
    running_loss = 0.0
    running_mae = 0.0
    
    # 混合精度訓練
    scaler = getattr(run_epoch, "scaler", None)
    if scaler is None and is_training:
        scaler = torch.amp.GradScaler('cuda')
        run_epoch.scaler = scaler
    
    progress_bar = tqdm(dataloader, desc=epoch_desc, leave=False)
    
    with torch.set_grad_enabled(is_training):
        for batch_idx, (sequences, targets) in enumerate(progress_bar):
            sequences, targets = sequences.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=is_training):
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                mae = torch.mean(torch.abs(outputs - targets))
            
            if is_training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            running_loss += loss.item() * sequences.size(0)
            running_mae += mae.item() * sequences.size(0)
            
            # 更新進度條
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}',
                'mem': f'{torch.cuda.memory_allocated(device)/1024/1024:.0f}MB'
            })
            
            # 記憶體管理
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    return epoch_loss, epoch_mae

def predict(model, image_sequence, transform, device):
    model.to(device)
    model.eval()
    processed_sequence = [transform(img.convert('RGB')) for img in image_sequence]
    input_tensor = torch.stack(processed_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.cpu().numpy().flatten()

# --- 主程式 ---
if __name__ == '__main__':
    writer = SummaryWriter()
    
    # 檢查資料檔案
    if not all([os.path.exists(p) for p in [TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH]]):
        print(f"錯誤：找不到分割好的資料檔案。請先執行 '1_create_split_files.py'。")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"將使用設備: {device}")

        # 優化的資料轉換
        train_transform = transforms.Compose([
            CustomTopCrop(CROP_TOP_PIXELS),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0)),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_test_transform = transforms.Compose([
            CustomTopCrop(CROP_TOP_PIXELS),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 建立資料集
        train_dataset = DrivingDataset(TRAIN_CSV_PATH, os.path.join(TRAIN_DIR, 'recorded_images'), 
                                     SEQUENCE_LENGTH, transform=train_transform, is_training=True)
        val_dataset = DrivingDataset(VAL_CSV_PATH, os.path.join(VAL_DIR, 'recorded_images'), 
                                   SEQUENCE_LENGTH, transform=val_test_transform, is_training=False)
        test_dataset = DrivingDataset(TEST_CSV_PATH, os.path.join(TEST_DIR, 'recorded_images'), 
                                    SEQUENCE_LENGTH, transform=val_test_transform, is_training=False)

        # 優化的資料載入器
        num_workers = 4  # 針對8G顯卡減少工作程序
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=num_workers, pin_memory=True, persistent_workers=True)

        print(f"資料載入完成 -> 訓練集: {len(train_dataset)} | 驗證集: {len(val_dataset)} | 測試集: {len(test_dataset)}")
        
        if torch.cuda.is_available():
            print(f"GPU顯存總量: {torch.cuda.get_device_properties(device).total_memory/1024/1024:.1f} MB")

        # 建立模型
        model = VisionTransformerDriver(D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
        
        if os.path.exists(BEST_MODEL_SAVE_PATH) and RETRAIN:
            print("使用之前的模型開始訓練")
            model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=device))

        # 優化器和排程器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5, 
                               betas=(0.9, 0.95), eps=1e-8)

        # 修正 OneCycleLR 排程器設置
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )

        # early stopping 參數調整
        patience = 30
        epochs_no_improve = 0
        best_val_loss = float('inf')
        best_val_mae = float('inf')

        print("\n--- 開始訓練與驗證 (已啟用早停法與OneCycle學習率排程) ---")
        
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            # 訓練階段
            train_desc = f"訓練中 Epoch {epoch+1:02d}/{EPOCHS}"
            train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, 
                                            device, is_training=True, epoch_desc=train_desc)
            scheduler.step()

            # 驗證階段
            val_desc = f"驗證中 Epoch {epoch+1:02d}/{EPOCHS}"
            val_loss, val_mae = run_epoch(model, val_loader, criterion, None, 
                                        device, is_training=False, epoch_desc=val_desc)

            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            # 記錄指標
            print(f'週期 [{epoch+1:02d}/{EPOCHS}] | 訓練損失: {train_loss:.4f} | 驗證損失: {val_loss:.4f} | '
                  f'訓練MAE: {train_mae:.4f} | 驗證MAE: {val_mae:.4f} | 學習率: {current_lr:.1e} | 耗時: {epoch_time:.2f}s')

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MAE/train', train_mae, epoch)
            writer.add_scalar('MAE/val', val_mae, epoch)
            writer.add_scalar('LR', current_lr, epoch)

            # 改進的模型保存策略
            save_condition = (val_loss < best_val_loss) or (val_mae < best_val_mae and val_loss < best_val_loss * 1.05)

            if save_condition:
                print(f'  驗證指標改善。儲存模型至 {BEST_MODEL_SAVE_PATH}')
                best_val_loss = min(best_val_loss, val_loss)
                best_val_mae = min(best_val_mae, val_mae)
                epochs_no_improve = 0
                torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            else:
                epochs_no_improve += 1
                print(f'  驗證指標未改善。計數: {epochs_no_improve}/{patience}')

            if epochs_no_improve >= patience:
                print(f"\n驗證損失已連續 {patience} 個週期未改善。觸發早停法！")
                break

        print("--- 訓練完成 ---")
        writer.close()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 最終測試評估
        print(f"\n--- 載入最佳模型進行最終評估 ---")
        model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=device))
        test_loss, test_mae = run_epoch(model, test_loader, criterion, None, device, 
                                      is_training=False, epoch_desc="測試中")
        print(f"--- 最終測試損失: {test_loss:.4f} | MAE: {test_mae:.4f} ---")
        
        # 預測範例
        print("\n--- 在測試集上執行預測範例 ---")
        if len(test_dataset) > 0:
            sample_sequence_tensor, true_speeds_tensor = test_dataset[0]
            sample_pil_images = []
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            
            for img_tensor in sample_sequence_tensor:
                inv_tensor = inv_normalize(img_tensor.cpu())
                pil_img = transforms.ToPILImage()(inv_tensor)
                sample_pil_images.append(pil_img)
            
            predicted_speeds = predict(model, sample_pil_images, val_test_transform, device)
            print(f"預測速度: 左={predicted_speeds[0]:.2f}, 右={predicted_speeds[1]:.2f}")
            print(f"真實速度: 左={true_speeds_tensor[0]:.2f}, 右={true_speeds_tensor[1]:.2f}")
        else:
            print("測試集為空，無法執行預測。")