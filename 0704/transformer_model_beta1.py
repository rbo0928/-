import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # 【修改點】匯入學習率排程器
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

# --- 參數設定 ---
# 資料路徑 (讀取分割好的檔案)
DATA_DIR = '2025_07_02/2/'
IMG_DIR = os.path.join(DATA_DIR, 'recorded_images')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train_data.csv')
VAL_CSV_PATH = os.path.join(DATA_DIR, 'val_data.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test_data.csv')
BEST_MODEL_SAVE_PATH = 'best_transformer_driver_model.pth'

# 預覽開關
SHOW_PREVIEW = True

# 模型與訓練參數
SEQUENCE_LENGTH = 20
BATCH_SIZE = 8
EPOCHS = 40 # 增加週期以給予早停法足夠的觀察空間
LEARNING_RATE = 1e-4 # 可以適當提高初始學習率，由排程器自動調整
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 圖片裁切參數
CROP_TOP_PIXELS = 280
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

# Transformer 模型參數
D_MODEL = 512
N_HEAD = 8
N_LAYERS = 3
DROPOUT = 0.4 # 【修改點】增加Dropout以加強正規化

# --- 1. 自定義資料集與轉換 ---
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

        apply_straight_aug = False
        shift_px = 0
        
        if self.is_training:
            is_straight = abs(l_speed - r_speed) < 1.5
            if is_straight and random.random() < 0.5:
                apply_straight_aug = True
                shift_direction = random.choice([-1, 1])
                shift_px = random.randint(20, 50) * shift_direction
                correction_strength = 2.5
                if shift_px < 0:
                    targets = torch.tensor([l_speed + correction_strength, r_speed - correction_strength], dtype=torch.float32)
                else:
                    targets = torch.tensor([l_speed - correction_strength, r_speed + correction_strength], dtype=torch.float32)

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

# --- 2. Transformer 模型架構 ---
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
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.output_fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x).view(batch_size, seq_len, self.d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, features), dim=1)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        transformer_output = self.transformer_encoder(x)
        cls_output = transformer_output[:, 0, :]
        out = self.output_fc(cls_output)
        return out

# --- 預覽與訓練函式 ---
def show_prediction_preview(sequences_batch, targets_batch, outputs_batch, mode="Validation"):
    # ... (此部分程式碼不變) ...
    batch_previews = []
    batch_size = sequences_batch.size(0)
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    for i in range(batch_size):
        image_tensor = sequences_batch[i, -1, :, :, :]
        true_speeds = targets_batch[i].cpu().numpy()
        pred_speeds = outputs_batch[i].cpu().detach().numpy()
        img_display = inv_normalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        img_display = (img_display * 255).astype(np.uint8)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        text_canvas = np.zeros((IMG_HEIGHT, 400, 3), dtype=np.uint8)
        mode_color = (0, 255, 255) if mode == "Training" else (255, 255, 0)
        cv2.putText(text_canvas, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(text_canvas, "Ground Truth", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(text_canvas, f"L: {true_speeds[0]:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(text_canvas, f"R: {true_speeds[1]:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(text_canvas, "Prediction", (210, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(text_canvas, f"L: {pred_speeds[0]:.2f}", (210, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(text_canvas, f"R: {pred_speeds[1]:.2f}", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        preview_row = np.hstack((img_display, text_canvas))
        batch_previews.append(preview_row)
    if batch_previews:
        final_preview = np.vstack(batch_previews)
        cv2.imshow("即時訓練預覽 (Live Training Preview)", final_preview)
        cv2.waitKey(1)

def run_epoch(model, dataloader, criterion, optimizer, device, is_training, epoch_desc=""):
    # ... (此部分程式碼不變) ...
    model.train() if is_training else model.eval()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=epoch_desc, leave=False)
    with torch.set_grad_enabled(is_training):
        for sequences, targets in progress_bar:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            if SHOW_PREVIEW and is_training:
                mode = "Training"
                show_prediction_preview(sequences, targets, outputs, mode=mode)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * sequences.size(0)
            progress_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def predict(model, image_sequence, transform, device):
    # ... (此部分程式碼不變) ...
    model.to(device)
    model.eval()
    processed_sequence = [transform(img.convert('RGB')) for img in image_sequence]
    input_tensor = torch.stack(processed_sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.cpu().numpy().flatten()

# --- 主程式 ---
if __name__ == '__main__':
    if not all([os.path.exists(p) for p in [TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH]]):
        print(f"錯誤：找不到分割好的資料檔案。請先執行 '1_create_split_files.py'。")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"將使用設備: {device}")

        train_transform = transforms.Compose([
            CustomTopCrop(CROP_TOP_PIXELS),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.RandomAffine(degrees=7, translate=(0.07, 0)),
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
        
        train_dataset = DrivingDataset(TRAIN_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform=train_transform, is_training=True)
        val_dataset = DrivingDataset(VAL_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform=val_test_transform, is_training=False)
        test_dataset = DrivingDataset(TEST_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform=val_test_transform, is_training=False)

        num_workers = 4
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

        print(f"資料載入完成 -> 訓練集: {len(train_dataset)} | 驗證集: {len(val_dataset)} | 測試集: {len(test_dataset)}")
        print(f"使用 {num_workers} 個子程序進行資料載入。")

        model = VisionTransformerDriver(D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
        criterion = nn.MSELoss()
        # 【修改點】在優化器中加入 weight_decay
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        # 【修改點】定義學習率排程器
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

        # 早停法變數初始化
        patience = 5 # 稍微增加早停的耐心，因為排程器會幫助模型跳出局部最優
        epochs_no_improve = 0
        best_val_loss = float('inf')

        print("\n--- 開始訓練與驗證 (已啟用早停法與學習率排程) ---")
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            train_desc = f"訓練中 Epoch {epoch+1:02d}/{EPOCHS}"
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True, epoch_desc=train_desc)
            
            val_desc = f"驗證中 Epoch {epoch+1:02d}/{EPOCHS}"
            val_loss = run_epoch(model, val_loader, criterion, None, device, is_training=False, epoch_desc=val_desc)
            
            # 【修改點】根據驗證損失調整學習率
            scheduler.step(val_loss)

            epoch_time = time.time() - start_time
            print(f'週期 [{epoch+1:02d}/{EPOCHS}] | 訓練損失: {train_loss:.4f} | '
                  f'驗證損失: {val_loss:.4f} | 耗時: {epoch_time:.2f}s')

            # 早停法邏輯判斷
            if val_loss < best_val_loss:
                print(f'  驗證損失從 {best_val_loss:.4f} 改善至 {val_loss:.4f}。儲存模型至 {BEST_MODEL_SAVE_PATH}')
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            else:
                epochs_no_improve += 1
                print(f'  驗證損失未改善。計數: {epochs_no_improve}/{patience}')

            if epochs_no_improve >= patience:
                print(f"\n驗證損失已連續 {patience} 個週期未改善。觸發早停法！")
                break

        print("--- 訓練完成 ---")
        
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

        # 載入表現最好的模型進行最終測試
        print(f"\n--- 載入最佳模型 (驗證損失: {best_val_loss:.4f}) 進行最終評估 ---")
        model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH))

        test_loss = run_epoch(model, test_loader, criterion, None, device, is_training=False, epoch_desc="測試中")
        print(f"--- 最終測試損失: {test_loss:.4f} ---")
        
        print("\n--- 在測試集上執行預測範例 ---")
        if len(test_dataset) > 0:
            sample_sequence_tensor, true_speeds_tensor = test_dataset[0]
            sample_pil_images = []
            inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            for img_tensor in sample_sequence_tensor:
                inv_tensor = inv_normalize(img_tensor.cpu())
                pil_img = transforms.ToPILImage()(inv_tensor)
                sample_pil_images.append(pil_img)
            predicted_speeds = predict(model, sample_pil_images, val_test_transform, device)
            print(f"預測速度: 左={predicted_speeds[0]:.2f}, 右={predicted_speeds[1]:.2f}")
            print(f"真實速度: 左={true_speeds_tensor[0]:.2f}, 右={true_speeds_tensor[1]:.2f}")
        else:
            print("測試集為空，無法執行預測。")
