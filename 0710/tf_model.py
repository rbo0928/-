import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR 
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
DATA_DIR = '2025_07_09/4/'
IMG_DIR = os.path.join(DATA_DIR, 'recorded_images')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train_data.csv')
VAL_CSV_PATH = os.path.join(DATA_DIR, 'val_data.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test_data.csv')
BEST_MODEL_SAVE_PATH = 'best_transformer_driver_model.pth'

# 繼續訓練的開關
CONTINUE_TRAINING = False 
LOAD_MODEL_PATH = BEST_MODEL_SAVE_PATH 

# 預覽開關
SHOW_PREVIEW = True

# 模型與訓練參數
SEQUENCE_LENGTH = 20
BATCH_SIZE = 8
EPOCHS = 40 
LEARNING_RATE = 1e-4 
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 【修改點】新增差動損失的權重
BETA = 0.5 # 控制差動損失的重要性

# 圖片裁切參數
CROP_TOP_PIXELS = 280
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

# Transformer 模型參數
D_MODEL = 512
N_HEAD = 8
N_LAYERS = 3
DROPOUT = 0.5 

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
    model.train() if is_training else model.eval()
    running_loss = 0.0
    running_main_loss = 0.0
    progress_bar = tqdm(dataloader, desc=epoch_desc, leave=False)
    
    with torch.set_grad_enabled(is_training):
        for sequences, targets in progress_bar:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            
            # --- 【修改點】計算複合損失 ---
            # 1. 主要的速度損失 (MSE)
            loss_main = criterion(outputs, targets)
            
            # 2. 差動損失 (懲罰轉向錯誤)
            pred_diff = outputs[:, 0] - outputs[:, 1] # 預測的輪速差
            true_diff = targets[:, 0] - targets[:, 1] # 真實的輪速差
            loss_diff = criterion(pred_diff, true_diff)
            
            # 3. 加權總損失
            total_loss = loss_main + BETA * loss_diff

            if SHOW_PREVIEW and is_training:
                mode = "Training"
                show_prediction_preview(sequences, targets, outputs, mode=mode)
            
            if is_training:
                optimizer.zero_grad()
                total_loss.backward() # 使用總損失進行反向傳播
                optimizer.step()
                
            running_loss += total_loss.item() * sequences.size(0)
            running_main_loss += loss_main.item() * sequences.size(0)
            progress_bar.set_postfix(total_loss=total_loss.item(), speed_loss=loss_main.item())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_main_loss = running_main_loss / len(dataloader.dataset)
    return epoch_loss, epoch_main_loss

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
    if not all([os.path.exists(p) for p in [TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH]]):
        print(f"錯誤：找不到分割好的資料檔案。請先執行 '1_create_split_files.py'。")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"將使用設備: {device}")

        train_transform = transforms.Compose([
            CustomTopCrop(CROP_TOP_PIXELS),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.RandomAffine(degrees=5, translate=(0.07, 0)), # 【修改點】微調增強
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

        # --- 模型初始化與載入流程 ---
        print("\n--- 初始化模型與相關元件 ---")
        
        model = VisionTransformerDriver(D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
        
        if CONTINUE_TRAINING:
            if os.path.exists(LOAD_MODEL_PATH):
                try:
                    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
                    print(f"成功從 {LOAD_MODEL_PATH} 載入預訓練權重，將繼續訓練。")
                except Exception as e:
                    print(f"載入模型權重時發生錯誤: {e}。將從頭開始訓練。")
            else:
                print(f"找不到預訓練模型檔案 {LOAD_MODEL_PATH}。將從頭開始訓練。")
        else:
            print("設定為從頭開始訓練新模型。")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

        # 早停法變數初始化
        patience = 5 
        epochs_no_improve = 0
        best_val_loss = float('inf')

        if CONTINUE_TRAINING and os.path.exists(BEST_MODEL_SAVE_PATH):
             print("繼續訓練模式：將從目前的驗證損失開始尋找更佳模型。")

        print("\n--- 開始訓練與驗證 (已啟用差動損失) ---")
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            train_desc = f"訓練中 Epoch {epoch+1:02d}/{EPOCHS}"
            train_total_loss, train_main_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True, epoch_desc=train_desc)
            
            val_desc = f"驗證中 Epoch {epoch+1:02d}/{EPOCHS}"
            val_total_loss, val_main_loss = run_epoch(model, val_loader, criterion, None, device, is_training=False, epoch_desc=val_desc)
            
            epoch_time = time.time() - start_time
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            print(f'週期 [{epoch+1:02d}/{EPOCHS}] | 訓練損失(總): {train_total_loss:.4f} (主: {train_main_loss:.4f}) | '
                  f'驗證損失(總): {val_total_loss:.4f} (主: {val_main_loss:.4f}) | 當前學習率: {current_lr:.1e} | 耗時: {epoch_time:.2f}s')

            # 早停法邏輯判斷 (基於主要的速度損失)
            if val_main_loss < best_val_loss:
                print(f'  驗證損失從 {best_val_loss:.4f} 改善至 {val_main_loss:.4f}。儲存模型至 {BEST_MODEL_SAVE_PATH}')
                best_val_loss = val_main_loss
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

        _, test_loss = run_epoch(model, test_loader, criterion, None, device, is_training=False, epoch_desc="測試中")
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
