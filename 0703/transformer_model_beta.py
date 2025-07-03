import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
MODEL_SAVE_PATH = 'transformer_driver_model_augmented.pth'

# 預覽開關
SHOW_PREVIEW = True

# 模型與訓練參數
SEQUENCE_LENGTH = 20
BATCH_SIZE = 8 # 增加批次大小以更好地利用GPU
EPOCHS = 20
LEARNING_RATE = 1e-5
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 圖片裁切參數
CROP_TOP_PIXELS = 260 # 調整為較合理的裁切值
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

# Transformer 模型參數
D_MODEL = 512
N_HEAD = 8
N_LAYERS = 3
DROPOUT = 0.1

# --- 【修改點】新增一個可序列化的自定義裁切類別 ---
class CustomTopCrop:
    """
    一個可序列化的類別，用於裁切圖片頂部。
    取代了在多程序中會導致錯誤的 lambda 函式。
    """
    def __init__(self, top_pixels):
        self.top_pixels = top_pixels

    def __call__(self, img):
        # TF.crop(img, top, left, height, width)
        return TF.crop(img, self.top_pixels, 0, ORIGINAL_HEIGHT - self.top_pixels, ORIGINAL_WIDTH)

# --- 1. 自定義資料集 (核心修改：加入針對性資料增強) ---
class DrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir, sequence_length, transform=None, is_training=False):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_training = is_training # 標記是否為訓練集，只對訓練集做增強

    def __len__(self):
        return len(self.annotations) - self.sequence_length + 1

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.sequence_length
        
        # --- 針對性資料增強邏輯 ---
        # 1. 先讀取目標速度，以判斷是否為直線
        target_row = self.annotations.iloc[end_index - 1]
        l_speed = target_row['lwheel']
        r_speed = target_row['rwheel']
        targets = torch.tensor([l_speed, r_speed], dtype=torch.float32)

        apply_straight_aug = False
        shift_px = 0
        
        # 只對訓練集進行增強
        if self.is_training:
            # 判斷是否為直線行駛
            is_straight = abs(l_speed - r_speed) < 1.5 # 放寬直線判斷標準
            
            # 50% 的機率對直線資料進行增強
            if is_straight and random.random() < 0.5:
                apply_straight_aug = True
                # 隨機決定平移方向和幅度
                shift_direction = random.choice([-1, 1]) # -1: 向左平移, 1: 向右平移
                shift_px = random.randint(20, 50) * shift_direction
                
                # 根據平移方向，創造一個修正後的輪速標籤
                correction_strength = 2.5 # 一個微小的轉向修正值
                if shift_px < 0: # 影像向左平移 (模擬車輛偏左)，需要向右轉修正
                    targets = torch.tensor([l_speed + correction_strength, r_speed - correction_strength], dtype=torch.float32)
                else: # 影像向右平移 (模擬車輛偏右)，需要向左轉修正
                    targets = torch.tensor([l_speed - correction_strength, r_speed + correction_strength], dtype=torch.float32)

        # 2. 讀取並處理影像序列
        sequence_images = []
        for i in range(start_index, end_index):
            img_name = self.annotations.iloc[i]['img_path']
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            
            # 如果觸發了直線增強，對序列中的每張圖應用相同的平移
            if apply_straight_aug:
                image = image.transform(image.size, Image.AFFINE, (1, 0, -shift_px, 0, 1, 0))

            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)

        images_tensor = torch.stack(sequence_images)
        return images_tensor, targets

# --- 2. Transformer 模型架構 ---
class PositionalEncoding(nn.Module):
    # ... (此部分程式碼不變) ...
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
    # ... (此部分程式碼不變) ...
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

# --- 預覽與訓練函式 (與之前版本相同) ---
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
            if SHOW_PREVIEW:
                mode = "Training" if is_training else "Validation"
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

        # 【修改點】使用自定義的 CustomTopCrop 類別取代 lambda
        train_transform = transforms.Compose([
            CustomTopCrop(CROP_TOP_PIXELS),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # 通用增強
            transforms.RandomAffine(degrees=5, translate=(0.05, 0)), # 通用增強
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
        
        # 初始化資料集時傳入不同的轉換流程和 is_training 標記
        train_dataset = DrivingDataset(TRAIN_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform=train_transform, is_training=True)
        val_dataset = DrivingDataset(VAL_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform=val_test_transform, is_training=False)
        test_dataset = DrivingDataset(TEST_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform=val_test_transform, is_training=False)

        # 在 DataLoader 中加入 num_workers 和 pin_memory
        num_workers = 4 # 建議設為CPU核心數的一半或全部，可自行調整
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

        print(f"資料載入完成 -> 訓練集: {len(train_dataset)} | 驗證集: {len(val_dataset)} | 測試集: {len(test_dataset)}")
        print(f"使用 {num_workers} 個子程序進行資料載入。")

        model = VisionTransformerDriver(D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("\n--- 開始訓練與驗證 Transformer 模型 ---")
        for epoch in range(EPOCHS):
            start_time = time.time()
            train_desc = f"訓練中 Epoch {epoch+1:02d}/{EPOCHS}"
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True, epoch_desc=train_desc)
            val_desc = f"驗證中 Epoch {epoch+1:02d}/{EPOCHS}"
            val_loss = run_epoch(model, val_loader, criterion, None, device, is_training=False, epoch_desc=val_desc)
            epoch_time = time.time() - start_time
            print(f'週期 [{epoch+1:02d}/{EPOCHS}] | 訓練損失: {train_loss:.4f} | 驗證損失: {val_loss:.4f} | 耗時: {epoch_time:.2f}s')

        print("--- 訓練完成 ---")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"模型已儲存至 {MODEL_SAVE_PATH}")
        
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

        print("\n--- 在測試集上評估最終模型 ---")
        original_show_preview_state = SHOW_PREVIEW
        SHOW_PREVIEW = False
        test_loss = run_epoch(model, test_loader, criterion, None, device, is_training=False, epoch_desc="測試中")
        SHOW_PREVIEW = original_show_preview_state
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
