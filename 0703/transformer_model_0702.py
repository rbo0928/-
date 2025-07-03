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
import torchvision.transforms.functional as TF
import cv2 # 【修改點】匯入 OpenCV
import numpy as np # 【修改點】匯入 NumPy
from tqdm import tqdm # 【修改點】匯入tqdm

# --- 參數設定 ---
# 資料路徑 (讀取分割好的檔案)
DATA_DIR = '2025_07_02/2/'
IMG_DIR = os.path.join(DATA_DIR, 'recorded_images')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train_data.csv')
VAL_CSV_PATH = os.path.join(DATA_DIR, 'val_data.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test_data.csv')
MODEL_SAVE_PATH = 'transformer_driver_model.pth'

# 新增預覽開關
SHOW_PREVIEW = True # 設為 True 以在訓練和驗證時顯示預覽視窗

# 模型與訓練參數
SEQUENCE_LENGTH = 20
BATCH_SIZE = 4
EPOCHS = 20 # Transformer模型可能需要更多時間收斂
LEARNING_RATE = 1e-5 # Transformer對學習率較敏感，建議使用較小的值
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 圖片裁切參數 (裁掉圖片頂部包含文字的部分)
# 您的圖片尺寸為 480x640，文字大約在前100-120像素，我們裁切120像素
CROP_TOP_PIXELS = 280
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

# Transformer 模型參數
# ResNet-18 輸出的特徵維度為 512
D_MODEL = 512  # 特徵維度 (必須與CNN輸出一致)
N_HEAD = 8     # 多頭注意力機制的頭數 (512需能被8整除)
N_LAYERS = 3   # Transformer Encoder的層數
DROPOUT = 0.1  # Dropout比例

# --- 1. 自定義資料集 ---
class DrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir, sequence_length, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        # 因為每個索引都需要往前看 sequence_length-1 筆，所以要確保不會超出範圍
        return len(self.annotations) - self.sequence_length + 1

    def __getitem__(self, index):
        # 我們從 `index` 開始，取到 `index + sequence_length`
        start_index = index
        end_index = index + self.sequence_length
        
        sequence_images = []
        for i in range(start_index, end_index):
            img_name = self.annotations.iloc[i]['img_path']
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)

        images_tensor = torch.stack(sequence_images)
        target_row = self.annotations.iloc[end_index - 1]
        targets = torch.tensor([target_row['lwheel'], target_row['rwheel']], dtype=torch.float32)
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
        # CNN特徵提取器 (換回 ResNet-18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 特殊的 [CLS] token，用於匯總序列資訊
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 輸出頭
        self.output_fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        
        # 1. 提取影像特徵
        features = self.cnn(x).view(batch_size, seq_len, self.d_model) # [Batch, Seq, Dim]
        
        # 2. 加入 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, features), dim=1) # [Batch, Seq+1, Dim]
        
        # 3. 加入位置編碼
        x = x.transpose(0, 1) # [Seq+1, Batch, Dim]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # [Batch, Seq+1, Dim]
        
        # 4. 通過 Transformer Encoder
        transformer_output = self.transformer_encoder(x) # [Batch, Seq+1, Dim]
        
        # 5. 取出 [CLS] token對應的輸出進行預測
        cls_output = transformer_output[:, 0, :]
        
        # 6. 通過最後的分類頭
        out = self.output_fc(cls_output)
        return out

# --- 【修改點】更新預覽函式以處理整個批次 ---
def show_prediction_preview(sequences_batch, targets_batch, outputs_batch, mode="Validation"):
    """建立並顯示一個包含整個批次預測結果的預覽視窗"""
    batch_previews = []
    batch_size = sequences_batch.size(0)

    # 反標準化轉換
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for i in range(batch_size):
        # 提取單一樣本的資料
        image_tensor = sequences_batch[i, -1, :, :, :] # 取序列中最後一張圖
        true_speeds = targets_batch[i].cpu().numpy()
        pred_speeds = outputs_batch[i].cpu().detach().numpy()

        # 將Tensor轉為可顯示的NumPy影像
        img_display = inv_normalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        img_display = (img_display * 255).astype(np.uint8)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

        # 建立文字畫布
        text_canvas = np.zeros((IMG_HEIGHT, 400, 3), dtype=np.uint8)
        
        # 顯示模式
        mode_color = (0, 255, 255) if mode == "Training" else (255, 255, 0)
        cv2.putText(text_canvas, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # 顯示真實速度
        cv2.putText(text_canvas, "Ground Truth", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(text_canvas, f"L: {true_speeds[0]:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(text_canvas, f"R: {true_speeds[1]:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # 顯示預測速度
        cv2.putText(text_canvas, "Prediction", (210, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(text_canvas, f"L: {pred_speeds[0]:.2f}", (210, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(text_canvas, f"R: {pred_speeds[1]:.2f}", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # 合併單一預覽行
        preview_row = np.hstack((img_display, text_canvas))
        batch_previews.append(preview_row)

    # 將所有預覽行垂直堆疊
    if batch_previews:
        final_preview = np.vstack(batch_previews)
        cv2.imshow("即時訓練預覽 (Live Training Preview)", final_preview)
        cv2.waitKey(1)


# --- 3. 訓練、驗證、測試函式 ---
def run_epoch(model, dataloader, criterion, optimizer, device, is_training, epoch_desc=""):
    model.train() if is_training else model.eval()
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=epoch_desc, leave=False)
    
    with torch.set_grad_enabled(is_training):
        for sequences, targets in progress_bar:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # 【修改點】在訓練和驗證時都顯示預覽
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

# --- 4. 預測函式 ---
def predict(model, image_sequence, transform, device):
    """對單一影像序列進行預測"""
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

        transform = transforms.Compose([
            # 步驟 1: 裁切圖片頂部以移除文字
            transforms.Lambda(lambda img: TF.crop(img, CROP_TOP_PIXELS, 0, ORIGINAL_HEIGHT - CROP_TOP_PIXELS, ORIGINAL_WIDTH)),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 建立三個資料集
        train_dataset = DrivingDataset(TRAIN_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform)
        val_dataset = DrivingDataset(VAL_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform)
        test_dataset = DrivingDataset(TEST_CSV_PATH, IMG_DIR, SEQUENCE_LENGTH, transform)

        # 建立三個資料載入器
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"資料載入完成 -> 訓練集: {len(train_dataset)} | 驗證集: {len(val_dataset)} | 測試集: {len(test_dataset)}")

        # 初始化模型
        model = VisionTransformerDriver(D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 訓練與驗證迴圈
        print("\n--- 開始訓練與驗證 Transformer 模型 ---")
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            # 【修改點】傳入描述文字
            train_desc = f"訓練中 Epoch {epoch+1:02d}/{EPOCHS}"
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True, epoch_desc=train_desc)
            
            val_desc = f"驗證中 Epoch {epoch+1:02d}/{EPOCHS}"
            val_loss = run_epoch(model, val_loader, criterion, None, device, is_training=False, epoch_desc=val_desc)
            
            epoch_time = time.time() - start_time
            print(f'週期 [{epoch+1:02d}/{EPOCHS}] | 訓練損失: {train_loss:.4f} | '
                  f'驗證損失: {val_loss:.4f} | 耗時: {epoch_time:.2f}s')

        print("--- 訓練完成 ---")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"模型已儲存至 {MODEL_SAVE_PATH}")
        
        # 在測試集上進行最終評估
        print("\n--- 在測試集上評估最終模型 ---")
        test_loss = run_epoch(model, test_loader, criterion, None, device, is_training=False, epoch_desc="測試中")
        print(f"--- 最終測試損失: {test_loss:.4f} ---")
        
        # 執行預測範例
        print("\n--- 在測試集上執行預測範例 ---")
        # 確保測試集有足夠的資料可以形成一個序列
        if len(test_dataset) > 0:
            sample_sequence_tensor, true_speeds_tensor = test_dataset[0]
            
            # 將Tensor轉回PIL Image以符合預測函式輸入
            sample_pil_images = []
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            for img_tensor in sample_sequence_tensor:
                inv_tensor = inv_normalize(img_tensor.cpu())
                pil_img = transforms.ToPILImage()(inv_tensor)
                sample_pil_images.append(pil_img)

            predicted_speeds = predict(model, sample_pil_images, transform, device)

            print(f"預測速度: 左={predicted_speeds[0]:.2f}, 右={predicted_speeds[1]:.2f}")
            print(f"真實速度: 左={true_speeds_tensor[0]:.2f}, 右={true_speeds_tensor[1]:.2f}")
        else:
            print("測試集為空，無法執行預測。")
