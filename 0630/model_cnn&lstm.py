import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as TF

import pandas as pd
from PIL import Image
import numpy as np
import os
import time
from tqdm import tqdm

# --- 參數設定 ---
# 資料路徑
DATA_DIR = '2025_06_30/1/'
TRAIN_CSV = os.path.join(DATA_DIR, 'train', 'log.csv')
TRAIN_IMG = os.path.join(DATA_DIR, 'train', 'images')
VAL_CSV = os.path.join(DATA_DIR, 'val', 'log.csv')
VAL_IMG = os.path.join(DATA_DIR, 'val', 'images')
TEST_CSV = os.path.join(DATA_DIR, 'test', 'log.csv')
TEST_IMG = os.path.join(DATA_DIR, 'test', 'images')
MODEL_SAVE_PATH = 'cnn_lstm_driver_model.pth'


# 模型與訓練參數
SEQUENCE_LENGTH = 10  # 使用過去5張圖片來預測當前速度
BATCH_SIZE = 4 # 批次大小
EPOCHS = 10  # 訓練的輪數
LEARNING_RATE = 0.001 # 學習率
# ResNet需要至少224x224的輸入，且建議使用其標準化的均值和標準差
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 圖片裁切參數 (裁掉圖片頂部包含文字的部分)
# 您的圖片尺寸為 480x640，文字大約在前100-120像素，我們裁切120像素
CROP_TOP_PIXELS = 120
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

# --- 2. 自定義資料集 ---
class DrivingDataset(Dataset):
    """
    自定義資料集，用於讀取影像序列和對應的速度標籤。
    """
    def __init__(self, csv_file, img_dir, sequence_length, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        # 因為需要往前看 sequence_length-1 筆資料，所以總長度要減掉
        return len(self.annotations) - self.sequence_length + 1

    def __getitem__(self, index):
        # 建立影像序列
        # 例如，若 index=0, sequence_length=5, 我們會取 0,1,2,3,4 的影像
        # 目標是第 4 筆(最後一筆)影像的速度
        start_index = index
        end_index = index + self.sequence_length
        
        sequence_images = []
        for i in range(start_index, end_index):
            img_name = self.annotations.iloc[i, 0]
            img_path = os.path.join(self.img_dir, img_name)
            # 確保以RGB格式讀取
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)

        # 將影像序列堆疊成一個 tensor
        images_tensor = torch.stack(sequence_images)

        # 目標速度是序列中最後一張影像對應的速度
        target_row = self.annotations.iloc[end_index - 1]
        l_speed = target_row['lwheel']
        r_speed = target_row['rwheel']
        targets = torch.tensor([l_speed, r_speed], dtype=torch.float32)

        return images_tensor, targets

# --- 3. CNN-LSTM 模型架構 ---
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        # CNN 特徵提取器 (使用預訓練的 ResNet-18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 去掉 ResNet 最後的全連接層，我們只需要特徵
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        
        # # 為了加速訓練和利用遷移學習，凍結 CNN 層的權重
        # for param in self.cnn.parameters():
        #     param.requires_grad = False
        
        # LSTM 時序分析器
        # ResNet-18 輸出的特徵維度為 512
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        
        # 全連接層，輸出最終預測
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x 的形狀: (batch_size, sequence_length, C, H, W)
        batch_size, seq_len, c, h, w = x.shape
        
        # 將 batch 和 sequence 維度合併，以符合 CNN 輸入
        x = x.view(batch_size * seq_len, c, h, w)
        
        # 通過 CNN 提取特徵
        # 如果凍結了CNN層，這裡使用 no_grad 可以節省計算資源
        # with torch.no_grad():
        features = self.cnn(x)
        
        # 將特徵維度還原成序列
        # features shape: (batch_size * seq_len, 512, 1, 1) -> (batch_size, seq_len, 512)
        features = features.view(batch_size, seq_len, -1)
        
        # 通過 LSTM
        # lstm_out 的形狀: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(features)
        
        # 我們只需要最後一個時間點的輸出進行預測
        last_time_step_out = lstm_out[:, -1, :]
        
        # 通過全連接層得到最終輸出
        out = self.fc(last_time_step_out)
        
        return out

# --- 4. 訓練函式 ---
def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    print("\n--- 開始訓練模型 ---")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train() # 設置為訓練模式
        running_loss = 0.0
        start_time = time.time()
        
        for sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # 前向傳播
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # 反向傳播與優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        
        epoch_loss = running_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f'--- Epoch [{epoch+1}/{num_epochs}] 完成 --- 平均損失: {epoch_loss:.4f}, 耗時: {epoch_time:.2f}s')

    print("--- 訓練完成 ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已儲存至 {MODEL_SAVE_PATH}")


# --- 5. 預測函式 ---
def predict(model, image_sequence, transform, device):
    model.to(device)
    model.eval() # 設置為評估模式
    
    # 預處理影像序列
    processed_sequence = [transform(img.convert('RGB')) for img in image_sequence]
    # 將單一樣本的序列堆疊，並增加一個 batch 維度
    input_tensor = torch.stack(processed_sequence).unsqueeze(0) 
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        
    return prediction.cpu().numpy().flatten()


# --- 主程式 ---
if __name__ == '__main__':
    # # 檢查並生成模擬資料
    # if not os.path.exists(CSV_PATH):
    #     generate_dummy_data(num_samples=100)

    # 設定設備 (GPU優先)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"將使用設備: {device}")

    # 定義影像轉換
    # 尺寸需符合 ResNet 輸入要求，並進行標準化
    # 【重要】在轉換流程中加入裁切與資料增強步驟
    transform = transforms.Compose([
        # # 步驟 1: 裁切圖片頂部以移除文字
        # transforms.Lambda(lambda img: TF.crop(img, CROP_TOP_PIXELS, 0, ORIGINAL_HEIGHT - CROP_TOP_PIXELS, ORIGINAL_WIDTH)),
        
        # # 【修改點】步驟 2: 對裁切後的圖片進行資料增強
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        # transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        
        # 步驟 3: 調整大小以符合模型輸入
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        
        # 步驟 4: 轉換為 Tensor
        transforms.ToTensor(),
        
        # 步驟 5: 標準化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 建立資料集和資料載入器
    train_dataset = DrivingDataset(TRAIN_CSV, TRAIN_IMG, SEQUENCE_LENGTH, transform)
    val_dataset = DrivingDataset(VAL_CSV, VAL_IMG, SEQUENCE_LENGTH, transform)
    test_dataset = DrivingDataset(TEST_CSV, TEST_IMG, SEQUENCE_LENGTH, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型、損失函數和優化器
    model = CNNLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 訓練與驗證
    print("\n--- 開始訓練模型 ---")
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # 驗證
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] 訓練損失: {train_loss:.4f}，驗證損失: {val_loss:.4f}")

        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"模型已儲存至 {MODEL_SAVE_PATH} (驗證損失最佳)")

    print("--- 訓練完成 ---")

    # 測試集評估
    print("\n--- 測試集評估 ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")