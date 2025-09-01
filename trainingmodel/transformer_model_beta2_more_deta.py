import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import torchvision.models as models
import math
import pandas as pd
from PIL import Image
import os
import time
from tqdm import tqdm
import numpy as np
import random
import torchvision.transforms.functional as TF

# --- 手動指定資料夾設定 ---
# 手動指定各個集合的資料夾路徑
MANUAL_FOLDER_ASSIGNMENT = True  # 設為True使用手動指定，False使用自動分割

# 訓練集資料夾（您可以在這裡指定想要用於訓練的資料夾）
TRAIN_FOLDERS = [
    '2025_08_14/2',
    '2025_08_14/3', 
    '2025_08_20/1',
    '混雜資料/2025_07_10/1',
    '混雜資料/2025_07_14/2',
    '混雜資料/2025_07_14/3',
    '混雜資料/2025_07_14/7',
    '混雜資料/2025_07_17/1',
    '混雜資料/2025_07_17/2',
    '2025_07_30/1',
    '2025_07_30/2',
    '2025_07_30/3',
    '2025_08_01/1',
    '2025_08_01/2',
    '混雜資料/2025_07_24/2',
    '混雜資料/2025_07_24/3',
    '混雜資料/2025_07_30/1',
    '混雜資料/2025_07_30/2',
]

# 驗證集資料夾（用於訓練過程中驗證模型性能）
VAL_FOLDERS = [
    '2025_08_01/3',
    '2025_08_21/1',
    '2025_08_21/2',
    '混雜資料/2025_07_24/1'
]

# 測試集資料夾（用於最終測試模型性能）
TEST_FOLDERS = [
    '2025_08_14/1',
    '2025_08_21/3',
    '混雜資料/2025_07_30/3',
    '混雜資料/2025_07_17/3',
]

# 單一資料夾模式（向後兼容）
SINGLE_DATA_DIR = '2025_08_20/1'

# 自動分割比例（僅在MANUAL_FOLDER_ASSIGNMENT=False時使用）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

BEST_MODEL_SAVE_PATH = 'beta2_at_0829.pth'

# 預覽開關
SHOW_PREVIEW = False
# 重複訓練開關
RETRAIN = True

# 優化後的模型與訓練參數
SEQUENCE_LENGTH = 9  # 減少序列長度以節省記憶體
BATCH_SIZE = 12        # 針對8G顯卡優化
EPOCHS = 80
LEARNING_RATE = 1e-4  # 提高初始學習率
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
N_LAYERS = 3
DROPOUT = 0.4

# 圖片裁切參數
CROP_TOP_PIXELS = 280
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

def load_multi_folder_data_manual(train_folders, val_folders, test_folders):
    """
    手動指定資料夾載入資料
    
    Args:
        train_folders: 訓練集資料夾清單
        val_folders: 驗證集資料夾清單  
        test_folders: 測試集資料夾清單
    
    Returns:
        dict: 包含train, val, test的字典
    """
    print("🔄 手動載入資料夾模式...")
    print(f"  📁 訓練集資料夾: {train_folders}")
    print(f"  📁 驗證集資料夾: {val_folders}")
    print(f"  📁 測試集資料夾: {test_folders}")
    
    # 設定基礎路徑（向上一層目錄）
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"🗂️ 基礎路徑: {base_path}")
    
    def load_folders(folders, set_name):
        all_data = []
        for data_dir in folders:
            # 構建完整路徑
            full_data_dir = os.path.join(base_path, data_dir)
            log_path = os.path.join(full_data_dir, 'log.csv')
            img_dir = os.path.join(full_data_dir, 'recorded_images')
            
            print(f"🔍 檢查 {set_name}集資料夾: {full_data_dir}")
            
            if not os.path.exists(log_path):
                print(f"⚠️ 跳過 {data_dir}: log.csv 不存在 (完整路徑: {log_path})")
                continue
                
            if not os.path.exists(img_dir):
                print(f"⚠️ 跳過 {data_dir}: recorded_images 資料夾不存在 (完整路徑: {img_dir})")
                continue
            
            try:
                df = pd.read_csv(log_path)
                
                # 檢查必要欄位
                required_cols = ['lwheel', 'rwheel']
                image_col = None
                
                for possible_name in ['img_path', 'image_path', 'image_name']:
                    if possible_name in df.columns:
                        image_col = possible_name
                        break
                
                if image_col is None:
                    print(f"⚠️ 跳過 {data_dir}: 找不到圖片路徑欄位")
                    continue
                    
                if not all(col in df.columns for col in required_cols):
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    print(f"⚠️ 跳過 {data_dir}: CSV缺少必要欄位 {missing_cols}")
                    continue
                
                # 標準化欄位名稱
                df = df.rename(columns={image_col: 'image_path'})
                
                # 添加資料夾資訊
                df['data_dir'] = data_dir
                df['full_image_path'] = df['image_path'].apply(lambda x: os.path.join(img_dir, os.path.basename(x)))
                
                # 檢查圖片檔案是否存在
                existing_images = df['full_image_path'].apply(os.path.exists)
                valid_df = df[existing_images].copy()
                
                if len(valid_df) == 0:
                    print(f"⚠️ 跳過 {data_dir}: 沒有有效的圖片檔案")
                    continue
                
                # 保持時間順序
                valid_df = valid_df.reset_index(drop=True)
                valid_df['folder_index'] = range(len(valid_df))
                
                all_data.append(valid_df)
                print(f"✅ {set_name}集 {data_dir}: 載入 {len(valid_df)} 筆資料")
                
            except Exception as e:
                print(f"❌ 載入 {data_dir} 時發生錯誤: {e}")
                continue
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    # 載入各個集合的資料
    train_data = load_folders(train_folders, "訓練")
    val_data = load_folders(val_folders, "驗證")
    test_data = load_folders(test_folders, "測試")
    
    total_data = len(train_data) + len(val_data) + len(test_data)
    print(f"📊 總共載入 {total_data} 筆資料")
    print(f"📋 資料分割 (手動指定): 訓練={len(train_data)} | 驗證={len(val_data)} | 測試={len(test_data)}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

# 原本的自動分割函數（保留向後兼容）
def load_multi_folder_data(data_dirs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    從多個資料夾載入並合併資料 - 修正版本，保持序列連續性
    
    Args:
        data_dirs: 資料夾路徑列表
        train_ratio: 訓練資料比例
        val_ratio: 驗證資料比例  
        test_ratio: 測試資料比例
    
    Returns:
        train_data, val_data, test_data: 合併後的資料字典
    """
    print("🔄 開始載入多資料夾資料...")
    
    all_folder_data = []  # 存儲每個資料夾的數據，保持獨立
    
    for data_dir in data_dirs:
        log_path = os.path.join(data_dir, 'log.csv')
        img_dir = os.path.join(data_dir, 'recorded_images')
        
        if not os.path.exists(log_path):
            print(f"⚠️ 跳過 {data_dir}: log.csv 不存在")
            continue
            
        if not os.path.exists(img_dir):
            print(f"⚠️ 跳過 {data_dir}: recorded_images 資料夾不存在")
            continue
        
        # 讀取CSV
        try:
            df = pd.read_csv(log_path)
            
            # 檢查必要欄位（支援多種命名方式）
            required_cols = ['lwheel', 'rwheel']
            image_col = None
            
            # 尋找圖片路徑欄位
            for possible_name in ['img_path', 'image_path', 'image_name']:
                if possible_name in df.columns:
                    image_col = possible_name
                    break
            
            if image_col is None:
                print(f"⚠️ 跳過 {data_dir}: 找不到圖片路徑欄位 (嘗試了: img_path, image_path, image_name)")
                continue
                
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                print(f"⚠️ 跳過 {data_dir}: CSV缺少必要欄位 {missing_cols}")
                continue
            
            # 標準化欄位名稱
            df = df.rename(columns={image_col: 'image_path'})
            
            # 為每筆資料添加資料夾資訊
            df['data_dir'] = data_dir
            df['full_image_path'] = df['image_path'].apply(lambda x: os.path.join(img_dir, os.path.basename(x)))
            
            # 檢查圖片檔案是否存在
            existing_images = df['full_image_path'].apply(os.path.exists)
            valid_df = df[existing_images].copy()
            
            if len(valid_df) == 0:
                print(f"⚠️ 跳過 {data_dir}: 沒有有效的圖片檔案")
                continue
            
            # 🔥 關鍵修正：保持時間順序，不要打亂單一資料夾內的資料
            valid_df = valid_df.reset_index(drop=True)
            
            # 為每個資料夾添加連續索引，確保序列連續性
            valid_df['folder_index'] = range(len(valid_df))
            
            all_folder_data.append({
                'data': valid_df,
                'folder': data_dir,
                'size': len(valid_df)
            })
            print(f"✅ {data_dir}: 載入 {len(valid_df)} 筆資料 (保持時序)")
            
        except Exception as e:
            print(f"❌ 載入 {data_dir} 時發生錯誤: {e}")
            continue
    
    if not all_folder_data:
        raise ValueError("❌ 沒有成功載入任何資料！")
    
    # 🔥 新的分割策略：按資料夾分割，而不是混合所有資料
    print("📊 採用按資料夾分割策略，保持序列完整性...")
    
    total_folders = len(all_folder_data)
    train_folder_count = max(1, int(total_folders * train_ratio))
    val_folder_count = max(1, int(total_folders * val_ratio))
    
    # 隨機分配資料夾到不同集合（但保持每個資料夾內的順序）
    random.seed(42)  # 確保可重現
    shuffled_folders = random.sample(all_folder_data, len(all_folder_data))
    
    train_folders = shuffled_folders[:train_folder_count]
    val_folders = shuffled_folders[train_folder_count:train_folder_count + val_folder_count]
    test_folders = shuffled_folders[train_folder_count + val_folder_count:]
    
    # 合併各集合的資料，保持每個資料夾內的順序
    train_data_list = [folder_info['data'] for folder_info in train_folders]
    val_data_list = [folder_info['data'] for folder_info in val_folders]
    test_data_list = [folder_info['data'] for folder_info in test_folders]
    
    # 🔥 關鍵：合併時不打亂，保持原始順序
    train_data = pd.concat(train_data_list, ignore_index=True) if train_data_list else pd.DataFrame()
    val_data = pd.concat(val_data_list, ignore_index=True) if val_data_list else pd.DataFrame()
    test_data = pd.concat(test_data_list, ignore_index=True) if test_data_list else pd.DataFrame()
    
    total_data = sum(len(folder['data']) for folder in all_folder_data)
    print(f"📊 總共載入 {total_data} 筆資料")
    print(f"📋 資料分割 (按資料夾): 訓練={len(train_data)} | 驗證={len(val_data)} | 測試={len(test_data)}")
    print(f"📁 資料夾分配:")
    print(f"  🎯 訓練集資料夾 ({len(train_folders)}個): {[f['folder'] for f in train_folders]}")
    print(f"  🎯 驗證集資料夾 ({len(val_folders)}個): {[f['folder'] for f in val_folders]}")
    print(f"  🎯 測試集資料夾 ({len(test_folders)}個): {[f['folder'] for f in test_folders]}")
    
    return {
        'train': train_data,
        'val': val_data, 
        'test': test_data
    }

def create_single_folder_data(data_dir):
    """向後兼容的單一資料夾載入"""
    train_csv = os.path.join(data_dir, 'train_data.csv')
    val_csv = os.path.join(data_dir, 'val_data.csv')
    test_csv = os.path.join(data_dir, 'test_data.csv')
    
    if all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        return {
            'train_csv': train_csv,
            'val_csv': val_csv,
            'test_csv': test_csv,
            'img_dir': os.path.join(data_dir, 'recorded_images')
        }
    else:
        # 如果沒有分割好的CSV，使用log.csv並自動分割
        log_path = os.path.join(data_dir, 'log.csv')
        if os.path.exists(log_path):
            return load_multi_folder_data([data_dir])
        else:
            raise FileNotFoundError(f"找不到 {data_dir} 的資料檔案")

def validate_sequence_integrity(data_dict, sequence_length=15):
    """
    驗證資料集的序列完整性
    
    Args:
        data_dict: 包含train/val/test的資料字典
        sequence_length: 序列長度
    
    Returns:
        validation_report: 驗證報告
    """
    print("\n🔍 驗證序列完整性...")
    
    report = {}
    
    for split_name, data in data_dict.items():
        print(f"\n📊 檢查 {split_name} 集...")
        
        if len(data) == 0:
            print(f"  ⚠️ {split_name} 集為空")
            continue
        
        # 檢查資料夾分布
        if 'data_dir' in data.columns:
            folder_counts = data['data_dir'].value_counts()
            print(f"  📁 資料夾分布: {dict(folder_counts)}")
            
            # 檢查跨資料夾的序列問題
            cross_folder_issues = 0
            total_sequences = len(data) - sequence_length + 1
            
            for i in range(total_sequences):
                start_folder = data.iloc[i]['data_dir']
                end_folder = data.iloc[i + sequence_length - 1]['data_dir']
                
                if start_folder != end_folder:
                    cross_folder_issues += 1
            
            cross_folder_ratio = cross_folder_issues / total_sequences if total_sequences > 0 else 0
            
            print(f"  🎯 可用序列數: {total_sequences}")
            print(f"  ⚠️ 跨資料夾序列: {cross_folder_issues} ({cross_folder_ratio:.2%})")
            
            if cross_folder_ratio > 0.1:  # 如果超過10%的序列跨資料夾
                print(f"  🚨 警告: 跨資料夾序列比例過高 ({cross_folder_ratio:.2%})")
            else:
                print(f"  ✅ 序列完整性良好 ({cross_folder_ratio:.2%} 跨資料夾)")
            
            report[split_name] = {
                'total_data': len(data),
                'total_sequences': total_sequences,
                'cross_folder_sequences': cross_folder_issues,
                'cross_folder_ratio': cross_folder_ratio,
                'folders': list(folder_counts.keys())
            }
        else:
            print(f"  ⚠️ 沒有資料夾資訊，無法驗證跨資料夾問題")
            report[split_name] = {
                'total_data': len(data),
                'total_sequences': len(data) - sequence_length + 1,
                'cross_folder_sequences': 0,
                'cross_folder_ratio': 0,
                'folders': ['unknown']
            }
    
    return report

# --- 1. 增強的資料集與轉換 ---
class CustomTopCrop:
    def __init__(self, top_pixels):
        self.top_pixels = top_pixels
    def __call__(self, img):
        return TF.crop(img, self.top_pixels, 0, ORIGINAL_HEIGHT - self.top_pixels, ORIGINAL_WIDTH)

class DrivingDataset(Dataset):
    def __init__(self, data_source, sequence_length, transform=None, is_training=False):
        """
        改進的DrivingDataset，支援多種資料來源
        
        Args:
            data_source: 可以是CSV檔案路徑或DataFrame
            sequence_length: 序列長度
            transform: 圖片轉換
            is_training: 是否為訓練模式
        """
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_training = is_training
        
        # 支援多種資料來源
        if isinstance(data_source, str):
            # 傳統方式：CSV檔案路徑
            self.annotations = pd.read_csv(data_source)
            self.img_dir = os.path.dirname(data_source).replace('log.csv', 'recorded_images')
        elif isinstance(data_source, pd.DataFrame):
            # 新方式：直接傳入DataFrame
            self.annotations = data_source.copy()
            self.img_dir = None  # 圖片路徑已經在DataFrame中
        else:
            raise ValueError("data_source 必須是CSV檔案路徑或pandas DataFrame")

    def __len__(self):
        return len(self.annotations) - self.sequence_length + 1

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.sequence_length
        
        # 🔥 安全檢查：確保序列來自同一資料夾，避免跨資料夾序列
        if 'data_dir' in self.annotations.columns and end_index <= len(self.annotations):
            start_dir = self.annotations.iloc[start_index]['data_dir']
            end_dir = self.annotations.iloc[end_index - 1]['data_dir']
            
            if start_dir != end_dir:
                # 如果跨越不同資料夾，尋找安全的起始點
                #print(f"⚠️ 序列跨越資料夾邊界 index={index}, 從 {start_dir} 到 {end_dir}")
                
                # 尋找當前資料夾內足夠的連續資料
                current_folder_mask = self.annotations['data_dir'] == start_dir
                current_folder_indices = self.annotations[current_folder_mask].index.tolist()
                
                # 檢查當前資料夾是否有足夠的連續資料
                available_in_folder = sum(1 for i in current_folder_indices if i >= start_index)
                
                if available_in_folder < self.sequence_length:
                    pass
                    # 如果當前資料夾剩餘資料不足，使用重複最後一張圖片的策略
                    # print(f"⚠️ 資料夾 {start_dir} 剩餘資料不足15張，使用填充策略")
                    # 可以選擇跳過這個index或使用填充，這裡選擇填充最後一張

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
        last_valid_image = None  # 用於填充策略

        for i in range(start_index, end_index):
            # 邊界檢查
            if i >= len(self.annotations):
                if last_valid_image is not None:
                    print(f"⚠️ 使用最後有效圖片填充 index={i}")
                    sequence_images.append(last_valid_image)
                    continue
                else:
                    print(f"⚠️ 無法獲取圖片 index={i}，使用黑色圖片")
                    if self.transform:
                        black_image = Image.new('RGB', (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), (0, 0, 0))
                        image = self.transform(black_image)
                    else:
                        image = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH)
                    sequence_images.append(image)
                    continue

            row = self.annotations.iloc[i]

            # 根據資料來源決定圖片路徑
            if 'full_image_path' in row:
                # 新方式：使用full_image_path
                img_path = row['full_image_path']
            elif 'img_path' in row:
                # 兼容舊格式
                img_name = row['img_path']
                img_path = os.path.join(self.img_dir, img_name)
            else:
                # 使用image_path
                img_name = row['image_path']
                if self.img_dir:
                    img_path = os.path.join(self.img_dir, img_name)
                else:
                    img_path = img_name

            try:
                image = Image.open(img_path).convert('RGB')

                if apply_straight_aug:
                    image = image.transform(image.size, Image.AFFINE, (1, 0, -shift_px, 0, 1, 0))

                if self.transform:
                    image = self.transform(image)

                sequence_images.append(image)
                last_valid_image = image  # 保存最後有效圖片

            except Exception as e:
                print(f"⚠️ 載入圖片失敗 {img_path}: {e}")
                if last_valid_image is not None:
                    # 使用最後有效圖片
                    sequence_images.append(last_valid_image)
                else:
                    # 使用黑色圖片替代
                    if self.transform:
                        black_image = Image.new('RGB', (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), (0, 0, 0))
                        image = self.transform(black_image)
                    else:
                        image = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH)
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
    
    # 根據設定選擇資料載入方式
    if MANUAL_FOLDER_ASSIGNMENT:
        print("🔄 使用手動指定資料夾模式...")
        try:
            data_dict = load_multi_folder_data_manual(TRAIN_FOLDERS, VAL_FOLDERS, TEST_FOLDERS)
            train_data = data_dict['train']
            val_data = data_dict['val']
            test_data = data_dict['test']
            
            # 驗證序列完整性
            integrity_report = validate_sequence_integrity(data_dict, SEQUENCE_LENGTH)
            
        except Exception as e:
            print(f"❌ 手動載入資料失敗: {e}")
            print("改為使用單一資料夾模式...")
            data_paths = create_single_folder_data(SINGLE_DATA_DIR)
            train_data = pd.read_csv(data_paths['train_csv'])
            val_data = pd.read_csv(data_paths['val_csv'])
            test_data = pd.read_csv(data_paths['test_csv'])
            
            train_data['img_dir'] = data_paths['img_dir']
            val_data['img_dir'] = data_paths['img_dir']
            test_data['img_dir'] = data_paths['img_dir']
    
    # 設定使用DataFrame模式
    use_dataframe_mode = True

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
    if use_dataframe_mode:
        # 使用DataFrame模式（新的多資料夾系統）
        train_dataset = DrivingDataset(train_data, SEQUENCE_LENGTH, 
                                     transform=train_transform, is_training=True)
        val_dataset = DrivingDataset(val_data, SEQUENCE_LENGTH, 
                                   transform=val_test_transform, is_training=False)
        test_dataset = DrivingDataset(test_data, SEQUENCE_LENGTH, 
                                    transform=val_test_transform, is_training=False)

    # 優化的資料載入器
    num_workers = 8  # 針對8G顯卡減少工作程序
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    print(f"資料載入完成 -> 訓練集: {len(train_dataset)} | 驗證集: {len(val_dataset)} | 測試集: {len(test_dataset)}")
    
    if MANUAL_FOLDER_ASSIGNMENT:
        print("📊 手動指定資料夾統計:")
        print(f"  📁 訓練集資料夾: {TRAIN_FOLDERS}")
        print(f"  📁 驗證集資料夾: {VAL_FOLDERS}")
        print(f"  📁 測試集資料夾: {TEST_FOLDERS}")
        
    if torch.cuda.is_available():
        print(f"GPU顯存總量: {torch.cuda.get_device_properties(device).total_memory/1024/1024:.1f} MB")

    print("🔧 開始創建模型...")
    try:
        # 建立模型
        model = VisionTransformerDriver(D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
        print("✅ 模型創建成功")
        
        if os.path.exists(BEST_MODEL_SAVE_PATH) and RETRAIN:
            print("使用之前的模型開始訓練")
            model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=device))
        elif os.path.exists(BEST_MODEL_SAVE_PATH):
            print(f"⚠️ 找到現有模型 {BEST_MODEL_SAVE_PATH}，但 RETRAIN=False，將從頭開始訓練")
        else:
            print("🆕 開始全新訓練")

    except Exception as e:
        print(f"❌ 模型創建失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # 優化器和排程器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5, 
                            betas=(0.9, 0.95), eps=1e-8)

    # 使用 ReduceLROnPlateau 替代 OneCycleLR
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',           # 監控驗證損失下降
        factor=0.5,           # 學習率減半
        patience=5,           # 5個epoch沒改善就降低學習率
        min_lr=1e-7,         # 最小學習率
        cooldown=2,          # 降低學習率後等待2個epoch再次檢查
        threshold=0.001,     # 改善的最小閾值
        threshold_mode='rel' # 相對改善（0.1%）
    )

    # early stopping 參數調整
    patience = 30
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_val_mae = float('inf')

    print("\n--- 開始訓練與驗證 (已啟用早停法與ReduceLROnPlateau學習率排程) ---")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # 訓練階段
        train_desc = f"訓練中 Epoch {epoch+1:02d}/{EPOCHS}"
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, 
                                        device, is_training=True, epoch_desc=train_desc)

        # 驗證階段
        val_desc = f"驗證中 Epoch {epoch+1:02d}/{EPOCHS}"
        val_loss, val_mae = run_epoch(model, val_loader, criterion, None, 
                                    device, is_training=False, epoch_desc=val_desc)

        # ReduceLROnPlateau 需要在驗證後調用
        scheduler.step(val_loss)

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