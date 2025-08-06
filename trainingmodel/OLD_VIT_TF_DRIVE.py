from collections import deque
import pybullet as p
from pybullet_utils import gazebo_world_parser
import pybullet_data
import cv2
import time
import random
import numpy as np
import datetime, os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# --- AI 模型整合部分 ---
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

class CustomTopCrop:
    def __init__(self, top_pixels):
        self.top_pixels = top_pixels
    def __call__(self, img):
        return TF.crop(img, self.top_pixels, 0, ORIGINAL_HEIGHT - self.top_pixels, ORIGINAL_WIDTH)

# --- 1. AI 模型參數 (必須與訓練時完全一致) ---
MODEL_PATH = 'best_model.pth' # 指定訓練好的模型檔案
# 模型參數 (針對 8GB VRAM 的建議 - 提升版)
SEQUENCE_LENGTH = 20
IMG_H, IMG_W = 200, 640
EMBED_DIM = 512      # 提升嵌入維度
DEPTH = 6            # 影像編碼器深度 (維持不變)
NUM_HEADS = 8        # 提升影像編碼器注意力頭 (需為 EMBED_DIM 的因數)
TEMPORAL_DEPTH = 4   # 時序 Transformer 深度 (維持不變)
TEMPORAL_HEADS = 8   # 提升時序 Transformer 注意力頭 (需為 EMBED_DIM 的因數)

# 圖片裁切參數 (裁掉圖片頂部包含文字的部分)
CROP_TOP_PIXELS = 280
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

data_log = []
SAVE_IMG = True

actual_lwheel_value = 0
actual_rwheel_value = 0
alpha = 0.3  # 越小回復越慢



# --- 2. AI 模型架構定義 (必須與訓練時的定義相同) ---
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


# --- 3. AI 模型載入與預測函式 ---
def load_model(model_path, device):
    """載入訓練好的模型"""
    print("正在載入 AI 模型...")
    model = MultimodalTemporalTransformer(
        seq_len=SEQUENCE_LENGTH, img_size=(IMG_H, IMG_W), embed_dim=EMBED_DIM,
        depth=DEPTH, num_heads=NUM_HEADS,
        temporal_depth=TEMPORAL_DEPTH, temporal_heads=TEMPORAL_HEADS
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("AI 模型已成功載入。")
        return model
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 '{model_path}'。自動駕駛模式將無法使用。")
        return None
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        return None

# 【*** 修正 1 ***】: 修改函式簽名，並處理速度序列
def predict_speeds(model, image_sequence, speed_sequence, transform, device):
    """使用模型預測速度"""
    if model is None or len(image_sequence) < SEQUENCE_LENGTH or len(speed_sequence) < SEQUENCE_LENGTH:
        return 0, 0 # 如果模型或序列有問題，返回安全值

    # 預處理影像序列
    processed_images = [transform(img) for img in image_sequence]
    image_tensor = torch.stack(processed_images).unsqueeze(0).to(device) # Shape: (1, T, C, H, W)

    # 預處理速度序列
    speed_tensor = torch.tensor(speed_sequence, dtype=torch.float32).unsqueeze(0).to(device) # Shape: (1, T)

    with torch.no_grad():
        # 將影像和速度一同傳入模型
        prediction = model(image_tensor, speed_tensor)

    predicted_speeds = prediction.cpu().numpy().flatten()
    return predicted_speeds[0], predicted_speeds[1] # lwheel, rwheel

# ---------------------------
# Lane offset (via OpenCV)
# ---------------------------
def get_lane_offset_by_opencv(img, width):
    # Step 1: 提取白色區域（避免抓到柏油）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    masked = cv2.bitwise_and(img, img, mask=white_mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Canny 邊緣
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Step 3: ROI 區域（畫面底部）
    roi_margin = 150
    roi = edges[img.shape[0] - roi_margin:, :]

    # Step 4: 分左右區域
    left_roi = roi[:, :width//2]
    right_roi = roi[:, width//2:]

    # Step 5: 各自找線段
    left_lines = cv2.HoughLinesP(left_roi, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=20)
    right_lines = cv2.HoughLinesP(right_roi, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=20)

    left_xs = []
    if left_lines is not None:
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            x_mid = (x1 + x2) / 2
            left_xs.append(x_mid)
            cv2.line(img, (x1, y1 + img.shape[0] - roi_margin), (x2, y2 + img.shape[0] - roi_margin), (0, 255, 0), 2)

    right_xs = []
    if right_lines is not None:
        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            x_mid = (x1 + x2) / 2 + width//2  # 因為是右半邊，要加偏移
            right_xs.append(x_mid)
            cv2.line(img, (x1 + width//2, y1 + img.shape[0] - roi_margin),
                     (x2 + width//2, y2 + img.shape[0] - roi_margin), (255, 0, 0), 2)

    # Step 6: 計算車道中心
    if left_xs and right_xs:
        lane_center = (np.mean(left_xs) + np.mean(right_xs)) / 2
        return lane_center - (width / 2)
    elif left_xs:
        return np.mean(left_xs) - (width / 2) + 100  # 偏左估中間
    elif right_xs:
        return np.mean(right_xs) - (width / 2) - 100  # 偏右估中間
    else:
        return 0.0

# ---------------------------
# Data Logging
# ---------------------------
def log_data(pic_num, img, side_value, wheel_value, lwheel_value, rwheel_value, speed_signed, seg_mask, width, height, lane_offset=0):
    img_name = f"{pic_num:05d}.png"
    img_path = os.path.join(folder_path, 'recorded_images', img_name)
    cv2.imwrite(img_path, img)

    entry = {
        "img_path": img_name,
        "steering": side_value,
        "throttle": wheel_value,
        "lwheel": lwheel_value,
        "rwheel": rwheel_value,
        "speed_signed": speed_signed,
        "lane_offset": lane_offset,
        "timestamp": datetime.datetime.now().isoformat()
    }
    data_log.append(entry)

def save_csv_log():
    df = pd.DataFrame(data_log)
    df.to_csv(os.path.join(folder_path, "log.csv"), index=False)

# ---------------------------
# Zebra crossing builder
# ---------------------------
def create_zebra_crossing(start_pos=[0, 0, 0.05], num_lines=6, spacing=0.3, line_size=[2, 0.2, 0.01]):
    for i in range(num_lines):
        basePosition = [start_pos[0], start_pos[1] + i * spacing, start_pos[2]]
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[line_size[0]/2, line_size[1]/2, line_size[2]/2])
        visBoxId = p.createVisualShape(p.GEOM_BOX, halfExtents=[line_size[0]/2, line_size[1]/2, line_size[2]/2], rgbaColor=[1,1,1,1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, baseVisualShapeIndex=visBoxId, basePosition=basePosition)

# ---------------------------
# PyBullet Initialization
# ---------------------------
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
gazebo_world_parser.parseWorld(p, filepath="worlds/new.world")
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)
create_zebra_crossing(start_pos=[5, 13.8, 0.0965], num_lines=9, spacing=0.3125)

# Humanoid
humanoidStartPos = [5, 13.3, 1]
humanoidStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
humanoid = p.loadURDF('human.urdf', humanoidStartPos, humanoidStartOrientation)
cid = p.createConstraint(humanoid, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [humanoidStartPos[0], humanoidStartPos[1], 0.5])
p.changeConstraint(cid, maxForce=50)

# Vehicle
r2d2StartPos = [2, 14.4, 2]
r2d2StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
r2d2 = p.loadURDF('real_car.urdf', r2d2StartPos, r2d2StartOrientation)
numJoints = p.getNumJoints(r2d2)

# Controls
d = 0.75
forward_speed = 20
pitch = p.addUserDebugParameter('camerapitch', 0, 360, 225)
yaw = p.addUserDebugParameter('camerayaw', 0, 360, 90)
distance = p.addUserDebugParameter('cameradistance', 0, 6, 2)

# Camera
width, height = 640, 480
fov, aspect, near, far = 60, width/height, 0.1, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Folder setup
now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
day_dir = now.strftime('%Y_%m_%d')
pic_num = 0
if not os.path.isdir(day_dir):
    os.mkdir(day_dir)
i = 1
while True:
    folder_name = str(i)
    folder_path = os.path.join(day_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, 'recorded_images'))
        if SAVE_IMG:
            os.makedirs(os.path.join(folder_path, 'deep'))
            os.makedirs(os.path.join(folder_path, 'segmentation'))
        break
    i += 1

# --- 主迴圈初始化 ---
# AI 變數
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ai_model = load_model(MODEL_PATH, device)
ai_transform = transforms.Compose([
        CustomTopCrop(CROP_TOP_PIXELS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 【*** 修正 2 ***】: 新增 speed_history 來儲存速度序列
image_history = deque(maxlen=SEQUENCE_LENGTH)
speed_history = deque(maxlen=SEQUENCE_LENGTH)
autodrive_enabled = False

# ---------------------------
# Main loop
# ---------------------------
recording = False
try:
    while True:
        keys = p.getKeyboardEvents()
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            recording = not recording
            print(f"[INFO] 模仿學習資料記錄 {'啟動' if recording else '暫停'}")

        if ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
            autodrive_enabled = not autodrive_enabled
            print(f"[INFO] AI driver {'on' if autodrive_enabled else 'off'}")
            # 切換模式時重設速度，避免暴衝
            actual_lwheel_value = 0
            actual_rwheel_value = 0
            # 清空歷史紀錄，避免使用舊資料預測
            image_history.clear()
            speed_history.clear()

        # Camera and Speed Calculation (This part must run every loop to get current state)
        r2d2_pos, r2d2_orn = p.getBasePositionAndOrientation(r2d2)
        p.resetDebugVisualizerCamera(
            cameraDistance=p.readUserDebugParameter(distance),
            cameraYaw=p.readUserDebugParameter(yaw),
            cameraPitch=p.readUserDebugParameter(pitch),
            cameraTargetPosition=r2d2_pos
        )

        camera_link_state = p.getLinkState(r2d2, numJoints - 1)
        camera_pos = camera_link_state[0]
        camera_orn = camera_link_state[1]
        camera_rot = p.getMatrixFromQuaternion(camera_orn)
        camera_forward = [camera_rot[0], camera_rot[3], camera_rot[6]]
        camera_up = [camera_rot[2], camera_rot[5], camera_rot[8]]
        camera_target = [camera_pos[0]+camera_forward[0], camera_pos[1]+camera_forward[1], camera_pos[2]+camera_forward[2]]

        view_matrix = p.computeViewMatrix(camera_pos, camera_target, camera_up)
        img_arr = p.getCameraImage(width, height, view_matrix, projection_matrix)
        rgb_img = img_arr[2]
        depth_buffer = np.reshape(img_arr[3], (height, width))
        seg_mask = np.reshape(img_arr[4], (height, width))

        img = np.reshape(np.array(rgb_img, dtype=np.uint8), (height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        linear_velocity, _ = p.getBaseVelocity(r2d2)
        speed_vec = np.array(linear_velocity)
        forward_vector = np.array([camera_forward[0], camera_forward[1], camera_forward[2]])
        speed_signed = np.dot(speed_vec, forward_vector)

        # 將當前畫面和速度存入歷史序列
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_history.append(pil_img)
        speed_history.append(speed_signed) # 【*** 修正 3 ***】: 將當前速度加入 history

        if autodrive_enabled and ai_model is not None:
            # --- AI 控制 ---
            # 確保序列已滿
            if len(image_history) == SEQUENCE_LENGTH:
                # 【*** 修正 4 ***】: 傳入影像和速度兩個序列
                pred_l, pred_r = predict_speeds(ai_model, list(image_history), list(speed_history), ai_transform, device)
                lwheel_value = pred_l
                rwheel_value = pred_r
            else:
                # 影像序列尚未集滿，暫不動作
                lwheel_value, rwheel_value = 0, 0
                # 在序列集滿前，可以印出提示
                print(f"Collecting data for AI... {len(image_history)}/{SEQUENCE_LENGTH}", end='\r')
        else:
            # --- 手動控制 ---
            wheel_value, side_value = 0, 0
            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                wheel_value = forward_speed
            elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                wheel_value = -forward_speed

            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                side_value = -1
            elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                side_value = 1

            rwheel_value = wheel_value * (1 - side_value * d)
            lwheel_value = wheel_value * (1 + side_value * d)

        # 慣性平滑
        actual_lwheel_value = (1 - alpha) * actual_lwheel_value + alpha * lwheel_value
        actual_rwheel_value = (1 - alpha) * actual_rwheel_value + alpha * rwheel_value

        for joint in [0, 1, 2, 3]:
            v = actual_lwheel_value if joint % 2 == 0 else actual_rwheel_value
            p.setJointMotorControl2(r2d2, joint, p.VELOCITY_CONTROL, targetVelocity=v)

        # HUD
        cv2.putText(img, f"Car Speed: {speed_signed:.2f} m/s", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
        for joint in [0, 1, 2, 3]:
            joint_state = p.getJointState(r2d2, joint)
            angular_velocity = joint_state[1]
            cv2.putText(img, f"Wheel {joint}: {angular_velocity:.2f} rad/s",
                        (10, 25 + joint * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)

        if recording:
            log_data(pic_num, img, side_value, wheel_value, actual_lwheel_value, actual_rwheel_value, speed_signed, seg_mask, width, height)
            depth_real = (far * near) / (far - (far - near) * depth_buffer)
            depth_mm = (depth_real * 1000).astype(np.uint16)
            cv2.imwrite(os.path.join(folder_path, 'deep', f"{pic_num:05d}.png"), depth_mm)
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)
            for obj_id in np.unique(seg_mask):
                color = [random.randint(0,255) for _ in range(3)]
                color_mask[seg_mask == obj_id] = color
            cv2.imwrite(os.path.join(folder_path, 'segmentation', f"{pic_num:05d}.png"), color_mask)
            pic_num += 1

        cv2.imshow("Car Camera", img)
        if cv2.waitKey(1) == 27:
            break
        p.stepSimulation()
        time.sleep(0.01)
finally:
    if SAVE_IMG and len(data_log) > 0:
        save_csv_log()
        print(f"[INFO] 已儲存 {len(data_log)} 筆模仿學習資料至：{folder_path}/log.csv")

        def split_dataset(csv_path, img_folder, output_folder,
                          val_ratio=0.1, test_ratio=0.1, random_state=42):
            df = pd.read_csv(csv_path)
            trainval_df, test_df = train_test_split(
                df, test_size=test_ratio, random_state=random_state, shuffle=True
            )
            val_size = val_ratio / (1 - test_ratio)
            train_df, val_df = train_test_split(
                trainval_df, test_size=val_size, random_state=random_state, shuffle=True
            )

            splits = {"train": train_df, "val": val_df, "test": test_df}
            for split_name, split_df in splits.items():
                split_img_dir = os.path.join(output_folder, split_name, "images")
                os.makedirs(split_img_dir, exist_ok=True)
                for _, row in split_df.iterrows():
                    src = os.path.join(img_folder, row["img_path"])
                    dst = os.path.join(split_img_dir, row["img_path"])
                    shutil.copy(src, dst)
                split_df.to_csv(os.path.join(output_folder, split_name, "log.csv"), index=False)

        split_dataset(
            csv_path=os.path.join(folder_path, "log.csv"),
            img_folder=os.path.join(folder_path, "recorded_images"),
            output_folder=folder_path,
            val_ratio=0.1,
            test_ratio=0.1
        )
        print("[INFO] 已將資料切分為 train/val/test 三組資料集")

    cv2.destroyAllWindows()
    p.disconnect()
