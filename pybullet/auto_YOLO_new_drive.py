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
import torchvision.transforms as transforms
import torchvision.models as models
import math
import torchvision.transforms.functional as TF
from PIL import Image
from ultralytics import YOLO
from collections import deque # 用於高效地處理影像序列

# --- 1. AI 模型參數 (必須與訓練時完全一致) ---
MODEL_PATH = 'best_transformer_driver_model_3.pth' # 指定訓練好的模型檔案
SEQUENCE_LENGTH = 15
IMG_HEIGHT = 224
IMG_WIDTH = 224
D_MODEL = 512  # ResNet-18 的輸出維度
N_HEAD = 8
N_LAYERS = 3
DROPOUT = 0.3

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

# --- 3. AI 模型載入與預測函式 ---
def load_model(model_path, device):
    """載入訓練好的模型"""
    print("正在載入 AI 模型...")
    model = VisionTransformerDriver(D_MODEL, N_HEAD, N_LAYERS, DROPOUT)
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

def predict_speeds(model, image_sequence, transform, device):
    """使用模型預測速度"""
    if model is None or len(image_sequence) < SEQUENCE_LENGTH:
        return 0, 0 # 如果模型或影像序列有問題，返回安全值

    # 預處理影像序列
    processed_sequence = [transform(img) for img in image_sequence]
    input_tensor = torch.stack(processed_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    predicted_speeds = prediction.cpu().numpy().flatten()
    return predicted_speeds[0], predicted_speeds[1] # lwheel, rwheel

# ==================== 新增：YOLO 緊急煞車判斷邏輯 ====================
def is_emergency_brake_required(detections, img_width, img_height, confidence_threshold=0.4, critical_area_threshold=0.03, critical_x_center=0.5, critical_x_width=0.7):
    """
    根據 YOLO 的行人偵測結果，判斷是否需要緊急煞車。

    Args:
        detections: YOLO 模型的偵測結果。
        img_width (int): 影像寬度。
        img_height (int): 影像高度。
        confidence_threshold (float): 偵測信心的門檻值。
        critical_area_threshold (float): 認定為危險的邊界框面積佔比。
        critical_x_center (float): 危險區域的水平中心點 (0.0 到 1.0)。
        critical_x_width (float): 危險區域的總寬度 (0.0 到 1.0)。

    Returns:
        bool: 如果需要緊急煞車則返回 True，否則返回 False。
    """
    for r in detections:
        for box in r.boxes:
            # 檢查是否為 'person' (在 COCO 資料集中，類別 ID 0 是 'person') 且信心足夠
            if box.cls[0] == 0 and box.conf[0] > confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0]
                box_area = (x2 - x1) * (y2 - y1)
                box_x_center = (x1 + x2) / 2
                image_area = img_width * img_height
                area_ratio = box_area / image_area

                # 判斷行人是否在車輛前方的危險區域內
                is_in_critical_x = abs(box_x_center / img_width - critical_x_center) < (critical_x_width / 2)

                if area_ratio > critical_area_threshold and is_in_critical_x:
                    print(f"!!! 緊急煞車觸發 !!! 偵測到行人在危險區域。面積佔比: {area_ratio:.2%}")
                    return True
    return False
# =======================================================================

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
#gazebo_world_parser.parseWorld(p, filepath="worlds/new.world")
#p.loadURDF(r"D:\Code\full_track_nowall.urdf", basePosition=[0,0,0.1])
p.loadURDF("plane.urdf")
p.setAdditionalSearchPath(os.path.join(os.getcwd(), "3Dmodel"))
p.loadURDF("test.urdf", basePosition=[0,0,-0.071])
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)
#create_zebra_crossing(start_pos=[5, 13.8, 0.0965], num_lines=9, spacing=0.3125)

# 人
humanoidStartPos = [5, 13.3, 0.09]
humanoidStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
humanoid = p.loadURDF('man.urdf', humanoidStartPos, humanoidStartOrientation)
cid = p.createConstraint(humanoid, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [humanoidStartPos[0], humanoidStartPos[1], 0.5])
p.changeConstraint(cid, maxForce=50)
current_yaw = np.pi/2
move_direction = 0
is_forward_pressed = False
is_backward_pressed = False
last_pos, _ = p.getBasePositionAndOrientation(humanoid)

# 車
r2d2StartPos = [0, 1.25, 2]
r2d2StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
r2d2 = p.loadURDF('front_car.urdf', r2d2StartPos, r2d2StartOrientation)
numJoints = p.getNumJoints(r2d2)

# Controls
d = 0.75
forward_speed = 20
#pitch = p.addUserDebugParameter('camerapitch', 0, 360, 225)
pitch = p.addUserDebugParameter('camerapitch', 0, 360, 269)
yaw = p.addUserDebugParameter('camerayaw', 0, 360, 90)
#distance = p.addUserDebugParameter('cameradistance', 0, 6, 2) 
distance = p.addUserDebugParameter('cameradistance', 0, 100, 10)
speed_slider = p.addUserDebugParameter('speed', -50, 50, 20)


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

# ==================== 新增：載入 YOLO 模型 ====================
pedestrian_model = None
if YOLO is not None:
    print("正在載入 YOLOv8 行人偵測模型...")
    try:
        pedestrian_model = YOLO('yolov8n.pt') # yolov8n.pt 是一個輕量且快速的模型
        pedestrian_model.to(device)
        print("YOLOv8 模型載入成功。")
    except Exception as e:
        print(f"載入 YOLO 模型時發生錯誤: {e}")
        pedestrian_model = None
# ===============================================================

ai_transform = transforms.Compose([
    transforms.Lambda(lambda img: TF.crop(img, CROP_TOP_PIXELS, 0, ORIGINAL_HEIGHT - CROP_TOP_PIXELS, ORIGINAL_WIDTH)),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_history = deque(maxlen=SEQUENCE_LENGTH)
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
        forward_speed = p.readUserDebugParameter(speed_slider)
        # <<< 修正 >>> 使用 "移除再新增" 的方法來重設拉桿數值
        if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
            # 1. 移除舊的拉桿
            p.removeUserDebugItem(speed_slider)
            # 2. 用新的預設值重新建立它，並將回傳的新ID存回變數中
            speed_slider = p.addUserDebugParameter('speed', -50, 50, 20)
            print("[INFO] 速度已重設為 20")      
         # 【新功能】切換自動駕駛模式
        if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
            autodrive_enabled = not autodrive_enabled
            print(f"[INFO] AI driver {'on' if autodrive_enabled else 'off'}")
            # 切換模式時重設速度，避免暴衝
            actual_lwheel_value = 0
            actual_rwheel_value = 0

        if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED: #飛高高
            p.resetBasePositionAndOrientation(r2d2, [0, 1.25, 25], [0, 0, 0, 1])
        if ord('9') in keys and keys[ord('9')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [0, 1.25, 0.35], [0, 0, 0, 1])             
        if ord('8') in keys and keys[ord('8')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [-12.5, 13.2, 0.35], [0, 0, 0, 1])       
        if ord('7') in keys and keys[ord('7')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [-5.16, 30.94, 0.35], [0, 0, 0, 1])
        if ord('6') in keys and keys[ord('6')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [0.6, 14.4, 0.35], [0, 0, 0, 1])    
        if ord('5') in keys and keys[ord('5')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [14.2, 25.9, 0.35], [0, 0, 0, 1]) 

        # 更新行人移動狀態
        if ord('2') in keys:
            if keys[ord('2')] & p.KEY_IS_DOWN:
                is_forward_pressed = True
            if keys[ord('2')] & p.KEY_WAS_RELEASED:
                is_forward_pressed = False

        if ord('3') in keys:
            if keys[ord('3')] & p.KEY_IS_DOWN:
                is_backward_pressed = True
            if keys[ord('3')] & p.KEY_WAS_RELEASED:
                is_backward_pressed = False

        if is_forward_pressed:
            move_direction = 1
        elif is_backward_pressed:
            move_direction = -1
        else:
            move_direction = 0

        # 更新行人朝向（左右轉）
        if ord('1') in keys and keys[ord('1')] & p.KEY_IS_DOWN:
            current_yaw += 0.05
        if ord('4') in keys and keys[ord('4')] & p.KEY_IS_DOWN:
            current_yaw -= 0.05


        if autodrive_enabled and ai_model is not None:
            # --- AI 控制 (整合 YOLO 安全覆寫) ---
            lwheel_value, rwheel_value = 0, 0

            if len(image_history) == SEQUENCE_LENGTH:
                brake_triggered = False
                
                # 1. 執行行人偵測安全檢查
                if pedestrian_model is not None:
                    latest_pil_img = image_history[-1]
                    detections = pedestrian_model(latest_pil_img, verbose=False) 
                    
                    if is_emergency_brake_required(detections, width, height):
                        # 若偵測到危險，覆寫速度為緊急煞車指令
                        lwheel_value, rwheel_value = -20, -20 # 強力煞車
                        brake_triggered = True

                # 2. 若未觸發煞車，則使用 Transformer 模型進行正常駕駛
                if not brake_triggered:
                    pred_l, pred_r = predict_speeds(ai_model, list(image_history), ai_transform, device)
                    lwheel_value = pred_l
                    rwheel_value = pred_r
            # 若影像序列未滿，lwheel_value 和 rwheel_value 會維持為 0，車輛靜止
        
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
        # =======================================================

        # 慣性平滑
        actual_lwheel_value = (1 - alpha) * actual_lwheel_value + alpha * lwheel_value
        actual_rwheel_value = (1 - alpha) * actual_rwheel_value + alpha * rwheel_value

        for joint in [0, 1, 2, 3]:
            v = actual_lwheel_value if joint % 2 == 0 else actual_rwheel_value
            p.setJointMotorControl2(r2d2, joint, p.VELOCITY_CONTROL, targetVelocity=v)

        # Camera
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
        # lane_offset = get_lane_offset_by_opencv(img, width)

        # 將當前畫面存入歷史序列 (PIL Image格式)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_history.append(pil_img)
        
        # Speed
        linear_velocity, _ = p.getBaseVelocity(r2d2)
        speed_vec = np.array(linear_velocity)
        forward_vector = np.array([camera_forward[0], camera_forward[1], camera_forward[2]])
        speed_signed = np.dot(speed_vec, forward_vector)

        # HUD
        cv2.putText(img, f"Car Speed: {speed_signed:.2f} m/s", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
        # cv2.putText(img, f"Lane Offset: {lane_offset:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)
        # 顯示四個輪子的角速度
        for joint in [0, 1, 2, 3]:
            joint_state = p.getJointState(r2d2, joint)
            angular_velocity = joint_state[1]  # jointState[1] 是角速度
            cv2.putText(img, f"Wheel {joint}: {angular_velocity:.2f} rad/s",
                        (10, 25 + joint * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)

        # 顯示模式狀態
        mode_text = f"Mode: {'AI Driver' if autodrive_enabled else 'Manual'}"
        mode_color = (0, 255, 0) if autodrive_enabled else (255, 255, 0)
        cv2.putText(img, mode_text, (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        if autodrive_enabled and brake_triggered:
             cv2.putText(img, "EMERGENCY BRAKE!", (width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
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

        # 新增切分功能
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

        # 執行切分
        split_dataset(
            csv_path=os.path.join(folder_path, "log.csv"),
            img_folder=os.path.join(folder_path, "recorded_images"),
            output_folder=folder_path,
            val_ratio=0.1,
            test_ratio=0.1
        )
        print("[INFO] 已將資料切分為 train/val/test 三組資料集")

    cv2.destroyAllWindows()
