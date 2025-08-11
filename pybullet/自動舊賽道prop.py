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
from collections import deque # 用於高效地處理影像序列

# --- 1. AI 模型參數 (必須與訓練時完全一致) ---
MODEL_PATH = 'best_transformer_driver_model.pth' # 指定訓練好的模型檔案
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

# ---------------------------
# 操作說明
# ---------------------------
def information():
    print('\n##############\n快捷鍵操作說明\n##############')
    print('快捷鍵要在Pybullet鳥瞰視窗才能作用')
    print('按1控制行人左轉\t按2控制行人前進\t按3控制行人後退\t按4控制行人右轉')
    print('按8到大跑道左側 按9到小跑道 按0到大跑道右側')
    print('按z重設車子速度','按r切換錄影模式','按a切換自動駕駛')
    print('按t在終端輸入座標傳送到指定位置')
    print('按ESC退出程式'+'\n')

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

# ---------------------------
# Data Logging
# ---------------------------
def log_data(folder_path, pic_num, img, side_value, wheel_value, lwheel_value, rwheel_value, speed_signed, seg_mask, width, height, autodrive_enabled):
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
        "is_autodrive": autodrive_enabled,
        "timestamp": datetime.datetime.now().isoformat()
    }
    data_log.append(entry)

def save_csv_log(folder_path):
    df = pd.DataFrame(data_log)
    df.to_csv(os.path.join(folder_path, "log.csv"), index=False)

# ---------------------------
# Data Split
# ---------------------------
def split_dataset(csv_path, img_folder, output_folder, val_ratio=0.1, test_ratio=0.1, random_state=42):
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
        
        # 建立一個集合來追蹤已複製的圖片
        copied_images = set()

        for _, row in split_df.iterrows():
            img_name = row["img_path"]
            if img_name not in copied_images:
                src = os.path.join(img_folder, img_name)
                dst = os.path.join(split_img_dir, img_name)
                shutil.copy(src, dst)
                copied_images.add(img_name)
        
        split_df.to_csv(os.path.join(output_folder, split_name, "log.csv"), index=False)
    print("[INFO] 已將資料切分為 train/val/test 三組資料集")


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
# 三角錐函數
# ---------------------------
def place_cones(positions):
    cone_ids = []
    cone_urdf_path = "cone.urdf"

    for pos in positions:
        try:
            cone_id = p.loadURDF(cone_urdf_path, basePosition=pos)
            cone_ids.append(cone_id)
        except p.error as e:
            print(f"[ERROR] 載入 URDF 失敗於位置 {pos}。錯誤訊息：{e}")
            return []
    return cone_ids

# ---------------------------
# 樹木函數
# ---------------------------
def place_trees(positions):
    tree_ids = []
    tree_urdf_path = "tree.urdf"
    
    for pos in positions:
        try:
            tree_id = p.loadURDF(tree_urdf_path, basePosition=pos)
            tree_ids.append(tree_id)
        except p.error as e:
            print(f"[ERROR] 載入樹木 URDF 失敗於位置 {pos}。錯誤訊息：{e}")
            return []
    return tree_ids

# ---------------------------
# 將建立資料夾的邏輯封裝成一個函式
# ---------------------------
def setup_recording_folders():
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    day_dir = now.strftime('%Y_%m_%d')
    if not os.path.isdir(day_dir):
        os.mkdir(day_dir)
    i = 1
    while True:
        folder_name = str(i)
        path = os.path.join(day_dir, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(os.path.join(path, 'recorded_images'))
            if SAVE_IMG:
                os.makedirs(os.path.join(path, 'deep'))
                os.makedirs(os.path.join(path, 'segmentation'))
                print(f"[INFO] 建立新資料夾於: {path}")
            return path
        i += 1


# ---------------------------
# PyBullet Initialization
# ---------------------------
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
gazebo_world_parser.parseWorld(p, filepath="worlds/0_prop.world")
p.loadURDF("plane.urdf")
p.setAdditionalSearchPath(os.path.join(os.getcwd(), "3Dmodel"))
p.loadURDF("small_track.urdf", basePosition=[0,0,0.055])
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)
create_zebra_crossing(start_pos=[-4.5, -15.9, 0.0965], num_lines=7, spacing=0.3125)

cone_positions = [
    [6.5, 14, 0.1], [2, 18, 0.1], [-22, 15, 0.1],
    [7, -14.3, 0.1], [-19, -14.3, 0.1], [-10, 15, 0.1]
]
placed_cone_ids = place_cones(cone_positions)

tree_positions = [
    [0 ,1, 0.0], [-11, 18, 0.0], [-15, 10, 0.0],
    [-18, 12, 0.0], [-20, 18, 0.0]
]
placed_tree_ids = place_trees(tree_positions)

# 人
humanoidStartPos = [-4.5, -15.5, 0.09]
humanoidStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
humanoid = p.loadURDF("man.urdf", humanoidStartPos, humanoidStartOrientation)
cid = p.createConstraint(humanoid, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [humanoidStartPos[0], humanoidStartPos[1], 0.5])
p.changeConstraint(cid, maxForce=50)
current_yaw = np.pi/2
move_direction = 0
is_forward_pressed = False
is_backward_pressed = False
last_pos, _ = p.getBasePositionAndOrientation(humanoid)

# 車
r2d2StartPos = [-7, -15.5, 0.35]
r2d2StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
r2d2 = p.loadURDF("front_car.urdf", r2d2StartPos, r2d2StartOrientation)
numJoints = p.getNumJoints(r2d2)

# Controls
d = 0.75
forward_speed = 20
pitch = p.addUserDebugParameter('camerapitch', 0, 360, 269.9999)
yaw = p.addUserDebugParameter('camerayaw', 0, 360, 90)
distance = p.addUserDebugParameter('cameradistance', 0, 100, 10) 
speed_slider = p.addUserDebugParameter('speed', -50, 50, 20)

# Camera
width, height = 640, 480
fov, aspect, near, far = 60, width/height, 0.1, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# --- 主迴圈初始化 ---
# AI 變數
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ai_model = load_model(MODEL_PATH, device)
ai_transform = transforms.Compose([
    transforms.Lambda(lambda img: TF.crop(img, CROP_TOP_PIXELS, 0, ORIGINAL_HEIGHT - CROP_TOP_PIXELS, ORIGINAL_WIDTH)),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_history = deque(maxlen=SEQUENCE_LENGTH)
autodrive_enabled = False

# 主迴圈變數
recording = False
first_record_press = True
folder_path = None
pic_num = 0
information()

try:
    while True:
        keys = p.getKeyboardEvents()
        forward_speed = p.readUserDebugParameter(speed_slider)
        
        # 重設速度滑桿
        if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
            p.removeUserDebugItem(speed_slider)
            speed_slider = p.addUserDebugParameter('speed', -50, 50, 20)
            print("[INFO] 速度已重設為 20") 
        
        # 切換錄影模式
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            recording = not recording
            print(f"[INFO] 模仿學習資料記錄 {'啟動' if recording else '暫停'}")
            if recording and first_record_press:
                folder_path = setup_recording_folders()
                first_record_press = False

        # 【新功能】切換自動駕駛模式
        if ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
            autodrive_enabled = not autodrive_enabled
            print(f"[INFO] AI driver {'on' if autodrive_enabled else 'off'}")
            # 切換模式時重設速度，避免暴衝
            actual_lwheel_value = 0
            actual_rwheel_value = 0

        # 車輛傳送
        if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
            print("\n----------------------------------------------------")
            print("請輸入車子的新座標 (X Y Z)，以空格分隔。例如：10 5 0.35")
            try:
                p.setRealTimeSimulation(0)
                coords_input = input("新座標 (X Y Z): ")
                x, y, z = map(float, coords_input.split())
                new_pos = [x, y, z]
                _, current_orn = p.getBasePositionAndOrientation(r2d2)
                p.resetBasePositionAndOrientation(r2d2, new_pos, current_orn)
                print(f"[INFO] 成功傳送車子至新座標: ({x:.2f}, {y:.2f}, {z:.2f})")
            except ValueError:
                print("[ERROR] 輸入格式錯誤！請確保您輸入了三個以空格分隔的數字。")
            finally:
                p.setRealTimeSimulation(1)

        if ord('8') in keys and keys[ord('8')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [2.3, 14.4, 0.35], [0, 0, 0, 1])
        if ord('9') in keys and keys[ord('9')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [0, -1.225, 0.5], [0, 0, 0, 1])            
        if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [-7, -15.5, 0.35], [0, 0, 0, 1])  

        # 更新行人移動狀態
        if ord('2') in keys:
            if keys[ord('2')] & p.KEY_IS_DOWN: is_forward_pressed = True
            if keys[ord('2')] & p.KEY_WAS_RELEASED: is_forward_pressed = False
        if ord('3') in keys:
            if keys[ord('3')] & p.KEY_IS_DOWN: is_backward_pressed = True
            if keys[ord('3')] & p.KEY_WAS_RELEASED: is_backward_pressed = False
        move_direction = 1 if is_forward_pressed else -1 if is_backward_pressed else 0
        if ord('1') in keys and keys[ord('1')] & p.KEY_IS_DOWN: current_yaw += 0.05
        if ord('4') in keys and keys[ord('4')] & p.KEY_IS_DOWN: current_yaw -= 0.05
        
        pos, _ = p.getBasePositionAndOrientation(humanoid)
        if move_direction != 0:
            dir_x = [np.cos(current_yaw), np.sin(current_yaw), 0]
            move_speed = 0.04 * move_direction
            last_pos = [pos[0] + dir_x[0]*move_speed, pos[1] + dir_x[1]*move_speed, pos[2]]
        else:
            last_pos = list(pos)
        stand_orientation = p.getQuaternionFromEuler([0, 0, current_yaw])
        p.resetBasePositionAndOrientation(humanoid, last_pos, stand_orientation)

        # -------------------
        # 駕駛模式邏輯
        # -------------------
        lwheel_value, rwheel_value = 0, 0
        side_value, wheel_value = 0, 0
        if autodrive_enabled and ai_model is not None:
            if len(image_history) == SEQUENCE_LENGTH:
                pred_l, pred_r = predict_speeds(ai_model, list(image_history), ai_transform, device)
                lwheel_value = pred_l
                rwheel_value = pred_r
                # 計算 steering 和 throttle 供記錄
                wheel_value = (lwheel_value + rwheel_value) / 2
                if wheel_value != 0:
                    side_value = (lwheel_value - rwheel_value) / (2 * wheel_value * d)
                else:
                    side_value = 0
            else:
                lwheel_value, rwheel_value = 0, 0
        else:
            # 手動控制
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

        # Camera
        r2d2_pos, r2d2_orn = p.getBasePositionAndOrientation(r2d2)
        p.resetDebugVisualizerCamera(
            cameraDistance=p.readUserDebugParameter(distance),
            cameraYaw=p.readUserDebugParameter(yaw),
            cameraPitch=p.readUserDebugParameter(pitch),
            cameraTargetPosition=r2d2_pos
        )
        camera_link_state = p.getLinkState(r2d2, numJoints - 1)
        camera_pos, camera_orn = camera_link_state[0], camera_link_state[1]
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
        
        # 將當前畫面存入歷史序列 (PIL Image格式)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_history.append(pil_img)

        # Speed
        linear_velocity, _ = p.getBaseVelocity(r2d2)
        speed_vec = np.array(linear_velocity)
        forward_vector = np.array([camera_forward[0], camera_forward[1], camera_forward[2]])
        speed_signed = np.dot(speed_vec, forward_vector)

        # HUD
        hud_text = f"XYZ: ({r2d2_pos[0]:.3f}, {r2d2_pos[1]:.3f}, {r2d2_pos[2]:.3f})"
        cv2.putText(img, hud_text, (290, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Mode: {'Auto' if autodrive_enabled else 'Manual'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"Recording: {'ON' if recording else 'OFF'}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if recording else (0, 0, 255), 2)
        for joint in [0, 1, 2, 3]:
            joint_state = p.getJointState(r2d2, joint)
            angular_velocity = joint_state[1]
            cv2.putText(img, f"Wheel {joint}: {angular_velocity:.2f} rad/s",
                        (10, 20 + joint * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)
        cv2.putText(img, f"Car Speed: {speed_signed:.2f} m/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 200), 2)
        
        if recording:
            log_data(folder_path, pic_num, img, side_value, wheel_value, actual_lwheel_value, actual_rwheel_value, speed_signed, seg_mask, width, height, autodrive_enabled)
            
            # 只有在手動錄製模式才儲存深度圖和分割圖，因為自動駕駛模式通常不需要
            if not autodrive_enabled and SAVE_IMG:
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
    if folder_path and SAVE_IMG and len(data_log) > 0:
        save_csv_log(folder_path)
        print(f"[INFO] 已儲存 {len(data_log)} 筆模仿學習資料至：{folder_path}/log.csv")
        
        # 執行切分
        split_dataset(
            csv_path=os.path.join(folder_path, "log.csv"),
            img_folder=os.path.join(folder_path, "recorded_images"),
            output_folder=folder_path,
            val_ratio=0.1,
            test_ratio=0.1
        )

    cv2.destroyAllWindows()