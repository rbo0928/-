import pybullet as p
from pybullet_utils import gazebo_world_parser
import pybullet_data
import cv2
import time
import numpy as np
import datetime, os
import pandas as pd
from sklearn.model_selection import train_test_split

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
SEQUENCE_LENGTH = 20
IMG_HEIGHT = 224
IMG_WIDTH = 224
D_MODEL = 512  # ResNet-18 的輸出維度
N_HEAD = 8
N_LAYERS = 3
DROPOUT = 0.4

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


# ---------------------------
# Lane offset (via OpenCV)
# ---------------------------
def get_lane_offset_by_opencv(img, width):

    # Step 1: 提取白色區域（避免抓到柏油）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    masked = cv2.bitwise_and(img, img, mask=white_mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
   # # Step 2: Canny 邊緣
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
# <<< 修改 >>> 增加 folder_path 參數，避免依賴全域變數
def log_data(folder_path, pic_num, img, side_value, wheel_value, lwheel_value, rwheel_value, speed_signed, seg_mask, width, height):
    img_name = f"{pic_num:05d}.png"
    img_path = os.path.join(folder_path, 'recorded_images', img_name)
    cv2.imwrite(img_path, img)

    entry = {
        "img_path": img_name,
        "steering": side_value,
        "throttle": wheel_value,
        "lwheel": actual_lwheel_value ,
        "rwheel": actual_rwheel_value,
        "speed_signed": speed_signed,
        # "lane_offset": lane_offset, # 您原本的程式碼已註解此行
        "timestamp": datetime.datetime.now().isoformat()
    }
    data_log.append(entry)

# <<< 修改 >>> 增加 folder_path 參數
def save_csv_log(folder_path):
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
# <<< 新增 >>> 將建立資料夾的邏輯封裝成一個函式
# ---------------------------
def setup_recording_folders():
    """
    建立本次執行的資料記錄資料夾，並返回資料夾路徑。
    """
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
                print(f"[INFO] 建立新資料夾於: {path}")
            return path
        i += 1

# ---------------------------
# PyBullet Initialization
# ---------------------------
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF(r"0710/test.urdf", basePosition=[0,0,0.1])
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)
#create_zebra_crossing(start_pos=[5, 13.8, 0.0965], num_lines=9, spacing=0.3125)

# Humanoid
humanoidStartPos = [5, 13.3, 1]
humanoidStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
humanoid = p.loadURDF('human.urdf', humanoidStartPos, humanoidStartOrientation)
cid = p.createConstraint(humanoid, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [humanoidStartPos[0], humanoidStartPos[1], 0.5])
p.changeConstraint(cid, maxForce=50)
current_yaw = np.pi/2
move_direction = 0
is_forward_pressed = False
is_backward_pressed = False
last_pos, _ = p.getBasePositionAndOrientation(humanoid)

# Vehicle
r2d2StartPos = [0, 1.25, 2]
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

# <<< 修改 >>> 移除啟動時就建立資料夾的程式碼
# pic_num = 0
# if not os.path.isdir(day_dir): ... (整段移除)

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

# ---------------------------
# Main loop
# ---------------------------
recording = False
first_record_press = True # <<< 新增 >>> 狀態旗標，判斷是否為第一次按下錄製
folder_path = None      # <<< 新增 >>> 初始化 folder_path
pic_num = 0             # <<< 新增 >>> 初始化圖片編號

try:
    while True:
        keys = p.getKeyboardEvents()
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            recording = not recording
            print(f"[INFO] 模仿學習資料記錄 {'啟動' if recording else '暫停'}")
            
            # <<< 修改 >>> 核心邏輯：當開始錄製且是第一次按下時，才建立資料夾
            if recording and first_record_press:
                folder_path = setup_recording_folders()
                first_record_press = False # 更新旗標，確保不再重複建立
        
        if ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
            autodrive_enabled = not autodrive_enabled
            print(f"[INFO] AI driver {'on' if autodrive_enabled else 'off'}")
            # 切換模式時重設速度，避免暴衝
            actual_lwheel_value = 0
            actual_rwheel_value = 0
        
        if autodrive_enabled and ai_model is not None:
            if len(image_history) == SEQUENCE_LENGTH:
                pred_l, pred_r = predict_speeds(ai_model, list(image_history), ai_transform, device)
                actual_lwheel_value = pred_l
                actual_rwheel_value = pred_r
            else:
                actual_lwheel_value, actual_rwheel_value = 0, 0
        else:
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

        image_history.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

        # Lane offset
        # lane_offset = get_lane_offset_by_opencv(img, width)

        # Speed
        linear_velocity, _ = p.getBaseVelocity(r2d2)
        speed_vec = np.array(linear_velocity)
        forward_vector = np.array([camera_forward[0], camera_forward[1], camera_forward[2]])
        speed_signed = np.dot(speed_vec, forward_vector)

        # HUD
        #hud_text = f"XYZ: ({r2d2_pos[0]:.3f}, {r2d2_pos[1]:.3f}, {r2d2_pos[2]:.3f})"
        #cv2.putText(img, hud_text, (290, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        #for joint in [0, 1, 2, 3]:
        #    joint_state = p.getJointState(r2d2, joint)
        #    angular_velocity = joint_state[1]
        #    cv2.putText(img, f"Wheel {joint}: {angular_velocity:.2f} rad/s",
        #                (10, 20 + joint * 25),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)
        #cv2.putText(img, f"Car Speed: {speed_signed:.2f} m/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 200), 2)
        
        if recording:
            # <<< 修改 >>> 傳入 folder_path 參數
            log_data(folder_path, pic_num, img, side_value, wheel_value, lwheel_value, rwheel_value, speed_signed, seg_mask, width, height)
            pic_num += 1

        cv2.imshow("Car Camera", img)
        if cv2.waitKey(1) == 27:
            break
        p.stepSimulation()
        time.sleep(0.01)
finally:
    # <<< 修改 >>> 增加條件判斷，確保只有在錄製過資料時才存檔
    if folder_path and SAVE_IMG and len(data_log) > 0:
        save_csv_log(folder_path) # <<< 修改 >>> 傳入 folder_path
        print(f"[INFO] 已儲存 {len(data_log)} 筆模仿學習資料至：{folder_path}/log.csv")

    cv2.destroyAllWindows()