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
import sys
import os

# 將專案根目錄添加到Python路徑中
# 這樣才能正確地從 pybullet 資料夾引用 trainingmodel 模組
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 移植自 自動新賽道.py 的 AI 模型整合部分 ---
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import math
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from PIL import Image
from collections import deque # 用於高效地處理影像序列
from trainingmodel.transformer_model_beta5 import PositionalEncoding, VisionTransformerDriver # 確保這個路徑正確

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
# --- 移植結束 ---

data_log = []
SAVE_IMG = True
actual_lwheel_value = 0
actual_rwheel_value = 0
alpha = 0.3  # 越小回復越慢
autodrive_enabled = False # <<< 新增 >>> 自動駕駛模式狀態旗標

# ---------------------------
# 操作說明
# ---------------------------
def information():
    print('\n##############快捷鍵操作說明##############')
    print('快捷鍵要在Pybullet鳥瞰視窗才能作用')
    print('按1控制行人左轉\t按2控制行人前進\t按3控制行人後退\t按4控制行人右轉')
    print('按5 6 7 8 9 0傳送 ')
    print('按z重設車子速度','按r切換錄影模式','按a切換自動駕駛') # <<< 修正 >>> 新增自動駕駛說明
    print('按t在終端輸入座標傳送到指定位置')
    print('退出前記得按r結束錄影，在按esc退出，不然會沒有log')
    print('\n#########################################')

# ---------------------------
# 斑馬線函數
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
    cone_urdf_path = r"cone.urdf"
    for pos in positions:
        try:
            # 嘗試載入 URDF 檔案
            cone_id = p.loadURDF(cone_urdf_path, basePosition=pos)
            cone_ids.append(cone_id)
            print(f"[INFO] 成功在座標 {pos} 放置三角錐，ID 為 {cone_id}")
        except p.error as e:
            # 如果載入失敗，捕捉錯誤並列印出詳細資訊
            print(f"[ERROR] 載入 URDF 失敗於位置 {pos}。錯誤訊息：{e}")
            print("[HINT] 這可能是因為 URDF 文件本身有語法錯誤，或者 URDF 文件內部引用的模型檔案 (.obj, .stl) 不存在。")
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
            # 載入樹木 URDF
            tree_id = p.loadURDF(tree_urdf_path, basePosition=pos)
            tree_ids.append(tree_id)
            print(f"[INFO] 成功在座標 {pos} 放置樹木，ID 為 {tree_id}")
        except p.error as e:
            # 處理載入錯誤，提供更詳細的提示
            print(f"[ERROR] 載入樹木 URDF 失敗於位置 {pos}。錯誤訊息：{e}")
            print("[HINT] 這可能是因為 URDF 文件本身有語法錯誤，或者內部引用的模型檔案 (.obj, .stl) 不存在。")
            return []
    
    return tree_ids

# ---------------------------
# PyBullet 環境建置
# ---------------------------
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setAdditionalSearchPath(os.path.join(os.getcwd(), "3Dmodel"))
p.loadURDF("test.urdf", basePosition=[0,0,-0.07])
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)

# 斑馬線
create_zebra_crossing(start_pos=[-14,-10.2, 0.001], num_lines=4, spacing=0.4)

# 三角錐
cone_positions = [
    [6.5, 14, 0.1],
    [2, 18, 0.1],
    [-22, 15, 0.1],
    [7, -14.3, 0.1],
    [-14, 1, 0.1],
     [-10, 15, 0.1],
     [12.3,4.03,0.1],
     [-5,-9.25,0.1]
]
placed_cone_ids = place_cones(cone_positions)

# 樹
tree_positions = [
    [0 ,-1, 0.0],
    [-11, 18, 0.0],
    [-15, 10, 0.0],
    [-18, 12, 0.0],
    [-20, 18, 0.0]
]
placed_tree_ids = place_trees(tree_positions)

# 人
humanoidStartPos = [-14,-9.8, 0.02]
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
r2d2StartPos = [0, 1.25, 2]
r2d2StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
r2d2 = p.loadURDF("front_car.urdf", r2d2StartPos, r2d2StartOrientation)
numJoints = p.getNumJoints(r2d2)

# 控制
d = 0.75
forward_speed = 20
#pitch = p.addUserDebugParameter('camerapitch', 0, 360, 225)
pitch = p.addUserDebugParameter('camerapitch', 0, 360, 269.9999)
yaw = p.addUserDebugParameter('camerayaw', 0, 360, 90)
#distance = p.addUserDebugParameter('cameradistance', 0, 6, 2) 
distance = p.addUserDebugParameter('cameradistance', 0, 100, 10) 
speed_slider = p.addUserDebugParameter('speed', -50, 50, 20)

# 鏡頭
width, height = 640, 480
fov, aspect, near, far = 60, width/height, 0.1, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# <<< 移植部分：AI 相關初始化 >>>
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ai_model = load_model(MODEL_PATH, device)
ai_transform = transforms.Compose([
    transforms.Lambda(lambda img: TF.crop(img, CROP_TOP_PIXELS, 0, ORIGINAL_HEIGHT - CROP_TOP_PIXELS, ORIGINAL_WIDTH)),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_history = deque(maxlen=SEQUENCE_LENGTH)
# <<< 移植結束 >>>

# ---------------------------
# 主迴圈
# ---------------------------
recording = False
first_record_press = True # <<< 新增 >>> 狀態旗標，判斷是否為第一次按下錄製
folder_path = None      # <<< 新增 >>> 初始化 folder_path
pic_num = 0             # <<< 新增 >>> 初始化圖片編號
information()
try:
    while True:
        keys = p.getKeyboardEvents()
        forward_speed = p.readUserDebugParameter(speed_slider)
        # <<< 修正 >>> 使用 "移除再新增" 的方法來重設拉桿數值
        if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
            # 1. 移除舊的拉桿
            p.removeUserDebugItem(speed_slider)
            # 2. 用新的預設值重新建立它，並將回傳的新ID存回變數中
            speed_slider = p.addUserDebugParameter('speed', -50, 50, 20)
            print("[INFO] 速度已重設為 20")
        
        # <<< 移植部分：切換自動駕駛模式的按鍵處理 >>>
        if ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
            autodrive_enabled = not autodrive_enabled
            print(f"[INFO] AI driver {'on' if autodrive_enabled else 'off'}")
            # 切換模式時重設速度，避免暴衝
            actual_lwheel_value = 0
            actual_rwheel_value = 0

        if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
            print("\n----------------------------------------------------")
            print("請輸入車子的新座標 (X Y Z)，以空格分隔。例如：10 5 0.35")
            
            try:
                # 暫停 PyBullet 的即時模擬，等待使用者輸入
                p.setRealTimeSimulation(0)
                
                # 讀取使用者輸入的座標字串
                coords_input = input("新座標 (X Y Z): ")
                
                # 將字串分割並轉換為浮點數
                x, y, z = map(float, coords_input.split())
                new_pos = [x, y, z]

                # 獲取車子當前的方向，並將其維持不變
                _, current_orn = p.getBasePositionAndOrientation(r2d2)
                
                # 傳送車子到新位置
                p.resetBasePositionAndOrientation(r2d2, new_pos, current_orn)
                
                print(f"[INFO] 成功傳送車子至新座標: ({x:.2f}, {y:.2f}, {z:.2f})")
                
            except ValueError:
                # 處理使用者輸入格式錯誤的情況
                print("[ERROR] 輸入格式錯誤！請確保您輸入了三個以空格分隔的數字。")
            finally:
                # 恢復 PyBullet 的即時模擬
                p.setRealTimeSimulation(1)


        if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED: #飛高高
            p.resetBasePositionAndOrientation(r2d2, [-17,-9.5, 0.35], [0, 0, 0, 1])
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

        # 行人位置更新
        pos, _ = p.getBasePositionAndOrientation(humanoid)
        if move_direction != 0:
            dir_x = [np.cos(current_yaw), np.sin(current_yaw), 0]
            move_speed = 0.04 * move_direction
            last_pos = [pos[0] + dir_x[0]*move_speed, pos[1] + dir_x[1]*move_speed, pos[2]]
        else:
            last_pos = list(pos)

        # 重設行人位置與方向
        stand_orientation = p.getQuaternionFromEuler([0, 0, current_yaw])
        p.resetBasePositionAndOrientation(humanoid, last_pos, stand_orientation)

        # <<< 移植部分：根據自動駕駛狀態切換控制邏輯 >>>
        if autodrive_enabled and ai_model is not None:
            # --- AI 控制 ---
            if len(image_history) == SEQUENCE_LENGTH:
                lwheel_value, rwheel_value = predict_speeds(ai_model, list(image_history), ai_transform, device)
            else:
                lwheel_value, rwheel_value = 0, 0
            # --- 避障邏輯 ---
            obstacle_detected = False
            avoid_direction = 0  # -1左避障, 1右避障, 0不避障
            retreat = False
            retreat_strength = 0
            closest_obstacle = None
            min_distance = 99999
            # 只考慮信心分數>0.5的障礙物
            if yolo_results:
                for result in yolo_results:
                    boxes = result.boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf > 0.5:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            box_w = x2 - x1
                            box_h = y2 - y1
                            box_center_x = (x1 + x2) // 2
                            # 假設box寬度>200視為足夠近
                            if box_w > 200:
                                obstacle_detected = True
                                # 距離越近，box_w越大
                                if box_w < min_distance:
                                    min_distance = box_w
                                    closest_obstacle = box_center_x
            if obstacle_detected and closest_obstacle is not None:
                # 左側: center_x < width//3, 右側: center_x > width*2//3, 中間: 其他
                if closest_obstacle < width//3:
                    avoid_direction = 1  # 障礙物在左，向右開
                elif closest_obstacle > width*2//3:
                    avoid_direction = -1 # 障礙物在右，向左開
                else:
                    avoid_direction = 0  # 障礙物在中間
                    retreat = True
                    # 根據距離決定後退強度
                    retreat_strength = min(30, int((min_distance-200)*0.5))

            if obstacle_detected:
                # 避障時，優先調整方向
                side_value = avoid_direction
                if retreat:
                    wheel_value = -retreat_strength  # 後退
                else:
                    wheel_value = forward_speed      # 正常前進
                rwheel_value = wheel_value * (1 - side_value * d)
                lwheel_value = wheel_value * (1 + side_value * d)
            else:
                side_value, wheel_value = 0, 0
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
        img = np.reshape(np.array(rgb_img, dtype=np.uint8), (height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_history.append(pil_img)

        # --- YOLO 物件偵測並繪製 ---
        # 取得YOLO偵測結果（用PIL格式）
        yolo_results = YOLO('yolo11n.pt')(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if yolo_results:
            for result in yolo_results:
                boxes = result.boxes
                names = result.names if hasattr(result, 'names') else None
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = names[cls_id] if names else str(cls_id)
                    # 只在信心分數大於0.5時才顯示YOLO偵測框與標籤
                    if conf > 0.5:
                        # 畫框
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                        # 標籤文字
                        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # HUD
        hud_text = f"XYZ: ({r2d2_pos[0]:.3f}, {r2d2_pos[1]:.3f}, {r2d2_pos[2]:.3f})"
        cv2.putText(img, hud_text, (290, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        for joint in [0, 1, 2, 3]:
            joint_state = p.getJointState(r2d2, joint)
            angular_velocity = joint_state[1]
            cv2.putText(img, f"Wheel {joint}: {angular_velocity:.2f} rad/s",
                        (10, 20 + joint * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)
        # <<< 移植部分：在HUD顯示自動駕駛狀態 >>>
        mode_text = "Auto" if autodrive_enabled else "Manual"
        cv2.putText(img, f"Mode: {mode_text}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        cv2.imshow("Car Camera", img)
        if cv2.waitKey(1) == 27:
            break
        p.stepSimulation()
        time.sleep(0.01)
finally:
    cv2.destroyAllWindows()