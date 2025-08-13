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
    print('按5 6 7 8 9 0傳送 ')
    print('按z重設車子速度','按r切換錄影模式','按a切換自動駕駛')
    print('按t在終端輸入座標傳送到指定位置')
    print('退出前記得按r結束錄影，在按esc退出，不然會沒有log'+'\n')

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
# 增加 folder_path 參數，避免依賴全域變數
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
# 增加 folder_path 參數
def save_csv_log(folder_path):
    df = pd.DataFrame(data_log)
    df.to_csv(os.path.join(folder_path, "log.csv"), index=False)
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
    cone_urdf_path = r"D:\Code\3Dmodel\cone.urdf"
    # <<< 關鍵修正 >>>
    # 在嘗試載入之前，先檢查檔案是否存在。
    if not os.path.exists(cone_urdf_path):
        # 如果檔案不存在，列印明確的錯誤訊息並停止。
        print(f"[ERROR] 找不到 URDF 檔案，請檢查路徑是否正確：{cone_urdf_path}")
        print("[ERROR] 載入三角錐失敗，請確認 D:\Code\3Dmodel 資料夾是否存在。")
        return []
    else:
        # 如果檔案存在，才執行載入
        print(f"[INFO] 找到 URDF 檔案，準備載入模型：{cone_urdf_path}")

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
    tree_urdf_path = r"D:\Code\3Dmodel\tree.urdf"

    if not os.path.exists(tree_urdf_path):
        print(f"[ERROR] 找不到 URDF 檔案，請檢查路徑是否正確：{tree_urdf_path}")
        return []
    
    print(f"[INFO] 找到 URDF 檔案，準備載入樹木模型：{tree_urdf_path}")

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
                print(f"[INFO] 建立新資料夾於: {path}")
            return path
        i += 1
# ---------------------------
# PyBullet 環境建置
# ---------------------------
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF(r"D:\Code\3Dmodel\test.urdf", basePosition=[0,0,-0.07])
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
humanoid = p.loadURDF(r"D:\Code\3Dmodel\man.urdf", humanoidStartPos, humanoidStartOrientation)
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
r2d2 = p.loadURDF(r"D:\Code\3Dmodel\front_car.urdf", r2d2StartPos, r2d2StartOrientation)
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

        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            recording = not recording
            print(f"[INFO] 模仿學習資料記錄 {'啟動' if recording else '暫停'}")
            
            # <<< 修改 >>> 核心邏輯：當開始錄製且是第一次按下時，才建立資料夾
            if recording and first_record_press:
                folder_path = setup_recording_folders()
                first_record_press = False # 更新旗標，確保不再重複建立

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

        # Vehicle control
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
        #lane_offset = get_lane_offset_by_opencv(img, width)

        # Speed
        linear_velocity, _ = p.getBaseVelocity(r2d2)
        speed_vec = np.array(linear_velocity)
        forward_vector = np.array([camera_forward[0], camera_forward[1], camera_forward[2]])
        speed_signed = np.dot(speed_vec, forward_vector)

        # HUD
        hud_text = f"XYZ: ({r2d2_pos[0]:.3f}, {r2d2_pos[1]:.3f}, {r2d2_pos[2]:.3f})"
        cv2.putText(img, hud_text, (290, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        for joint in [0, 1, 2, 3]:
            joint_state = p.getJointState(r2d2, joint)
            angular_velocity = joint_state[1]
            cv2.putText(img, f"Wheel {joint}: {angular_velocity:.2f} rad/s",
                        (10, 20 + joint * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)
        cv2.putText(img, f"Car Speed: {speed_signed:.2f} m/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 200), 2)
        
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