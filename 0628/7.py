import pybullet as p
from pybullet_utils import gazebo_world_parser
import pybullet_data
import cv2
import time
import random
import numpy as np
import datetime, os
import pandas as pd

data_log = []
SAVE_IMG = True

actual_lwheel_value = 0
actual_rwheel_value = 0
alpha = 0.1  # 越小回復越慢

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
def log_data(pic_num, img, side_value, wheel_value, lwheel_value, rwheel_value, speed_signed, seg_mask, width, height, lane_offset):
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
humanoid = p.loadURDF('straight_scaled_0.5x.urdf', humanoidStartPos, humanoidStartOrientation)
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
        for joint in [0, 1, 2, 3]:
            v = lwheel_value if joint % 2 == 0 else rwheel_value
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
        lane_offset = get_lane_offset_by_opencv(img, width)

        # Speed
        linear_velocity, _ = p.getBaseVelocity(r2d2)
        speed_vec = np.array(linear_velocity)
        forward_vector = np.array([camera_forward[0], camera_forward[1], camera_forward[2]])
        speed_signed = np.dot(speed_vec, forward_vector)

        # HUD
        cv2.putText(img, f"Car Speed: {speed_signed:.2f} m/s", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
        cv2.putText(img, f"Lane Offset: {lane_offset:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)
        # 顯示四個輪子的角速度
        for joint in [0, 1, 2, 3]:
            joint_state = p.getJointState(r2d2, joint)
            angular_velocity = joint_state[1]  # jointState[1] 是角速度
            cv2.putText(img, f"Wheel {joint}: {angular_velocity:.2f} rad/s",
                        (10, 25 + joint * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)

        if recording:
            log_data(pic_num, img, side_value, wheel_value, lwheel_value, rwheel_value, speed_signed, seg_mask, width, height, lane_offset)
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
    cv2.destroyAllWindows()
