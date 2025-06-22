import pybullet as p
from pybullet_utils import gazebo_world_parser
import pybullet_data
import cv2
import time
import random
import numpy as np
import datetime,os

SAVE_IMG = True
wheel_speeds_log = []

#定義斑馬線函數
def create_zebra_crossing(start_pos=[0, 0, 0.05], num_lines=6, spacing=0.3, line_size=[2, 0.2, 0.01]):
    for i in range(num_lines):
        basePosition = [start_pos[0], start_pos[1] + i * spacing, start_pos[2]]
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[line_size[0]/2, line_size[1]/2, line_size[2]/2])
        visBoxId = p.createVisualShape(p.GEOM_BOX, halfExtents=[line_size[0]/2, line_size[1]/2, line_size[2]/2], rgbaColor=[1,1,1,1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, baseVisualShapeIndex=visBoxId, basePosition=basePosition)

# 連接物理引擎
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 載入世界檔案
gazebo_world_parser.parseWorld(p, filepath="worlds/new.world")

# 設定重力
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)

# 呼叫斑馬線
create_zebra_crossing(start_pos=[5, 13.8, 0.0965], num_lines=9, spacing=0.3125)


# 載入行人
humanoidStartPos = [5, 13.3,1]
humanoidStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
humanoid = p.loadURDF('straight.urdf', humanoidStartPos, humanoidStartOrientation)

# 加上支撐桿，避免倒下
cid = p.createConstraint(
    parentBodyUniqueId=humanoid,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_POINT2POINT,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[humanoidStartPos[0], humanoidStartPos[1], 0.5]
)
p.changeConstraint(cid, maxForce=50)

# 行人行走參數
current_yaw = np.pi/2
move_direction = 0
is_forward_pressed = False
is_backward_pressed = False
last_pos, _ = p.getBasePositionAndOrientation(humanoid)

# 載入車輛
r2d2StartPos = [2, 14.4, 2]
r2d2StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
r2d2 = p.loadURDF('real_car.urdf', r2d2StartPos, r2d2StartOrientation)
numJoints = p.getNumJoints(r2d2)

# 車子控制參數
d = 0.75
forward_speed = 20
steering_angle = 0.0
max_steering_angle = 0.5  # 約28度，可自行調整
steering_step = 0.03      # 每次按鍵改變的角度
# 鏡頭控制參數
pitch = p.addUserDebugParameter('camerapitch', 0, 360, 225)
yaw = p.addUserDebugParameter('camerayaw', 0, 360, 90)
distance = p.addUserDebugParameter('cameradistance', 0, 6, 2)

# 相機設定
width, height = 640, 480
fov, aspect, near, far = 60, width/height, 0.1, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

#檔案設定
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
        if SAVE_IMG:
            os.makedirs(os.path.join(folder_path, 'color'))
            os.makedirs(os.path.join(folder_path, 'deep'))
            os.makedirs(os.path.join(folder_path, 'segmentation'))
        break
    i += 1

try:
    while True:
        # 讀取鍵盤事件
        keys = p.getKeyboardEvents()

        if ord('o') in keys and keys[ord('o')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [0, 0, 0.5], [0, 0, 0, 1])

        if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(r2d2, [5, 14.4, 0.5], [0, 0, 0, 1])
    
        # 更新行人移動狀態
        if ord('i') in keys:
            if keys[ord('i')] & p.KEY_IS_DOWN:
                is_forward_pressed = True
            if keys[ord('i')] & p.KEY_WAS_RELEASED:
                is_forward_pressed = False

        if ord('k') in keys:
            if keys[ord('k')] & p.KEY_IS_DOWN:
                is_backward_pressed = True
            if keys[ord('k')] & p.KEY_WAS_RELEASED:
                is_backward_pressed = False

        if is_forward_pressed:
            move_direction = 1
        elif is_backward_pressed:
            move_direction = -1
        else:
            move_direction = 0

        # 更新行人朝向（左右轉）
        if ord('j') in keys and keys[ord('j')] & p.KEY_IS_DOWN:
            current_yaw += 0.05
        if ord('l') in keys and keys[ord('l')] & p.KEY_IS_DOWN:
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

        # 維持人物站直（設定腿部關節位置）
        for joint_index in range(4):
            p.setJointMotorControl2(humanoid, jointIndex=joint_index, controlMode=p.POSITION_CONTROL, targetPosition=0)

        # 車子移動控制
        wheel_velocity = 0
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            wheel_velocity = forward_speed
        elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            wheel_velocity = -forward_speed

    # 方向盤控制
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            steering_angle += steering_step
            steering_angle = min(steering_angle, max_steering_angle)
        elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            steering_angle -= steering_step
            steering_angle = max(steering_angle, -max_steering_angle)
        else:
            # 自動回正（可選）
            steering_angle *= 0.9

        # 設定前輪轉向角
        p.setJointMotorControl2(r2d2, 0, p.POSITION_CONTROL, targetPosition=steering_angle, force=10)
        p.setJointMotorControl2(r2d2, 1, p.POSITION_CONTROL, targetPosition=steering_angle, force=10)
        # 設定後輪速度
        p.setJointMotorControl2(r2d2, 2, p.VELOCITY_CONTROL, targetVelocity=wheel_velocity, force=100)
        p.setJointMotorControl2(r2d2, 3, p.VELOCITY_CONTROL, targetVelocity=wheel_velocity, force=100)

        # 更新相機位置（上帝視角）
        r2d2_pos, _ = p.getBasePositionAndOrientation(r2d2)
        p.resetDebugVisualizerCamera(
            cameraDistance=p.readUserDebugParameter(distance),
            cameraYaw=p.readUserDebugParameter(yaw),
            cameraPitch=p.readUserDebugParameter(pitch),
            cameraTargetPosition=r2d2_pos
        )

        # 更新車載視角畫面
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
        #rgb img
        img = np.reshape(np.array(rgb_img, dtype=np.uint8), (height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        #deep img
        depth_real = (far * near) / (far - (far - near) * depth_buffer)
        depth_mm = (depth_real * 1000).astype(np.uint16)
        #segmentation img
        seg_mask = np.reshape(img_arr[4], (height, width))
        unique_ids = np.unique(seg_mask)
        id_to_color = {obj_id: [random.randint(0,255) for _ in range(3)] for obj_id in unique_ids}
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for obj_id, color in id_to_color.items():
            color_mask[seg_mask == obj_id] = color

        #save img file
        #if SAVE_IMG:
            #file_name = pic_num
            #cv2.imwrite(f'./{folder_path}/color/{pic_num}.png', img)
            #cv2.imwrite(f'./{folder_path}/deep/{pic_num}.png', depth_mm)
            #cv2.imwrite(f'./{folder_path}/segmentation/{pic_num}.png', color_mask)
            #pic_num += 1


        # 在影像上標示HUD文字
        hud_text = f"XYZ: ({r2d2_pos[0]:.2f}, {r2d2_pos[1]:.2f}, {r2d2_pos[2]:.2f}) | Ground Z: 0.00"
        cv2.putText(img, hud_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # 顯示四個輪子的角速度
        wheel_speeds = []
        for joint in [0, 1, 2, 3]:
            joint_state = p.getJointState(r2d2, joint)
            wheel_speeds.append(joint_state[1])  # jointState[1] 是 angular velocity

        for idx, speed in enumerate(wheel_speeds):
            wheel_text = f"Wheel {idx}: {speed:.2f} rad/s"
            cv2.putText(img, wheel_text, (10, 60 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2, cv2.LINE_AA)

        wheel_speeds_log.append([wheel_speeds[0],wheel_speeds[1]])

        # 計算整台車的有向速度（含正負）
        linear_velocity, _ = p.getBaseVelocity(r2d2)
        speed_vec = np.array(linear_velocity)
        _, orn = p.getBasePositionAndOrientation(r2d2)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward_vector = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        speed_signed = np.dot(speed_vec, forward_vector)

        # 顯示整車速度（有正負）
        cv2.putText(img, f"Car Speed: {speed_signed:.2f} m/s", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2, cv2.LINE_AA)

        # 顯示相機畫面
        cv2.imshow("Car Camera", img)

        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break

        # 推進模擬
        p.stepSimulation()
        time.sleep(0.01)

finally:
    # 模擬結束後儲存為 .npy
    #velocity_array = np.array(wheel_speeds_log)
    #npy_path = os.path.join(folder_path, 'velocity_log.npy')
    #np.save(npy_path, velocity_array)

    # 關閉OpenCV視窗
    cv2.destroyAllWindows()