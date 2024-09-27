
import pyrealsense2 as rs
import numpy as np
import cv2
from toch_point_caliration import get_base_coordinate_arm1, get_base_coordinate_arm2
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append("..")
from RoboticsToolBox.Bestman_Elephant import Bestman_Real_Elephant

# 实例化Bestman_Real_Elephant对象
bestman = Bestman_Real_Elephant("172.20.10.8", 5001)
bestman2 = Bestman_Real_Elephant("172.20.10.7", 5001)

# 机械臂使能，运行一次即可
bestman.state_on()
bestman2.state_on()

# 初始化 RealSense 流程
pipeline = rs.pipeline()
config = rs.config()

# 启用深度流和颜色流
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 开始流
pipeline.start(config)

# 创建对齐对象 (将深度图像对齐到颜色图像)
align_to = rs.stream.color
align = rs.align(align_to)

# 存储点击的坐标
click_points = []

# 操作类型
ARM1_GRAB = 0
ARM1_RELEASE = 1
ARM2_GRAB = 2
ARM2_RELEASE = 3

operation = ARM1_GRAB

# 创建线程池
executor = ThreadPoolExecutor(max_workers=2)

# 移动机械臂到指定位置抓取
def move_arm(arm:Bestman_Real_Elephant, x_base, y_base, z_base, operation):
    print(f"{arm}-operation:{operation}-moveing to coordinates:({x_base, y_base, z_base})")
    if operation == ARM1_GRAB or operation == ARM2_GRAB:
        arm._set_arm_coords([x_base, y_base, 230, 175, 0, 120], speed=800)
        arm.open_gripper()
        arm._set_arm_coords([x_base, y_base, 165, 175, 0, 120], speed=800)
        arm.close_gripper()
        arm._set_arm_coords([x_base, y_base, 230, 175, 0, 120], speed=800)
    else:
        arm._set_arm_coords([x_base, y_base, 230, 175, 0, 120], speed=800)
        arm._set_arm_coords([x_base, y_base, 180, 175, 0, 120], speed=800)
        arm.open_gripper()
        arm._set_arm_coords([x_base, y_base, 230, 175, 0, 120], speed=800)
        arm.close_gripper()
        # arm.set_arm_joint_values([-90, -120.0, 120.0, -90.0, -90.0, -0.0], speed=800)
        

# 鼠标回调函数
def get_mouse_click(event, x, y, flags, param):
    global click_points, operation
    if event == cv2.EVENT_LBUTTONDOWN:
        aligned_depth_frame, color_image = param
        
        # 获取点击位置的深度值 z
        z = aligned_depth_frame.get_distance(x, y) * 1000  # 将 z 值从米转换为毫米
        
        print(f"点击位置 (u, v): ({x}, {y}), 深度值 z: {z} 毫米")

        # 图像坐标系转换到机械臂坐标系
        if operation == ARM1_GRAB or operation == ARM1_RELEASE:
            x_base, y_base, z_base = get_base_coordinate_arm1(x, y, z)
        else:
            x_base, y_base, z_base = get_base_coordinate_arm2(x, y, z)

        click_points.append((x_base, y_base, z_base))

        if len(click_points) == 1:
            if operation == ARM1_GRAB:
                executor.submit(move_arm, bestman, *click_points[0], operation)
                operation = ARM2_GRAB
            elif operation == ARM2_GRAB:
                executor.submit(move_arm, bestman2, *click_points[0], operation)
                operation = ARM1_RELEASE
            elif operation == ARM1_RELEASE:
                executor.submit(move_arm, bestman, *click_points[0], operation)
                operation = ARM2_RELEASE
            elif operation == ARM2_RELEASE:
                executor.submit(move_arm, bestman2, *click_points[0], operation)
                operation = ARM1_GRAB
            click_points.clear()
try:
    while True:
        # 获取一帧数据
        frames = pipeline.wait_for_frames()

        # 对齐深度帧到颜色帧
        aligned_frames = align.process(frames)

        # 获取对齐后的深度帧和颜色帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 将颜色帧转换为 NumPy 数组
        color_image = np.asanyarray(color_frame.get_data())

        # 显示颜色图像
        cv2.imshow('Aligned RGB Image', color_image)

        # 设置鼠标回调函数
        cv2.setMouseCallback('Aligned RGB Image', get_mouse_click, (aligned_depth_frame, color_image))

        # 等待按键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()