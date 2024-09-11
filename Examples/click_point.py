
import pyrealsense2 as rs
import numpy as np
import cv2
from toch_point_caliration import move_arm, get_base_coordinate

import sys
sys.path.append("..")
from RoboticsToolBox.Bestman_Elephant import Bestman_Real_Elephant

# 实例化Bestman_Real_Elephant对象
bestman = Bestman_Real_Elephant("192.168.43.243", 5001)

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

# 判断此次鼠标点击是抓取操作还是释放操作 0：抓取 1：释放
flag = 0

# 鼠标回调函数
def get_mouse_click(event, x, y, flags, param):
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        aligned_depth_frame, color_image = param
        
        # 获取点击位置的深度值 z
        z = aligned_depth_frame.get_distance(x, y) * 1000  # 将 z 值从米转换为毫米
        
        print(f"点击位置 (u, v): ({x}, {y}), 深度值 z: {z} 毫米")

        # 图像坐标系转换到机械臂坐标系
        x_base, y_base, z_base = get_base_coordinate(x, y, z)

        # 机械臂夹爪抓取图像中指定位置的物品
        if flag == 0:
            bestman.set_arm_coords([x_base, y_base, 230, 175, 0, 120], speed=800)
            bestman.open_gripper()
            bestman.set_arm_coords([x_base, y_base, 165, 175, 0, 120], speed=800)
            bestman.close_gripper()
            bestman.set_arm_coords([x_base, y_base, 230, 175, 0, 120], speed=800)
            flag = 1
        # 机械臂夹爪在图像中指定位置释放物品并回到标准姿态
        else:
            bestman.set_arm_coords([x_base, y_base, 230, 175, 0, 120], speed=800)
            bestman.set_arm_coords([x_base, y_base, 180, 175, 0, 120], speed=800)
            bestman.open_gripper()
            bestman.close_gripper()
            bestman.set_arm_joint_values([-90, -120.0, 120.0, -90.0, -90.0, -0.0], speed=800)
            flag = 0

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