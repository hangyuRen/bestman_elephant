'''
Author: hyuRen
Date: 2024-10-09 09:39:00
LastEditors: hyuRen
LastEditTime: 2024-10-17 18:02:03
'''
import pyrealsense2 as rs
import numpy as np
import cv2
from Examples.toch_point_caliration import get_base_coordinate_arm1, get_base_coordinate_arm2
from multiprocessing import Pool, Process
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append("..")
from RoboticsToolBox.Bestman_Elephant import Bestman_Real_Elephant
from gdino.gdino import GDINO, get_box_center
from Visualization.my_camera import Camera
from PIL import Image
import torch

# 实例化Bestman_Real_Elephant对象
bestman = Bestman_Real_Elephant("172.20.10.8", 5001)
# bestman2 = Bestman_Real_Elephant("172.20.10.7", 5001)

# # 机械臂使能，执行一次即可
# bestman.state_on()
# bestman2.state_on()


# 移动机械臂到指定位置抓取
def move_arm(x_base, y_base, z_base):
    print(f"moveing to coordinates:({x_base, y_base, z_base})")
    bestman.set_arm_coords([x_base, y_base, 230, 175, 0, 120], speed=800)
    # bestman.open_gripper()
    # bestman.set_single_coord('Z', -75, 800)
    # bestman.close_gripper()
    # bestman.set_single_coord('Z', 75, 800)
    

if __name__ == '__main__':
    model = GDINO(model_dir='./gdino/gdino_model')
    camera = Camera()
    try:
        while True:
            color_image, depth_image = camera.get_data()
            # 显示颜色图像
            cv2.imshow('Aligned RGB Image', color_image)

            text_prompt = str(input('text prompt = '))
            out = model.predict(
                [Image.fromarray(color_image).convert('RGB')],
                [text_prompt],
                0.3,
                0.25,
            )
            scores = out[0]['scores']
            highest_score_index = torch.argmax(scores).item() 
            center_x, center_y = get_box_center(box_list = out[0]['boxes'][highest_score_index].tolist())
            x_base, y_base, z_base = camera.get_base_coordinate(center_x, center_y, 0, camera.T_Cam2Robot_arm1)
            move_arm(x_base, y_base, z_base)
            # 等待按键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.camera_stop()
        cv2.destroyAllWindows()