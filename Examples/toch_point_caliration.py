""" 
# @Author: Youbin Yao 
# @Date: 2024-09-01 13:24:30
# @Last Modified by:   Youbin Yao 
# @Last Modified time: 2024-09-01 13:24:30  
""" 
import sys
import os
sys.path.append(os.getcwd())
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'RoboticsToolBox'))
sys.path.append(os.path.join(parent_dir, 'Visualization'))
import numpy as np
import cv2
import json
from RoboticsToolBox import Bestman_Real_Elephant
from Visualization import calibration_eye_to_hand, Transform
# from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
# from ikpy.chain import Chains
# from ikpy.link import OriginLink, URDFLink

# 读取 JSON 文件中的相机参数
json_file_path = r'C:/Users/86153/Desktop/BestMan_Elephant-bestman/camera_params.json'
with open(json_file_path, 'r') as file:
    camera_params = json.load(file)

# 提取相机参数
mtx = np.array(camera_params['mtx']).reshape((3, 3))
dist = np.array(camera_params['dist'])
rvecs = np.array(camera_params['rvecs'])
tvecs = np.array(camera_params['tvecs'])

# 假设图像中目标点的像素坐标 (u, v) 和深度 Z
# u, v= 655, 292 # 假设中心点
# Z = 764 # 深度
 
# 提取内参矩阵(中的参数
mtx = np.array(
[[644.7208252,    0.,         631.20788574],
 [  0. ,        643.23901367, 372.19293213],
 [  0.,           0.,           1.        ]])
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]
print(mtx)
# 将图像坐标 (u, v, z) 转换为相机坐标 (X, Y, Z)
# X = (u - cx) * Z / fx
# Y = (v - cy) * Z / fy
# Z = Z
# P_camera = np.array([X, Y, Z, 1.0])
# print(P_camera)

# arm 1
# T_Cam2Robot =np.array(
# [[-0.996940745126952,	0.0763025946083072,	-0.0169429856211645,	220.504604304431],
# [0.0745146425823138,	0.993269739676621,	0.0886723873794279,	-291.621328965515,],
# [0.0235948881444421,	0.0871386154283942,	-0.995916734950605,	910.918427688599,],
# [0,	0,	0,	1,]]) # 有误差 x_base -= 20 y_base += 5
# # arm 1
T_Cam2Robot_arm1 =np.array(
[[-9.99810248e-01,  5.95333441e-03, -1.85479259e-02,  2.41222153e+02],
 [ 4.60860048e-03,  9.97414322e-01,  7.17177231e-02, -2.68426413e+02],
 [ 1.89269265e-02,  7.16186346e-02, -9.97252497e-01,  8.90397554e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
 
# arm 2
T_Cam2Robot_arm2 =np.array(
[[ 9.99517264e-01, -3.00053157e-02, -8.05725336e-03,  2.32466261e+02],
 [-3.03055063e-02, -9.98733683e-01, -4.01572709e-02, -3.05379448e+02],
 [-6.84211874e-03,  4.03820647e-02, -9.99160885e-01,  8.85636510e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
# point_base = np.dot(T_Cam2Robot, P_camera)
# x_base, y_base, z_base = point_base[0], point_base[1], point_base[2]

# print(f"目标点在机械臂基座坐标系下的位置: ({x_base}, {y_base}, {z_base})")
def get_base_coordinate_arm1(u, v, z):
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    P_camera = np.array([X, Y, Z, 1.0])
    point_base = np.dot(T_Cam2Robot_arm1, P_camera)
    return point_base[0], point_base[1], point_base[2]


def get_base_coordinate_arm2(u, v, z):
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    P_camera = np.array([X, Y, Z, 1.0])
    point_base = np.dot(T_Cam2Robot_arm2, P_camera)
    return point_base[0], point_base[1], point_base[2]


# bestman = Bestman_Real_Elephant("192.168.43.243", 5001)

# def move_arm(x_base, y_base, z_base, bestman):
    # bestman.get_current_cartesian()
    # bestman.get_current_joint_values()
    # bestman.set_arm_coords([x_base, y_base, 200, 175, 0, 120], speed=800)
    # bestman.set_arm_coords([x_base, y_base, 170, 175, 0, 120], speed=800)
    # bestman.get_current_cartesian()
    # bestman.get_current_joint_values()



# bestman = Bestman_Real_Elephant("192.168.43.243", 5001)
# bestman.state_on()
# bestman.get_current_cartesian()
# bestman.get_current_joint_values()
# 定义垂直向下的欧拉角
# bestman.set_arm_joint_values([0.0, -120.0, 120.0, -90.0, -90.0, -0.0],speed=500)

# bestman.set_arm_coords([x_base,y_base,180, 175,0,120],speed=800)
# bestman.get_current_cartesian()
# bestman.get_current_joint_values() 
# 将基座坐标系中的点转换回相机坐标系
# P_cam = np.dot(T_cam2base, [0,0,0,1])
# print(P_cam) 
#结果验证，原则上来说，每次结果相差较小

# for i in range(len(tvecs)):

#     RT_end_to_base=np.column_stack((R_gripper2base_list[i],t_gripper2base_list[i]))
#     # print('RT_end_to_base', RT_end_to_base)
#     RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    

#     RT_chess_to_cam=np.column_stack((R_target2cam_list[i],t_target2cam_list[i]))
#     # print('RT_chess_to_cam', RT_chess_to_cam)
#     RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
#     # print(RT_chess_to_cam)

#     RT_cam_to_end=np.column_stack((R_cam2base,t_cam2base))
#     RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
#     # print(RT_cam_to_end)

#     RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam#即为固定的棋盘格相对于机器人基坐标系位姿
#     RT_chess_to_base=np.linalg.inv(RT_chess_to_base)
#     print('第',i,'次')
#     print(RT_chess_to_base[:3,:])
#     print('')
 