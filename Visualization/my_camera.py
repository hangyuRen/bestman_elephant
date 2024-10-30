import numpy as np
import pyrealsense2 as rs
import cv2

class Camera():
    def __init__(self,width=1280, height=720, fps=30):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.scale = None
        self.pipeline = None
        self.camera_intrinsic = np.loadtxt('./Visualization/cam_pose/camera_intrinsic.txt', delimiter=' ')
        self.T_Cam2Robot_arm1 = np.loadtxt('./Visualization/cam_pose/T_Cam2Robot_arm1.txt', delimiter=' ')
        self.T_Cam2Robot_arm2 = np.loadtxt('./Visualization/cam_pose/T_Cam2Robot_arm2.txt', delimiter=' ')
        # print(self.camera_intrinsic.shape, self.T_Cam2Robot_arm1, self.T_Cam2Robot_arm2)
        self.fx = self.camera_intrinsic[0, 0]
        self.fy = self.camera_intrinsic[1, 1]
        self.cx = self.camera_intrinsic[0, 2]
        self.cy = self.camera_intrinsic[1, 2]
        # print(self.fx, self.fy, self.cx, self.cy)
        self.connect()


    def connect(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        # Start streaming
        cfg = self.pipeline.start(config)

        # # Determine intrinsics
        # rgb_profile = cfg.get_stream(rs.stream.color)
        # self.intrinsics = self.get_intrinsics(rgb_profile)
        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print("camera depth scale:",self.scale)
        print("D415 have connected ...")


    def get_data(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        # align
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # no align
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.float32)
        # depth_image *= self.scale
        depth_image = np.expand_dims(depth_image, axis=2)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image


    def plot_image(self):
        color_image,depth_image = self.get_data()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imwrite('color_image.png', color_image)
        cv2.waitKey(5000)

    def camera_stop(self):
        self.pipeline.stop()

    def get_base_coordinate(self, x, y, z, T_mtx):
        X = (x - self.cx) * z / self.fx
        Y = (y - self.cy) * z / self.fy
        Z = z
        P_camera = np.array([X, Y, Z, 1.0])
        point_base = np.dot(T_mtx, P_camera)
        return point_base[0], point_base[1], point_base[2]
    
if __name__=='__main__':
    camera = Camera()