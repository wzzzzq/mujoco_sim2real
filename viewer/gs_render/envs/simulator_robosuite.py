import os
import time
import traceback
from abc import abstractmethod

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import glfw
import OpenGL.GL as gl
import ctypes
import cv2
import sys
import rospy
from nav_msgs.msg import Odometry, Path

from mobile_ai import MOBILE_AI_ASSERT_DIR
from mobile_ai.utils import BaseConfig
from mobile_ai.gaussian_renderer import GSRenderer
from cam_msg.msg import CameraPose
from std_msgs.msg import Bool, Float64MultiArray
from termcolor import cprint
# from mobile_ai.utils.base_config import BaseConfig
import mujoco
import h5py

try:
    from mobile_ai.gaussian_renderer.util_gaussian import multiple_quaternion_vector3d, multiple_quaternions
    MOBILE_AI_GAUSSIAN_RENDERER = True

except ImportError:
    traceback.print_exc()
    print("Warning: gaussian_splatting renderer not found. Please install the required packages to use it.")
    MOBILE_AI_GAUSSIAN_RENDERER = False


def setRenderOptions(options):
    # 启用透明渲染
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    # 启用接触力显示
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # 设置参考坐标系为物体坐标系
    options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    pass

class SimulatorBaseRoboSuite:
    running = True # 仿真运行标志
    obs = None     # 观测数据

    cam_id = -1  # -1表示自由视角
    last_cam_id = -1
    render_cnt = 0
    camera_names = []
    camera_pose_changed = False
    camera_rmat = np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ])

    mouse_pressed = {
        'left': False,
        'right': False,
        'middle': False
    }
    mouse_pos = {
        'x': 0,
        'y': 0
    }

    options = mujoco.MjvOption()

    def __init__(self, config:BaseConfig):
        
        self.config = config
        self.decimation = self.config.decimation
        self.delta_t = 0.1
        self.render_fps = self.config.render_set["fps"]

        if self.config.enable_render:
            self.config.use_gaussian_renderer = True
            if self.config.use_gaussian_renderer:
                self.gs_renderer = GSRenderer(self.config.gs_model_dict, 640, 480)
                # self.gs_renderer = GSRenderer(self.config.gs_model_dict, self.config.render_set["width"], self.config.render_set["height"])
                self.last_cam_id = self.cam_id
                self.show_gaussian_img = True
                self.gs_renderer.set_camera_fovy(45.0 * np.pi / 180.)

        self.window = None
        self.glfw_initialized = False

        # ros 
        rospy.init_node('gs_render')
        # 订阅定位消息
        self.cmd_vel_sub = rospy.Subscriber("/localization", Odometry, self.localization_callback)
        self.localization = None
        self.localization_cam = None

        self.pose_sub = rospy.Subscriber("/camera_pose", CameraPose, self.camera_pose_callback)

        ## INIT everything for record data
        
        self.camera_poses = {}
        self.default_pose = {
            "position": np.array([0, 0, 0.95]),  # defalut position
            "orientation": np.array([0, 0, 0, 1])  # defalut orientation
        }
        self.left_joint_pos_sub = rospy.Subscriber('/left_joint_array', Float64MultiArray, self.left_joint_callback)
        self.right_joint_pos_sub = rospy.Subscriber('/right_joint_array', Float64MultiArray, self.right_joint_callback)
        self.is_done_episode_sub = rospy.Subscriber('/is_done_episode', Bool,self.is_done_episode_callback)
        self.left_arm_cur_joint_pos = np.array([0,0,0,0,0,0,0])
        self.right_arm_cur_joint_pos = np.array([0,0,0,0,0,0,0])
        self.received_joint_data = False
        self.record_start_sub = rospy.Subscriber('/start_record', Bool, self.start_record_callback)
        self.start_record = False
        self.record_flag = False
        self.episode_idx = 0
        self.left_joint_ready = False
        self.right_joint_ready = False

        

        camera_names = ["mobilebase0_top_camera", "robot0_wrist_cam_left", "robot0_wrist_cam_right"]
        self.data_dict = {}
        self.data_dict = {
            'observations': {
                'images': {cam_name: [] for cam_name in camera_names},
                'qpos': [],
                'actions': []
            }
        }
        self.image_height = 480
        self.image_width = 640
        # print(f"111{self.config.obs_rgb_cam_id}")
        for cam_id in self.config.obs_rgb_cam_id:          
            self.camera_poses[cam_id] = self.default_pose

        self.finish_record_episode_pub = rospy.Publisher("/finish_record_episode_once", Bool, queue_size=10)
        #TODO change to load from config
        self.camera_intrinsics = {
            0: {
                # "K": np.array([[630.7139/2, 0, 656.1203290131507/2], [0, 634.57349/1.5, 367.10173/1.5], [0, 0, 1]]),
                "K": np.array([[644.0582885742188/2, 0,647.5768432617188/2], [0, 643.2357177734375/1.5, 357.7709655761719/1.5], [0, 0, 1]]),
                "img_height": self.image_height,
                "img_width":self.image_width                
            },
            1: {
                # "K": np.array([[630.7139/2, 0, 656.1203290131507/2], [0, 634.57349/1.5, 367.10173/1.5], [0, 0, 1]]),
                "K": np.array([[644.0582885742188/2, 0,647.5768432617188/2], [0, 643.2357177734375/1.5, 357.7709655761719/1.5], [0, 0, 1]]),
                "img_height": self.image_height,
                "img_width":self.image_width
            },
            2: {
                 # "K": np.array([[630.7139/2, 0, 656.1203290131507/2], [0, 634.57349/1.5, 367.10173/1.5], [0, 0, 1]]),
                "K": np.array([[644.0582885742188/2, 0,647.5768432617188/2], [0, 643.2357177734375/1.5, 357.7709655761719/1.5], [0, 0, 1]]),
                "img_height":  self.image_height,
                "img_width":self.image_width
            },
            3: {
                 # "K": np.array([[630.7139/2, 0, 656.1203290131507/2], [0, 634.57349/1.5, 367.10173/1.5], [0, 0, 1]]),
                "K": np.array([[644.0582885742188/2, 0,647.5768432617188/2], [0, 643.2357177734375/1.5, 357.7709655761719/1.5], [0, 0, 1]]),
                "img_height":  self.image_height,
                "img_width":self.image_width
            },
            4: {
                 # "K": np.array([[630.7139/2, 0, 656.1203290131507/2], [0, 634.57349/1.5, 367.10173/1.5], [0, 0, 1]]),
                "K": np.array([[644.0582885742188/2, 0,647.5768432617188/2], [0, 643.2357177734375/1.5, 357.7709655761719/1.5], [0, 0, 1]]),
                "img_height":  self.image_height,
                "img_width":self.image_width
            },
            5: {
                # "K": np.array([[630.7139/2, 0, 656.1203290131507/2], [0, 634.57349/1.5, 367.10173/1.5], [0, 0, 1]]),
                "K": np.array([[644.0582885742188/2, 0,647.5768432617188/2], [0, 643.2357177734375/1.5, 357.7709655761719/1.5], [0, 0, 1]]),
                "img_height":  self.image_height,
                "img_width":self.image_width
            },
            6: {
                 # "K": np.array([[630.7139/2, 0, 656.1203290131507/2], [0, 634.57349/1.5, 367.10173/1.5], [0, 0, 1]]),
                "K": np.array([[644.0582885742188/2, 0,647.5768432617188/2], [0, 643.2357177734375/1.5, 357.7709655761719/1.5], [0, 0, 1]]),
                "img_height":  self.image_height,
                "img_width":self.image_width
            },
            
        }
        
        # 启动窗口化显示
        if not hasattr(self.config.render_set, "window_title"):
            self.config.render_set["window_title"] = "MOBILE AI Simulator"
        
        if self.config.enable_render and not self.config.headless:
            try:
                if not glfw.init():
                    raise RuntimeError("无法初始化GLFW")
                self.glfw_initialized = True
                
                # 设置OpenGL版本和窗口属性
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
                glfw.window_hint(glfw.VISIBLE, True)
                
                # 创建窗口
                self.window = glfw.create_window(
                    self.config.render_set["width"],
                    self.config.render_set["height"],
                    self.config.render_set.get("window_title", "DISCOVERSE Simulator"),
                    None, None
                )
                
                if not self.window:
                    glfw.terminate()
                    raise RuntimeError("无法创建GLFW窗口")
                
                glfw.make_context_current(self.window)
                glfw.swap_interval(1)

                # 初始化OpenGL设置
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)
                gl.glShadeModel(gl.GL_SMOOTH)
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                
                # 设置回调
                glfw.set_key_callback(self.window, self.on_key)
                glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
                glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
                
            except Exception as e:
                print(f"GLFW初始化失败: {e}")
                if self.glfw_initialized:
                    glfw.terminate()
                self.config.headless = True
                self.window = None

        self.last_render_time = time.time()

    def is_done_episode_callback(self,msg):
        self.record_flag = msg.data
        print(f"is done : {self.record_flag}")

    def left_joint_callback(self, msg):
        self.left_arm_cur_joint_pos = msg.data
        self.left_joint_ready = True

    def right_joint_callback(self, msg):
        self.right_arm_cur_joint_pos = msg.data
        self.right_joint_ready = True

    def start_record_callback(self, msg):
        self.start_record = msg.data
        # print(f"start record :{self.start_record}")

    def localization_callback(self, msg):
        # 更新定位结果
        self.localization = msg

    def camera_pose_callback(self, msg):
        """update camera pose"""
        # print(f"接收到相机位姿: {msg}")
        
        cam_id = msg.camera_id
        if cam_id == 1:
            self.localization_cam = msg

        """update camera pose"""
        position = np.array([msg.camera_pose.position.x, msg.camera_pose.position.y, msg.camera_pose.position.z])
        orientation = np.array([msg.camera_pose.orientation.x, msg.camera_pose.orientation.y,
                                 msg.camera_pose.orientation.z, msg.camera_pose.orientation.w])
        
        if 'top' in msg.camera_name.lower():
            position = np.array([msg.camera_pose.position.x +0.01, msg.camera_pose.position.y+0.04, msg.camera_pose.position.z+0.04])
            rot = Rotation.from_quat(orientation)
            delta_rot = Rotation.from_euler('xz', [-2.8,-1.7], degrees=True)
            rot_new = rot * delta_rot
            orientation = rot_new.as_quat()
        if 'wrist' in msg.camera_name.lower():
            # print("wrist....")
            position = np.array([msg.camera_pose.position.x, msg.camera_pose.position.y , msg.camera_pose.position.z])
            rot = Rotation.from_quat(orientation)  # [x, y, z, w] format
            # 绕相机局部 x 轴逆转 15 度
            delta_rot = Rotation.from_euler('xz', [0.01,-1.5], degrees=True)
            rot_new = rot * delta_rot  # 注意顺序：右乘是绕自身坐标系
            orientation = rot_new.as_quat()  # 回到 [x, y, z, w] 

        self.camera_poses[cam_id] = {
            "position": position,
            "orientation": orientation,
            "name": msg.camera_name
        }



    def render(self):
        self.render_cnt += 1
        # 更新 gs 场景
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            self.update_gs_scene()

        # 遍历每一个相机得到 gs 渲染的 rgb 图像
        self.img_rgb_obs_s = {}
        # for id in self.config.obs_rgb_cam_id:
        #     # print(f"self.config.obs_rgb_cam_id: {self.config.obs_rgb_cam_id}")
        #     img = self._getRgbImg(id)
        #     self.img_rgb_obs_s[id] = img

        if self.cam_id in self.config.obs_rgb_cam_id:
            # print("111")
            # img_vis = self.img_rgb_obs_s[self.cam_id]
            img_vis = self.getRgbImg(self.cam_id)
        else:
            # print("222")
            # print("process here !!! ")
            img_rgb = self.getRgbImg(self.cam_id)
            img_vis = img_rgb

        if not self.config.headless and self.window is not None:
            try:
                if glfw.window_should_close(self.window):
                    self.running = False
                    return
                    
                glfw.make_context_current(self.window)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                
                gl.glViewport(0, 0, self.config.render_set["width"], self.config.render_set["height"])
                
                gl.glRasterPos2i(-1, -1)

                if sys.platform == "darwin":
                    gl.glPixelZoom(2.0, 2.0)
                
                if img_vis is not None:
                    img_vis = img_vis[::-1]
                    img_vis = np.ascontiguousarray(img_vis)
                    gl.glDrawPixels(img_vis.shape[1], img_vis.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_vis.tobytes())
                
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                if self.config.sync:
                    current_time = time.time()
                    wait_time = max(1.0/self.render_fps - (current_time - self.last_render_time), 0)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self.last_render_time = time.time()
                    
            except Exception as e:
                print(f"渲染错误: {e}")

    def _getRgbImg(self, cam_id):
        if cam_id == -1:
            self.gs_renderer.set_camera_fovy(45.0 * np.pi / 180.0)
        if self.last_cam_id != cam_id and cam_id > -1:
            self.gs_renderer.set_camera_fovy(45.0 * np.pi / 180.0)
        self.last_cam_id = cam_id
        trans, quat_wxyz = self.getCameraPose(cam_id)
        self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
        return self.gs_renderer.render()
    

    '''
    rewrite getRgbImg function to use camera pose from ros topic
    '''
    def getRgbImg(self, cam_id, trans=[0,0,0], quat_xyzw=[0,0,0,1]):
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            # cprint(text=f"camera_id:{cam_id}",color='cyan',attrs=['bold'])
            # if cam_id == -1:
            #     self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            #     self.gs_renderer.set_camera_fovy(self.mj_model.vis.global_.fovy * np.pi / 180.0)
            if self.last_cam_id != cam_id and cam_id > -1:
                #### ziheng ji 2025.03.31
                '''
                change to not relay on mujoco.mjv_moveCamera
                '''
                intrinsics = self.camera_intrinsics[cam_id]["K"]
                img_height = self.camera_intrinsics[cam_id]["img_height"]
                img_width = self.camera_intrinsics[cam_id]["img_width"]
                fy = intrinsics[1,1]
                # fovy = 2 * np.arctan(img_height / (2 * fy))
                fovy = 1.14
                # print(f"fovy : {fovy}")
                self.gs_renderer.set_camera_fovy(fovy)
                #### ziheng ji 2025.03.31

            self.last_cam_id = cam_id
            # trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_xyzw)
            # print("09009")
            return self.gs_renderer.render()
        else:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            elif cam_id > -1:
                self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
            else:
                return None
            rgb_img = self.renderer.render()
            return rgb_img
    

    def on_mouse_move(self, window, xpos, ypos):
        if self.cam_id == -1:
            dx = xpos - self.mouse_pos['x']
            dy = ypos - self.mouse_pos['y']
            height = self.config.render_set["height"]
            
            action = None
            if self.mouse_pressed['left']:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif self.mouse_pressed['right']:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif self.mouse_pressed['middle']:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM

            if action is not None:
                self.camera_pose_changed = True
                mujoco.mjv_moveCamera(self.mj_model,  action,  dx/height,  dy/height, self.renderer.scene, self.free_camera)

        self.mouse_pos['x'] = xpos
        self.mouse_pos['y'] = ypos

    def on_mouse_button(self, window, button, action, mods):
        is_pressed = action == glfw.PRESS
        
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed['left'] = is_pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_pressed['right'] = is_pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_pressed['middle'] = is_pressed

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            is_ctrl_pressed = (mods & glfw.MOD_CONTROL)
            
            if is_ctrl_pressed:
                if key == glfw.KEY_G:  # Ctrl + G
                    if self.config.use_gaussian_renderer:
                        self.show_gaussian_img = not self.show_gaussian_img
                        self.gs_renderer.renderer.need_rerender = True
                elif key == glfw.KEY_D:  # Ctrl + D
                    if self.config.use_gaussian_renderer:
                        self.gs_renderer.renderer.need_rerender = True
                    if self.renderer._depth_rendering:
                        self.renderer.disable_depth_rendering()
                    else:
                        self.renderer.enable_depth_rendering()
            else:
                if key == glfw.KEY_H:  # 'h': 显示帮助
                    self.printHelp()
                elif key == glfw.KEY_P:  # 'p': 打印信息
                    self.printMessage()
                elif key == glfw.KEY_R:  # 'r': 重置状态
                    self.reset()
                elif key == glfw.KEY_ESCAPE:  # ESC: 切换到自由视角
                    self.cam_id = -1
                    self.camera_pose_changed = True
                elif key == glfw.KEY_RIGHT_BRACKET:  # ']': 下一个相机
                    if self.mj_model.ncam:
                        self.cam_id += 1
                        self.cam_id = self.cam_id % self.mj_model.ncam
                elif key == glfw.KEY_LEFT_BRACKET:  # '[': 上一个相机
                    if self.mj_model.ncam:
                        self.cam_id += self.mj_model.ncam - 1
                        self.cam_id = self.cam_id % self.mj_model.ncam

    def printHelp(self):
        """打印帮助信息"""
        print("\n=== keyboard not allowed to control ===")
        # print("H: 显示此帮助信息")
        # print("P: 打印当前状态信息")
        # print("R: 重置模拟器状态")
        # print("G: 切换高斯渲染（如果可用）")
        # print("D: 切换深度渲染")
        # print("Ctrl+G: 组合键切换高斯模式")
        # print("Ctrl+D: 组合键切换深度图模式")
        # print("ESC: 切换到自由视角")
        # print("[: 切换到上一个相机")
        # print("]: 切换到下一个相机")
        # print("\n=== 鼠标控制说明 ===")
        # print("左键拖动: 旋转视角")
        # print("右键拖动: 平移视角")
        # print("中键拖动: 缩放视角")
        # print("================\n")

    def printMessage(self):
        """打印当前状态信息"""
        print("\n=== 当前状态 ===")
        print(f"当前相机ID: {self.cam_id}")
        if self.cam_id >= 0:
            print(f"相机名称: {self.camera_names[self.cam_id]}")
        print(f"高斯渲染: {'开启' if self.show_gaussian_img else '关闭'}")
        print(f"深度渲染: {'开启' if self.renderer._depth_rendering else '关闭'}")
        print("==============\n")

    def resetState(self):
        # mujoco.mj_resetData(self.mj_model, self.mj_data)
        # mujoco.mj_forward(self.mj_model, self.mj_data)
        self.camera_pose_changed = True

    def update_gs_scene(self):
        # print("Base class update_gs_scene")

        # self.gs_renderer.set_obj_pose("robot0_piper_left_link2", np.array([0,0,1]), np.array([1,0,0,0]))
        for name in self.config.obj_list + self.config.rb_link_list:
            trans, quat_wxyz = self.getObjPose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        if self.gs_renderer.update_gauss_data:
            
            self.gs_renderer.update_gauss_data = False
            self.gs_renderer.renderer.need_rerender = True
            self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
            self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])

    def getObjPose(self, name):
        try:
            position = self.mj_data.body(name).xpos
            quat = self.mj_data.body(name).xquat
            return position, quat
        except KeyError:
            try:
                position = self.mj_data.geom(name).xpos
                quat = Rotation.from_matrix(self.mj_data.geom(name).xmat.reshape((3,3))).as_quat()[[3,0,1,2]]
                return position, quat
            except KeyError:
                print("Invalid object name: {}".format(name))
                return None, None
    
    def getCameraPose(self, cam_id):
        if self.localization_cam:
            # print(self.localization_cam)
            cur_pos_x = self.localization_cam.camera_pose.position.x
            cur_pos_y = self.localization_cam.camera_pose.position.y
            cur_pos_z = self.localization_cam.camera_pose.position.z

            cur_orientation_x = self.localization_cam.camera_pose.orientation.x
            cur_orientation_y = self.localization_cam.camera_pose.orientation.y
            cur_orientation_z = self.localization_cam.camera_pose.orientation.z
            cur_orientation_w = self.localization_cam.camera_pose.orientation.w

            trans = np.array([cur_pos_x, cur_pos_y, cur_pos_z], dtype=np.float32)
            # 直接赋值四元数 (quaternion, wxyz顺序)
            quat_wxyz = np.array([cur_orientation_w, cur_orientation_x, cur_orientation_y, cur_orientation_z], dtype=np.float32)
        else:
            trans = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # 明确指定float32类型
            # 直接赋值四元数 (quaternion, wxyz顺序)
            quat_wxyz = np.array([0.49999816, 0.50000184, -0.5, -0.5], dtype=np.float32)

        return trans, quat_wxyz

            


        # if cam_id == -1:
        #     rotation_matrix = self.camera_rmat @ Rotation.from_euler('xyz', [self.free_camera.elevation * np.pi / 180.0, self.free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
        #     camera_position = self.free_camera.lookat + self.free_camera.distance * rotation_matrix[:3,2]
        # else:
        #     rotation_matrix = np.array(self.mj_data.camera(self.camera_names[cam_id]).xmat).reshape((3,3))
        #     camera_position = self.mj_data.camera(self.camera_names[cam_id]).xpos

        # return camera_position, Rotation.from_matrix(rotation_matrix).as_quat()[[3,0,1,2]]

    def __del__(self):
        import mujoco
        import glfw

        # ✅ 防止访问空上下文时报错
        def safe_get_current_context():
            try:
                return glfw._get_current_context()
            except AttributeError:
                return None

        def safe_terminate():
            try:
                if glfw.get_current_context() is not None:
                    glfw.terminate()
            except Exception:
                pass

        # 替换 GLFW 方法，防止报错
        glfw._get_current_context = glfw.get_current_context
        glfw.get_current_context = safe_get_current_context
        glfw.terminate = safe_terminate

        try:
            # 1️⃣ 清理窗口资源
            if hasattr(self, 'window') and self.window is not None:
                if glfw.get_current_context() is not None:
                    glfw.destroy_window(self.window)
                    self.window = None

            # 2️⃣ 清理 MuJoCo 上下文
            mujoco.gl.shutdown()

        except Exception as e:
            print(f"清理资源时出错: {str(e)}")

        finally:
            # 3️⃣ 调用父类析构函数
            try:
                if hasattr(super(), '__del__'):
                    super().__del__()
            except Exception:
                pass

    # ------------------------------------------------------------------------------
    # ---------------------------------- Override ----------------------------------
    def reset(self):
        self.resetState()
        if self.config.enable_render:
            self.render()
        self.render_cnt = 0
        return self.getObservation()

    def updateControl(self, action):
        pass

    # 包含了一些需要子类实现的抽象方法
    @abstractmethod
    # def post_physics_step(self):
    #     pass


    def save_to_hdf5(self, data_dict, dataset_dir, episode_idx, camera_names):
        """
        将 data_dict 保存为 HDF5 文件
        
        参数:
            data_dict: 包含观测和动作数据的字典
            dataset_dir: 保存目录路径
            episode_idx: 当前episode编号
            camera_names: 相机名称列表
        """
        # 确保目录存在
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 计算最大时间步数
        max_timesteps = len(data_dict['observations']['qpos'])
        
        # 创建HDF5文件路径
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        
        t0 = time.time()
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2 * 2) as root:
            # 添加元数据
            root.attrs['sim'] = True
            
            # 创建observations组
            obs = root.create_group('observations')
            
            # 保存图像数据
            img_group = obs.create_group('images')
            for cam_name in camera_names:
                img_data = np.array(data_dict['observations']['images'][cam_name])
                img_group.create_dataset(
                    cam_name,
                    data=img_data,
                    dtype='uint8',
                    chunks=(1, 480, 640, 3)  # 分块存储提高读取效率
                )
            
            # 保存关节位置 (qpos)
            qpos_data = np.array(data_dict['observations']['qpos'])
            obs.create_dataset('qpos', data=qpos_data)
            
            # 保存动作 (actions)
            action_data = np.array(data_dict['observations']['actions'])
            root.create_dataset('action', data=action_data)
        
        print(f'保存完成，耗时: {time.time() - t0:.1f}秒')
        print(f'文件已保存至: {dataset_path}')
        self.data_dict = {
            'observations': {
                'images': {cam_name: [] for cam_name in camera_names},
                'qpos': [],
                'actions': []
            }
        }
        return dataset_path
    

    def record_frame(self, data_dict, left_arm_actions, right_arm_actions, frame_top, frame_wrist_left, frame_wrist_right):
        
        # 记录关节位置（12维：6左臂 + 6右臂）
        left_arm = left_arm_actions
        right_arm = right_arm_actions
        qpos = np.concatenate([left_arm, right_arm])
        self.data_dict['observations']['qpos'].append(qpos)
        
        # 记录动作
        self.data_dict['observations']['actions'].append(qpos)

        # top 相机
        top_camera_name = "mobilebase0_top_camera"

        # 左腕部相机
        wrist_left_camera_name = "robot0_wrist_cam_left"
        # 右腕部相机
        wrist_right_camera_name = "robot0_wrist_cam_right"

        self.data_dict['observations']['images'][top_camera_name].append(frame_top)
        self.data_dict['observations']['images'][wrist_left_camera_name].append(frame_wrist_left)
        self.data_dict['observations']['images'][wrist_right_camera_name].append(frame_wrist_right)
    # @abstractmethod

    def post_physics_step(self):
        self.finish_once = Bool()
        self.finish_once.data = False

        # print("here")
        all_images = []
        frame_top = None
        frame_wrist_left = None
        frame_wrist_right = None
        
        def tile_images(images, rows, cols):
            assert len(images) <= rows * cols
            img_h, img_w, img_c = images[0].shape
            canvas = np.zeros((img_h * rows, img_w * cols, img_c), dtype=np.uint8)

            for idx, img in enumerate(images):
                row = idx // cols
                col = idx % cols
                canvas[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w, :] = img

            return canvas
        

        for cam_id, pose in self.camera_poses.copy().items():
            
            # pose = self.camera_poses[cam_id]
            position, orientation = pose["position"], pose["orientation"]
            self.gs_renderer.set_camera_pose(position, orientation)
            img_rgb = self.getRgbImg(cam_id, position, orientation)
            if img_rgb is not None:
                img_rgb_cv2 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                # window_name = f"camera_{cam_id}"
                # cv2.imshow(window_name, img_rgb)
                # cv2.waitKey(1)
                cam_name = pose.get("name",f"Cam {cam_id}")
                cv2.putText(img_rgb_cv2, cam_name, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                all_images.append(img_rgb_cv2)
                if cam_name == "mobilebase0_top_camera":
                    frame_top = img_rgb
                elif cam_name == "robot0_wrist_cam_left":
                    frame_wrist_left = img_rgb
                elif cam_name == "robot0_wrist_cam_right":
                    frame_wrist_right = img_rgb


        if self.record_data == True and self.start_record == True:
            if self.left_joint_ready and self.right_joint_ready:
                if (self.left_joint_ready and self.right_joint_ready and
                    self.left_arm_cur_joint_pos is not None and
                    self.right_arm_cur_joint_pos is not None and
                    frame_top is not None and
                    frame_wrist_left is not None and
                    frame_wrist_right is not None):
        
                    print("both arms and images ready")
                    # self.record_frame(
                    #     self.data_dict,
                    #     self.left_arm_cur_joint_pos,
                    #     self.right_arm_cur_joint_pos,
                    #     frame_top,
                    #     frame_wrist_left,
                    #     frame_wrist_right
                    # )
                self.left_joint_ready = False
                self.right_joint_ready = False
            else:
                rospy.logwarn_once("data lack something & skip to record frame")
        elif self.record_data == False:
            rospy.logwarn_once("Record data set to false")
        elif self.start_record == False:
            rospy.logwarn_once("mujoco not launched")

        if self.record_flag == True:
            camera_names_record = ["mobilebase0_top_camera", "robot0_wrist_cam_left", "robot0_wrist_cam_right"]
            dataset_dir = "/home/cfy/cfy/cfy/gs/mobile_ai_gs/data/compare_gs"
            self.save_to_hdf5(self.data_dict, dataset_dir, self.episode_idx, camera_names_record)
            self.record_flag = False
            self.finish_once.data = True
            self.finish_record_episode_pub.publish(self.finish_once.data)
            self.episode_idx += 1
    
        canvas_img = tile_images(all_images, rows=2, cols=3)
        scale_percent = 100 # 可调整该比例，直到适合你的屏幕
        width = int(canvas_img.shape[1] * scale_percent / 100)
        height = int(canvas_img.shape[0] * scale_percent / 100)
        resized_canvas_img = cv2.resize(canvas_img, (width, height))

        cv2.imshow("All Cameras", resized_canvas_img)
        cv2.waitKey(1)

    @abstractmethod
    def getChangedObjectPose(self):
        raise NotImplementedError("pubObjectPose is not implemented")

    @abstractmethod
    def checkTerminated(self):
        raise NotImplementedError("checkTerminated is not implemented")    

    @abstractmethod
    def getObservation(self):
        raise NotImplementedError("getObservation is not implemented")

    @abstractmethod
    def getPrivilegedObservation(self):
        raise NotImplementedError("getPrivilegedObservation is not implemented")

    @abstractmethod
    def getReward(self):
        raise NotImplementedError("getReward is not implemented")
    
    # ---------------------------------- Override ----------------------------------
    # ------------------------------------------------------------------------------

    def step(self, action=None): # 主要的仿真步进函数
        # for _ in range(self.decimation):
        #     self.updateControl(action)
        #     mujoco.mj_step(self.mj_model, self.mj_data)

        terminated = self.checkTerminated()
        if terminated:
            self.resetState()
        
        
        if self.config.enable_render and self.render_cnt-1 < 100000 * self.render_fps:
            self.render()
        self.post_physics_step()
        return self.getObservation(), self.getPrivilegedObservation(), self.getReward(), terminated, {}

    def view(self):
        # self.mj_data.time += self.delta_t
        # self.mj_data.qvel[:] = 0
        # mujoco.mj_forward(self.mj_model, self.mj_data)
        # if self.render_cnt-1 < self.mj_data.time * self.render_fps:
        self.render()