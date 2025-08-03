import os
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from mujoco_sim2real.viewer.gs_render.gaussian_renderer import util_gaussian
from mujoco_sim2real.viewer.gs_render.gaussian_renderer.renderer_cuda import CUDARenderer

from mujoco_sim2real.viewer.gs_render import MOBILE_AI_ASSERT_DIR

class GSRenderer:
    print(f"GS !!!!!!")
    def __init__(self, models_dict:dict, render_width=1920, render_height=1080):
        self.width = render_width                                    # 渲染宽度
        self.height = render_height                                  # 渲染高度

        self.camera = util_gaussian.Camera(self.height, self.width)  # 相机对象

        self.update_gauss_data = False                               # 是否需要更新高斯数据的标志

        self.scale_modifier = 1.                                     # 缩放因子
        # 初始化 CUDA 渲染器
        self.renderer = CUDARenderer(self.camera.w, self.camera.h)
        self.camera_tran = np.zeros(3)                               # 相机平移向量
        self.camera_quat = np.zeros(4)                               # 相机旋转四元数

        self.gaussians_all:dict[util_gaussian.GaussianData] = {}
        self.gaussians_idx = {}
        self.gaussians_size = {}
        idx_sum = 0

        # 加载高斯模型
        gs_model_dir = Path(os.path.join(MOBILE_AI_ASSERT_DIR, "3dgs"))

        bg_key = "background"
        # 
        data_path = Path(os.path.join(gs_model_dir, models_dict[bg_key]))
        # 加载高斯模型
        gs = util_gaussian.load_ply(data_path)
        if "background_env" in models_dict.keys():
            bgenv_key = "background_env"
            bgenv_gs = util_gaussian.load_ply(Path(os.path.join(gs_model_dir, models_dict[bgenv_key])))
            gs.xyz = np.concatenate([gs.xyz, bgenv_gs.xyz], axis=0)
            gs.rot = np.concatenate([gs.rot, bgenv_gs.rot], axis=0)
            gs.scale = np.concatenate([gs.scale, bgenv_gs.scale], axis=0)
            gs.opacity = np.concatenate([gs.opacity, bgenv_gs.opacity], axis=0)
            gs.sh = np.concatenate([gs.sh, bgenv_gs.sh], axis=0)

        # 记录每个模型的起始索引和大小（gaussians_idx 和 gaussians_size）
        self.gaussians_all[bg_key] = gs
        self.gaussians_idx[bg_key] = idx_sum
        self.gaussians_size[bg_key] = gs.xyz.shape[0]
        idx_sum = self.gaussians_size[bg_key]

        for i, (k, v) in enumerate(models_dict.items()):
            if k != "background" and k != "background_env":
                data_path = Path(os.path.join(gs_model_dir, v))
                gs = util_gaussian.load_ply(data_path)
                self.gaussians_all[k] = gs
                self.gaussians_idx[k] = idx_sum
                self.gaussians_size[k] = gs.xyz.shape[0]
                idx_sum += self.gaussians_size[k]

        # 将加载的高斯数据传递给 CUDA 渲染器
        self.update_activated_renderer_state(self.gaussians_all)

        for name in self.gaussians_all.keys():
            try:
                self.gaussians_all[name].R = self.gaussians_all[name].R.numpy()
            except:
                pass
    
    def load_single_model(self, name: str, rel_path: str):
        """
        动态加载一个新的高斯模型，并更新渲染器状态
        :param name: 模型名称（如 "apple")
        :param rel_path: 相对 MOBILE_AI_ASSERT_DIR/3dgs/ 的路径
        """
        if name in self.gaussians_all:
            print(f"[GSRenderer] Model '{name}' already loaded, skipping.")
            return

        # 构造完整路径
        gs_model_dir = Path(os.path.join(MOBILE_AI_ASSERT_DIR, "3dgs"))
        data_path = Path(os.path.join(gs_model_dir, rel_path))

        if not data_path.exists():
            raise FileNotFoundError(f"[GSRenderer] Path not found: {data_path}")

        # 加载 ply 文件
        gs = util_gaussian.load_ply(data_path)

        # 添加到字典与索引
        start_idx = sum(self.gaussians_size.values())
        self.gaussians_all[name] = gs
        self.gaussians_idx[name] = start_idx
        self.gaussians_size[name] = gs.xyz.shape[0]

        # 注册到 CUDA renderer
        self.update_activated_renderer_state(self.gaussians_all)

        # numpy 转换处理（可选）
        try:
            self.gaussians_all[name].R = self.gaussians_all[name].R.numpy()
        except Exception:
            pass

        print(f"[GSRenderer] Successfully loaded model '{name}' from '{rel_path}'")



    # 惰性更新相机内参（仅当 camera.is_intrin_dirty=True 时生效）
    def update_camera_intrin_lazy(self):
        if self.camera.is_intrin_dirty:
            self.renderer.update_camera_intrin(self.camera)
            self.camera.is_intrin_dirty = False

    def update_activated_renderer_state(self, gaus: util_gaussian.GaussianData):
        self.renderer.update_gaussian_data(gaus)
        self.renderer.set_scale_modifier(self.scale_modifier)
        self.renderer.update_camera_pose(self.camera)
        self.renderer.update_camera_intrin(self.camera)
        self.renderer.set_render_reso(self.camera.w, self.camera.h)
    # 设置指定物体的位姿, 若位姿变化则更新 CUDA 渲染器的数据
    def set_obj_pose(self, obj_name, trans, quat_wzyx):
        
        if not ((self.gaussians_all[obj_name].origin_rot == quat_wzyx).all() and (self.gaussians_all[obj_name].origin_xyz == trans).all()):
            self.update_gauss_data = True
            self.gaussians_all[obj_name].origin_rot = quat_wzyx.copy()
            self.gaussians_all[obj_name].origin_xyz = trans.copy()
            self.renderer.gau_xyz_all_cu[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name],:] = torch.from_numpy(trans).cuda().requires_grad_(False)
            self.renderer.gau_rot_all_cu[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name],:] = torch.from_numpy(quat_wzyx).cuda().requires_grad_(False)
    # 设置相机的位姿
    def set_camera_pose(self, trans, quat_xyzw):
        if not ((self.camera_tran == trans).all() and (self.camera_quat == quat_xyzw).all()):
            self.camera_tran[:] = trans[:]
            self.camera_quat[:] = quat_xyzw[:]
            rmat = Rotation.from_quat(quat_xyzw).as_matrix()
            self.renderer.update_camera_pose_from_topic(self.camera, rmat, trans)
    # 设置相机的 fov
    def set_camera_fovy(self, fovy):
        if not fovy == self.camera.fovy:
            self.camera.fovy = fovy
            self.camera.is_intrin_dirty = True

    def render(self):
        self.update_camera_intrin_lazy()
        return self.renderer.draw()
