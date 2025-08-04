'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''

from mujoco_sim2real.viewer.gs_render.gaussian_renderer import util_gaussian
import numpy as np
import torch
from dataclasses import dataclass
# from diff_gaussian_rasterization import GaussianRasterizer

from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from pytorch3d.transforms import quaternion_to_matrix
@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-2]

@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

def gaus_cuda_from_cpu(gau: util_gaussian) -> GaussianDataCUDA:
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus

class CUDARenderer:
    def __init__(self, w, h):
        super().__init__()
        raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1,
            "tanfovy": 1,
            "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "scale_modifier": 1.,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 3,
            "campos": None,
            "prefiltered": False,
            "debug": False
        }
        self.raster_settings = GaussianRasterizationSettingsStorage(**raster_settings)

        self.depth_render = False
        self.need_rerender = True
        self.render_rgb_img = None
        self.render_depth_img = None

        self.world_view_transform = None

    def update_gaussian_data(self, gaus: util_gaussian.GaussianData):
        self.need_rerender = True
        if type(gaus) is dict:
            gau_xyz = []
            gau_rot = []
            gau_s = []
            gau_a = []
            gau_c = []
            for gaus_item in gaus.values():
                gau_xyz.append(gaus_item.xyz)
                gau_rot.append(gaus_item.rot)
                gau_s.append(gaus_item.scale)
                gau_a.append(gaus_item.opacity)
                gau_c.append(gaus_item.sh)
            self.gau_env_idx = gau_xyz[0].shape[0]
            gau_xyz = np.concatenate(gau_xyz, axis=0)
            gau_rot = np.concatenate(gau_rot, axis=0)
            gau_s = np.concatenate(gau_s, axis=0)
            gau_a = np.concatenate(gau_a, axis=0)
            gau_c = np.concatenate(gau_c, axis=0)
            gaus_all = util_gaussian.GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c)
            self.gaussians = gaus_cuda_from_cpu(gaus_all)
        else:
            self.gaussians = gaus_cuda_from_cpu(gaus)
        self.raster_settings.sh_degree = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

        num_points = self.gaussians.xyz.shape[0]

        self.gau_ori_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_ori_xyz_all_cu[..., :] = torch.from_numpy(gau_xyz).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu[..., :] = torch.from_numpy(gau_rot).cuda().requires_grad_(False)

        self.gau_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)

    def set_scale_modifier(self, modifier):
        self.need_rerender = True
        self.raster_settings.scale_modifier = float(modifier)

    def set_render_reso(self, w, h):
        self.need_rerender = True
        self.raster_settings.image_height = int(h)
        self.raster_settings.image_width = int(w)

    def update_camera_pose(self, camera: util_gaussian.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0,2], :] *= -1
        
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings.viewmatrix = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings.campos = torch.tensor(camera.position).float().cuda()
        self.raster_settings.projmatrix = torch.tensor(proj.T).float().cuda()

    def update_camera_pose_from_topic(self, camera: util_gaussian.Camera, rmat, trans):
        self.need_rerender = True

        camera.position = np.array(trans).astype(np.float32)
        camera.target = camera.position - (1. * rmat[:3,2]).astype(np.float32)

        Tmat = np.eye(4)
        Tmat[:3,:3] = rmat
        Tmat[:3,3] = trans
        Tmat[0:3, [1,2]] *= -1
        transpose = np.array([[-1.0,  0.0,  0.0,  0.0],
                              [ 0.0, -1.0,  0.0,  0.0],
                              [ 0.0,  0.0,  1.0,  0.0],
                              [ 0.0,  0.0,  0.0,  1.0]])
        view_matrix = transpose @ np.linalg.inv(Tmat)
        self.world_view_transform = torch.from_numpy(view_matrix).float().cuda()


        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings.projmatrix = torch.tensor(proj.T).float().cuda()
        self.raster_settings.viewmatrix = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings.campos = torch.tensor(camera.position).float().cuda()

    def update_camera_intrin(self, camera: util_gaussian.Camera):
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings.tanfovx = hfovx
        self.raster_settings.tanfovy = hfovy

    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.gaussians.rot)

    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.gaussians.scale.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_normal(self, camera_center):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = camera_center - self.gaussians.xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global

    def draw(self):
        # print(f"draw !!!!!!")
        # if not self.need_rerender:
        #     return self.render_rgb_img, self.render_depth_img

        # self.need_rerender = False


        raster_settings = PlaneGaussianRasterizationSettings(
            image_height=self.raster_settings.image_height,
            image_width=self.raster_settings.image_width,
            tanfovx=self.raster_settings.tanfovx,
            tanfovy=self.raster_settings.tanfovy,
            bg=self.raster_settings.bg,
            scale_modifier=self.raster_settings.scale_modifier,
            viewmatrix=self.raster_settings.viewmatrix,
            projmatrix=self.raster_settings.projmatrix,
            sh_degree=self.raster_settings.sh_degree,
            campos=self.raster_settings.campos,
            prefiltered=False,
            render_geo=True,
            debug=self.raster_settings.debug,
        )
        means3D = self.gaussians.xyz
        screenspace_points = torch.zeros_like(self.gaussians.xyz, dtype=self.gaussians.xyz.dtype, requires_grad=True, device="cuda") + 0
        screenspace_points_abs = torch.zeros_like(self.gaussians.xyz, dtype=self.gaussians.xyz.dtype, requires_grad=True, device="cuda") + 0
        means2D = screenspace_points
        means2D_abs = screenspace_points_abs


        global_normal = self.get_normal(self.raster_settings.campos)
        
        if self.world_view_transform is not None:
            world_view_transform = self.world_view_transform.clone().detach()
        else:
            world_view_transform = torch.eye(4).float().cuda()  # 初始化为单位矩阵
        local_normal = global_normal @ world_view_transform[:3, :3]
        pts_in_cam = means3D @ world_view_transform[:3, :3] + world_view_transform[3, :3]
        depth_z = pts_in_cam[:, 2]
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance

        rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)
        rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=self.gaussians.sh,
            colors_precomp=None,
            opacities=self.gaussians.opacity,
            scales=self.gaussians.scale,
            rotations=self.gaussians.rot,
            all_map=input_all_map,
            cov3D_precomp=None,
        )

        # print(f"rendered_image type : {type(rendered_image)}")
        # if isinstance(rendered_image, tuple):
        #     print("rgb_img is a tuple, length:", len(rendered_image))
        
        return rendered_image
