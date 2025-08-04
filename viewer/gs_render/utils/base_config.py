
class BaseConfig:
    decimation     = 2                    # 降采样因子（2 表示每 2 步仿真才渲染 1 次，用于提升性能）
    timestep       = 0.005                # 仿真的时间步长
    sync           = True                 # 是否同步仿真和渲染
    headless       = False                # 是否无头模式（True 禁用可视化，用于服务器训练）
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" :  720
    }
    obs_rgb_cam_id = -1
    obs_depth_cam_id = -1
    rb_link_list   = []
    obj_list       = []
    gs_model_dict  = {}
    use_gaussian_renderer = False
    enable_render = True