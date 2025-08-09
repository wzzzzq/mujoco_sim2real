from dm_control import mujoco
from dm_control.mujoco.export import export_with_assets

# 导出模型为GLB（包含所有网格）
physics = mujoco.Physics.from_xml_path("mobile_ai.xml")
export_with_assets(physics, "output.glb")

