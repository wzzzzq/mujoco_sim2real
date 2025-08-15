import pyrealsense2 as rs
import numpy as np
import cv2
import mujoco
import os
import torch
from scipy.spatial.transform import Rotation as R
from mujoco_sim2real.viewer.gs_render.gaussian_renderer import GSRenderer
from skimage.metrics import structural_similarity as ssim
import xml.etree.ElementTree as ET

class MuJoCoCalibrationScene:
    """MuJoCo scene for camera calibration with RealSense using mobile robot environment and GS rendering"""
    
    def __init__(self):
        # Use the mobile robot scene from mobile_robot_env
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.xml_path = os.path.join(script_dir, 'model_assets', 'mobile_ai_robot', 'scene.xml')
        self.mobile_ai_xml_path = os.path.join(script_dir, 'model_assets', 'mobile_ai_robot', 'mobile_ai.xml')
        xml_path = self.xml_path
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Store original camera parameters for calibration
        self.original_camera_pos = None
        self.original_camera_euler = None
        self.current_camera_pos = None
        self.current_camera_euler = None
        self.calibration_step_size = 0.01  # Step size for position adjustments
        self.calibration_angle_step = 0.05  # Step size for orientation adjustments
        
        # Initialize the mobile robot to a reasonable state (similar to mobile_robot_env reset)
        self._initialize_mobile_robot()
        
        # Set up GS rendering parameters (similar to mobile_robot_env)
        self.rgb_fovy = 65
        self.rgb_fovx = 90
        self.rgb_width = 640
        self.rgb_height = 480
        self.gs_model_dict = {}
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize GS renderer
        self._setup_gs_renderer()

        self.camera_name = "3rd"  # Camera name from mobile robot environment
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if self.camera_id == -1:
            # If named camera not found, use camera 0 if available
            if self.model.ncam > 0:
                self.camera_id = 0
                self.camera_name = f"camera_{self.camera_id}"
            else:
                print("No cameras found in MuJoCo scene, using free camera")
                self.camera_id = -1
        
        # Store original camera parameters
        if self.camera_id >= 0:
            self._store_original_camera_params()
                
        print(f"Using camera: {self.camera_name} (ID: {self.camera_id})")
    
    def _setup_gs_renderer(self):
        """Set up Gaussian Splatting renderer similar to mobile_robot_env"""
        # Construct model_assets path
        asset_root = os.path.join(self.script_dir, "model_assets", "3dgs_asserts")

        # Set up gs_model_dict paths (similar to mobile_robot_env)
        self.gs_model_dict["background"] = os.path.join(asset_root, "scene", "1lou_0527_res.ply")
        
        self.gs_model_dict["mobile_ai"] = os.path.join(asset_root, "robot", "chassis", "1.ply")
        
        # Left arm components (Piper arm links)
        self.gs_model_dict["left_base_link"] = os.path.join(asset_root, "robot", "piper", "arm_link1_rot.ply")
        self.gs_model_dict["left_link1"] = os.path.join(asset_root, "robot", "piper", "arm_link1_rot.ply")
        self.gs_model_dict["left_link2"] = os.path.join(asset_root, "robot", "piper", "arm_link2_rot.ply")
        self.gs_model_dict["left_link3"] = os.path.join(asset_root, "robot", "piper", "arm_link3_rot.ply")
        self.gs_model_dict["left_link4"] = os.path.join(asset_root, "robot", "piper", "arm_link4_rot.ply")
        self.gs_model_dict["left_link5"] = os.path.join(asset_root, "robot", "piper", "arm_link5_rot.ply")
        self.gs_model_dict["left_link6"] = os.path.join(asset_root, "robot", "piper", "arm_link6_rot.ply")
        self.gs_model_dict["left_link7"] = os.path.join(asset_root, "robot", "piper", "arm_link7_rot.ply")
        self.gs_model_dict["left_link8"] = os.path.join(asset_root, "robot", "piper", "arm_link8_rot.ply")
        
        # Right arm components (Piper arm links) 
        self.gs_model_dict["right_base_link"] = os.path.join(asset_root, "robot", "piper", "arm_link1_rot.ply")
        self.gs_model_dict["right_link1"] = os.path.join(asset_root, "robot", "piper", "arm_link1_rot.ply")
        self.gs_model_dict["right_link2"] = os.path.join(asset_root, "robot", "piper", "arm_link2_rot.ply")
        self.gs_model_dict["right_link3"] = os.path.join(asset_root, "robot", "piper", "arm_link3_rot.ply")
        self.gs_model_dict["right_link4"] = os.path.join(asset_root, "robot", "piper", "arm_link4_rot.ply")
        self.gs_model_dict["right_link5"] = os.path.join(asset_root, "robot", "piper", "arm_link5_rot.ply")
        self.gs_model_dict["right_link6"] = os.path.join(asset_root, "robot", "piper", "arm_link6_rot.ply")
        self.gs_model_dict["right_link7"] = os.path.join(asset_root, "robot", "piper", "arm_link7_rot.ply")
        self.gs_model_dict["right_link8"] = os.path.join(asset_root, "robot", "piper", "arm_link8_rot.ply")
        
        # Objects and environment
        self.gs_model_dict["desk"] = os.path.join(asset_root, "object", "desk", "1louzhuozi.ply")
        self.gs_model_dict["apple"] = os.path.join(asset_root, "object", "apple", "apple_res.ply")
        
        # Update robot link lists to include full mobile robot
        self.mobile_base_list = ["mobile_ai"]
        self.left_arm_list = ["left_base_link", "left_link1", "left_link2", "left_link3", "left_link4", "left_link5", "left_link6", "left_link7", "left_link8"]
        self.right_arm_list = ["right_base_link", "right_link1", "right_link2", "right_link3", "right_link4", "right_link5", "right_link6", "right_link7", "right_link8"]
        self.robot_link_list = self.mobile_base_list + self.left_arm_list + self.right_arm_list
        self.item_list = ["apple", "desk"]
        
        # Initialize GSRenderer
        self.gs_renderer = GSRenderer(self.gs_model_dict, self.rgb_width, self.rgb_height)
        self.gs_renderer.set_camera_fovy(self.rgb_fovy * np.pi / 180.)
    
    def _initialize_mobile_robot(self):
        """Initialize mobile robot to a reasonable state similar to mobile_robot_env reset"""
        # Initialize left arm joint positions (similar to mobile_robot_env reset)
        # Left arm joints are at indices 7-13 in qpos (6 arm joints + 1 gripper)
        self.data.qpos[7] = 0.0      # left_joint1
        self.data.qpos[8] = 1.1      # left_joint2
        self.data.qpos[9] = -0.95    # left_joint3
        self.data.qpos[10] = 0.0     # left_joint4
        self.data.qpos[11] = 0.976   # left_joint5
        self.data.qpos[12] = 0.0     # left_joint6
        self.data.qpos[13] = 0.035   # left_gripper
        
        # Initialize base position (freejoint at indices 0-6)
        self.data.qpos[0] = 0.15     # x position
        self.data.qpos[1] = 0.0      # y position  
        self.data.qpos[2] = -0.133   # z position
        self.data.qpos[3] = 1.0      # qw (identity quaternion)
        self.data.qpos[4] = 0.0      # qx
        self.data.qpos[5] = 0.0      # qy
        self.data.qpos[6] = 0.0      # qz
        
        # Initialize right arm to neutral position
        # Right arm joints are at indices 15-21 in qpos
        for i in range(15, 22):
            self.data.qpos[i] = 0.0
            
        # Reset velocities to zero
        self.data.qvel[:] = 0.0
        
        # Place apple on table (similar to mobile_robot_env _reset_object_pose)
        # Apple should be at a reasonable position on the table
        apple_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
        if apple_body_id != -1:
            apple_joint_id = self.model.body_jntadr[apple_body_id]
            apple_qposadr = self.model.jnt_qposadr[apple_joint_id]
            
            # Place apple on table surface
            table_center_x = 0.7
            table_center_y = 0.15
            table_surface_height = 0.73 + 0.01115 + 0.025  # table + apple offset
            
            self.data.qpos[apple_qposadr] = table_center_x      # x
            self.data.qpos[apple_qposadr + 1] = table_center_y  # y
            self.data.qpos[apple_qposadr + 2] = table_surface_height  # z
            # Keep apple orientation as identity quaternion
            self.data.qpos[apple_qposadr + 3] = 1.0  # qw
            self.data.qpos[apple_qposadr + 4] = 0.0  # qx
            self.data.qpos[apple_qposadr + 5] = 0.0  # qy
            self.data.qpos[apple_qposadr + 6] = 0.0  # qz
            
            # Reset apple velocity
            apple_qveladr = self.model.jnt_dofadr[apple_joint_id]
            self.data.qvel[apple_qveladr:apple_qveladr + 6] = 0.0
        
        # Forward simulation to update the scene
        mujoco.mj_forward(self.model, self.data)
    
    def _get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get body position and quaternion"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        
        position = np.asarray(self.data.body(body_id).xpos, dtype=np.float32)
        quaternion = np.asarray(self.data.body(body_id).xquat, dtype=np.float32)
        
        return position, quaternion
    
    def update_gs_scene(self):
        """Update GS scene with current robot and object poses"""
        # Update all robot components using regular body poses
        for name in self.robot_link_list:
            trans, quat_wxyz = self._get_body_pose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        # Update environment objects
        for name in self.item_list:
            trans, quat_wxyz = self._get_body_pose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        def multiple_quaternion_vector3d(qwxyz, vxyz):
            qw = qwxyz[..., 0]
            qx = qwxyz[..., 1]
            qy = qwxyz[..., 2]
            qz = qwxyz[..., 3]
            vx = vxyz[..., 0]
            vy = vxyz[..., 1]
            vz = vxyz[..., 2]
            qvw = -vx*qx - vy*qy - vz*qz
            qvx =  vx*qw - vy*qz + vz*qy
            qvy =  vx*qz + vy*qw - vz*qx
            qvz = -vx*qy + vy*qx + vz*qw
            vx_ =  qvx*qw - qvw*qx + qvz*qy - qvy*qz
            vy_ =  qvy*qw - qvz*qx - qvw*qy + qvx*qz
            vz_ =  qvz*qw + qvy*qx - qvx*qy - qvw*qz
            return torch.stack([vx_, vy_, vz_], dim=-1).cuda().requires_grad_(False)
        
        def multiple_quaternions(qwxyz1, qwxyz2):
            q1w = qwxyz1[..., 0]
            q1x = qwxyz1[..., 1]
            q1y = qwxyz1[..., 2]
            q1z = qwxyz1[..., 3]

            q2w = qwxyz2[..., 0]
            q2x = qwxyz2[..., 1]
            q2y = qwxyz2[..., 2]
            q2z = qwxyz2[..., 3]

            qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
            qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
            qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
            qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

            return torch.stack([qw_, qx_, qy_, qz_], dim=-1).cuda().requires_grad_(False)

        if self.gs_renderer.update_gauss_data:
            self.gs_renderer.update_gauss_data = False
            self.gs_renderer.renderer.need_rerender = True
            self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
            self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])
    
    def render_scene(self):
        """Render the current scene using GS renderer from the specified camera"""
        try:
            # Forward simulation to update the scene
            mujoco.mj_forward(self.model, self.data)
            
            # Update GS scene with current poses
            self.update_gs_scene()
            
            if self.camera_id >= 0:
                # Get camera pose from MuJoCo
                cam_pos = self.data.cam_xpos[self.camera_id]
                cam_rot = self.data.cam_xmat[self.camera_id].reshape((3, 3))
                cam_quat = R.from_matrix(cam_rot).as_quat()

                # Set GS camera parameters and render
                self.gs_renderer.set_camera_fovy(self.rgb_fovy * np.pi / 180.)
                self.gs_renderer.set_camera_pose(cam_pos, cam_quat)
                
                with torch.inference_mode():
                    rgb_img = self.gs_renderer.render()

                if isinstance(rgb_img, torch.Tensor):
                    rgb_img = rgb_img.detach().cpu().numpy()

                # If [3, H, W] format, convert to [H, W, 3]
                if rgb_img.shape[0] == 3 and len(rgb_img.shape) == 3:
                    rgb_img = np.transpose(rgb_img, (1, 2, 0))

                # Ensure values in [0,1] range and convert to uint8
                rgb_img = np.clip(rgb_img, 0, 1)
                rgb_img = (rgb_img * 255).astype(np.uint8)
                
                return rgb_img
            else:
                print("No valid camera found for GS rendering")
                return np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"GS rendering error: {e}")
            # Return black image if rendering fails
            return np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)

    def _store_original_camera_params(self):
        """Store original camera parameters from XML"""
        try:
            tree = ET.parse(self.mobile_ai_xml_path)
            root = tree.getroot()
            
            camera_elem = root.find(f".//camera[@name='{self.camera_name}']")
            if camera_elem is not None:
                # Parse position
                pos_str = camera_elem.get('pos', '0 0 0')
                self.original_camera_pos = np.array([float(x) for x in pos_str.split()])
                self.current_camera_pos = self.original_camera_pos.copy()
                
                # Parse orientation (euler angles)
                euler_str = camera_elem.get('euler', '0 0 0')
                self.original_camera_euler = np.array([float(x) for x in euler_str.split()])
                self.current_camera_euler = self.original_camera_euler.copy()
                
                print(f"Original camera pos: {self.original_camera_pos}")
                print(f"Original camera euler: {self.original_camera_euler}")
            else:
                print("Could not find 3rd camera in XML")
        except Exception as e:
            print(f"Error reading camera parameters from XML: {e}")
            # Fallback values
            self.original_camera_pos = np.array([0.125, 0.0, 1.405])
            self.original_camera_euler = np.array([0, -1.03, -1.57])
            self.current_camera_pos = self.original_camera_pos.copy()
            self.current_camera_euler = self.original_camera_euler.copy()

    def _update_camera_in_xml(self, pos, euler):
        """Update camera position and orientation in XML file"""
        try:
            tree = ET.parse(self.mobile_ai_xml_path)
            root = tree.getroot()

            camera_elem = root.find(f".//camera[@name='{self.camera_name}']")
            if camera_elem is not None:
                # Update position
                pos_str = f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
                camera_elem.set('pos', pos_str)
                
                # Update orientation
                euler_str = f"{euler[0]:.6f} {euler[1]:.6f} {euler[2]:.6f}"
                camera_elem.set('euler', euler_str)
                
                # Write back to file
                tree.write(self.mobile_ai_xml_path, encoding='utf-8', xml_declaration=True)
                return True
            else:
                print("Could not find 3rd camera in XML")
                return False
        except Exception as e:
            print(f"Error updating camera parameters in XML: {e}")
            return False

    def _reload_model_with_new_camera(self, pos, euler):
        """Reload MuJoCo model with updated camera parameters"""
        try:
            # Update XML file
            if not self._update_camera_in_xml(pos, euler):
                return False
            
            # Reload model
            new_model = mujoco.MjModel.from_xml_path(self.xml_path)
            new_data = mujoco.MjData(new_model)
            
            # Copy current state to new model
            new_data.qpos[:] = self.data.qpos[:]
            new_data.qvel[:] = self.data.qvel[:]
            
            # Update model and data
            self.model = new_model
            self.data = new_data
            
            # Update camera ID (should be the same but just in case)
            self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
            
            # Forward simulation to update the scene
            mujoco.mj_forward(self.model, self.data)
            
            return True
            
        except Exception as e:
            print(f"Error reloading model: {e}")
            return False

    def _calculate_image_similarity(self, img1, img2):
        """Calculate similarity between two images using SSIM and MSE"""
        try:
            # Convert to grayscale
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1
                
            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2
            
            # Resize images to same size if needed
            if gray1.shape != gray2.shape:
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
            # Calculate SSIM
            ssim_score = ssim(gray1, gray2)
            
            # Calculate normalized MSE (lower is better, so we return 1-normalized_mse)
            mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
            normalized_mse = mse / (255.0 ** 2)
            mse_similarity = 1.0 - normalized_mse
            
            # Combine scores (SSIM is more perceptually relevant)
            combined_score = 0.7 * ssim_score + 0.3 * mse_similarity
            
            return combined_score
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def calibrate_camera_grid_search(self, target_image, grid_size=7, pos_step=0.05, angle_step=0.1):
        """Grid search calibration method that searches around current position"""
        print("Starting grid search camera calibration...")
        
        # Store target image
        if len(target_image.shape) == 3:
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_image
        
        # Get initial similarity score
        initial_rendered = self.render_scene()
        if initial_rendered is not None:
            initial_gray = cv2.cvtColor(initial_rendered, cv2.COLOR_RGB2GRAY)
            initial_similarity = self._calculate_image_similarity(target_gray, initial_gray)
            print(f"Initial similarity score: {initial_similarity:.4f}")
        else:
            initial_similarity = 0.0
        
        best_score = initial_similarity
        best_pos = self.current_camera_pos.copy()
        best_euler = self.current_camera_euler.copy()
        
        # Create grid around current position
        print(f"Grid search parameters: size={grid_size}x{grid_size}x{grid_size}, pos_step={pos_step}, angle_step={angle_step}")
        
        # Calculate grid ranges
        half_grid = grid_size // 2
        pos_offsets = [(i - half_grid) * pos_step for i in range(grid_size)]
        angle_offsets = [(i - half_grid) * angle_step for i in range(grid_size)]
        
        total_combinations = grid_size ** 6  # 3 position + 3 angle parameters
        print(f"Total combinations to test: {total_combinations}")
        
        count = 0
        best_count = 0
        
        # Grid search over position and orientation
        for dx in pos_offsets:
            for dy in pos_offsets:
                for dz in pos_offsets:
                    for droll in angle_offsets:
                        for dpitch in angle_offsets:
                            for dyaw in angle_offsets:
                                count += 1
                                
                                # Calculate new position and orientation
                                test_pos = self.current_camera_pos + np.array([dx, dy, dz])
                                test_euler = self.current_camera_euler + np.array([droll, dpitch, dyaw])
                                
                                try:
                                    # Update camera parameters and reload model
                                    if not self._reload_model_with_new_camera(test_pos, test_euler):
                                        continue
                                    
                                    # Render scene
                                    rendered_image = self.render_scene()
                                    if rendered_image is None:
                                        continue
                                    
                                    # Convert to grayscale
                                    if len(rendered_image.shape) == 3:
                                        rendered_gray = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2GRAY)
                                    else:
                                        rendered_gray = rendered_image
                                    
                                    # Calculate similarity
                                    similarity = self._calculate_image_similarity(target_gray, rendered_gray)
                                    
                                    # Check if this is the best so far
                                    if similarity > best_score:
                                        best_score = similarity
                                        best_pos = test_pos.copy()
                                        best_euler = test_euler.copy()
                                        best_count = count
                                        print(f"  New best at {count}/{total_combinations}: pos={test_pos}, euler={test_euler}, similarity={similarity:.4f}")
                                    
                                    # Progress update every 1000 iterations
                                    if count % 1000 == 0:
                                        print(f"  Progress: {count}/{total_combinations} ({100*count/total_combinations:.1f}%), current best: {best_score:.4f}")
                                
                                except Exception as e:
                                    # Skip this combination if there's an error
                                    continue
        
        # Apply the best result
        improvement = best_score - initial_similarity
        if improvement > 0.001:  # Small threshold to avoid noise
            print(f"\nGrid search successful!")
            print(f"Original pos: {self.current_camera_pos}")
            print(f"Optimized pos: {best_pos}")
            print(f"Original euler: {self.current_camera_euler}")
            print(f"Optimized euler: {best_euler}")
            print(f"Improvement: {improvement:.4f}")
            print(f"Final similarity score: {best_score:.4f}")
            print(f"Best found at iteration {best_count}/{total_combinations}")
            
            # Update current parameters
            self.current_camera_pos = best_pos
            self.current_camera_euler = best_euler
            
            # Final reload with optimized parameters
            success = self._reload_model_with_new_camera(best_pos, best_euler)
            
            return success, best_score
        else:
            print(f"\nNo significant improvement found.")
            print(f"Best improvement: {improvement:.4f}")
            print(f"Final similarity score: {best_score:.4f}")
            # Restore original camera
            self._reload_model_with_new_camera(self.current_camera_pos, self.current_camera_euler)
            return False, initial_similarity

    def calibrate_camera_adaptive_grid_search(self, target_image, initial_grid_size=5, initial_step=0.1, refinement_levels=3):
        """Adaptive grid search that starts coarse and refines around the best result"""
        print("Starting adaptive grid search camera calibration...")
        
        # Store target image
        if len(target_image.shape) == 3:
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_image
        
        # Get initial similarity score
        initial_rendered = self.render_scene()
        if initial_rendered is not None:
            initial_gray = cv2.cvtColor(initial_rendered, cv2.COLOR_RGB2GRAY)
            initial_similarity = self._calculate_image_similarity(target_gray, initial_gray)
            print(f"Initial similarity score: {initial_similarity:.4f}")
        else:
            initial_similarity = 0.0
        
        best_score = initial_similarity
        best_pos = self.current_camera_pos.copy()
        best_euler = self.current_camera_euler.copy()
        search_center_pos = self.current_camera_pos.copy()
        search_center_euler = self.current_camera_euler.copy()
        
        current_pos_step = initial_step
        current_angle_step = initial_step
        
        for level in range(refinement_levels):
            print(f"\nRefinement level {level + 1}/{refinement_levels}")
            print(f"Grid size: {initial_grid_size}x{initial_grid_size}x{initial_grid_size}")
            print(f"Position step: {current_pos_step:.4f}, Angle step: {current_angle_step:.4f}")
            print(f"Search center pos: {search_center_pos}")
            print(f"Search center euler: {search_center_euler}")
            
            # Create grid around search center
            half_grid = initial_grid_size // 2
            pos_offsets = [(i - half_grid) * current_pos_step for i in range(initial_grid_size)]
            angle_offsets = [(i - half_grid) * current_angle_step for i in range(initial_grid_size)]
            
            level_best_score = best_score
            level_best_pos = best_pos.copy()
            level_best_euler = best_euler.copy()
            
            total_combinations = initial_grid_size ** 6
            count = 0
            
            # Grid search over position and orientation
            for dx in pos_offsets:
                for dy in pos_offsets:
                    for dz in pos_offsets:
                        for droll in angle_offsets:
                            for dpitch in angle_offsets:
                                for dyaw in angle_offsets:
                                    count += 1
                                    
                                    # Calculate new position and orientation
                                    test_pos = search_center_pos + np.array([dx, dy, dz])
                                    test_euler = search_center_euler + np.array([droll, dpitch, dyaw])
                                    
                                    try:
                                        # Update camera parameters and reload model
                                        if not self._reload_model_with_new_camera(test_pos, test_euler):
                                            continue
                                        
                                        # Render scene
                                        rendered_image = self.render_scene()
                                        if rendered_image is None:
                                            continue
                                        
                                        # Convert to grayscale
                                        if len(rendered_image.shape) == 3:
                                            rendered_gray = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2GRAY)
                                        else:
                                            rendered_gray = rendered_image
                                        
                                        # Calculate similarity
                                        similarity = self._calculate_image_similarity(target_gray, rendered_gray)
                                        
                                        # Check if this is the best so far
                                        if similarity > level_best_score:
                                            level_best_score = similarity
                                            level_best_pos = test_pos.copy()
                                            level_best_euler = test_euler.copy()
                                            print(f"    New best at {count}/{total_combinations}: similarity={similarity:.4f}")
                                        
                                        # Progress update every 500 iterations
                                        if count % 500 == 0:
                                            print(f"    Progress: {count}/{total_combinations} ({100*count/total_combinations:.1f}%), best: {level_best_score:.4f}")
                                    
                                    except Exception as e:
                                        # Skip this combination if there's an error
                                        continue
            
            # Update best results and search center for next level
            if level_best_score > best_score:
                best_score = level_best_score
                best_pos = level_best_pos.copy()
                best_euler = level_best_euler.copy()
                print(f"  Level {level + 1} improved score to: {best_score:.4f}")
            else:
                print(f"  Level {level + 1} no improvement, best remains: {best_score:.4f}")
            
            # Update search center for next refinement level
            search_center_pos = best_pos.copy()
            search_center_euler = best_euler.copy()
            
            # Reduce step size for next level
            current_pos_step *= 0.5
            current_angle_step *= 0.5
        
        # Apply the best result
        improvement = best_score - initial_similarity
        if improvement > 0.001:  # Small threshold to avoid noise
            print(f"\nAdaptive grid search successful!")
            print(f"Original pos: {self.current_camera_pos}")
            print(f"Optimized pos: {best_pos}")
            print(f"Original euler: {self.current_camera_euler}")
            print(f"Optimized euler: {best_euler}")
            print(f"Improvement: {improvement:.4f}")
            print(f"Final similarity score: {best_score:.4f}")
            
            # Update current parameters
            self.current_camera_pos = best_pos
            self.current_camera_euler = best_euler
            
            # Final reload with optimized parameters
            success = self._reload_model_with_new_camera(best_pos, best_euler)
            
            return success, best_score
        else:
            print(f"\nNo significant improvement found.")
            print(f"Best improvement: {improvement:.4f}")
            print(f"Final similarity score: {best_score:.4f}")
            # Restore original camera
            self._reload_model_with_new_camera(self.current_camera_pos, self.current_camera_euler)
            return False, initial_similarity

    def reset_camera_to_original(self):
        """Reset camera to original parameters"""
        print("Resetting camera to original parameters...")
        success = self._reload_model_with_new_camera(self.original_camera_pos, self.original_camera_euler)
        if success:
            self.current_camera_pos = self.original_camera_pos.copy()
            self.current_camera_euler = self.original_camera_euler.copy()
            print("Camera reset successful")
        else:
            print("Camera reset failed")
        return success

if __name__ == "__main__":
    # Initialize MuJoCo calibration scene
    mujoco_scene = MuJoCoCalibrationScene()
    
    # Configure RealSense streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    print("Camera Calibration Tool")
    print("Controls:")
    print("- 'q' or ESC: Quit")
    print("- 's': Save current frames")
    print("- 'r': Reset MuJoCo scene")
    print("- 'g': Start grid search calibration (systematic search)")
    print("- 'a': Start adaptive grid search calibration (multi-level)")
    print("- 'o': Reset camera to original position")
    print("- '+': Increase camera transparency (less visible)")
    print("- '-': Decrease camera transparency (more visible)")
    
    frame_counter = 0
    alpha = 0.5  # Initial transparency level
    calibration_target = None  # Store target image for calibration
    
    try:
        while True:
            # Get RealSense RGB frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            # Convert RealSense frame to numpy array
            realsense_image = np.asanyarray(color_frame.get_data())
            
            # Get MuJoCo rendered frame
            mujoco_image = mujoco_scene.render_scene()
            
            # Convert MuJoCo RGB to BGR for display
            if mujoco_image is not None:
                mujoco_image_bgr = cv2.cvtColor(mujoco_image, cv2.COLOR_RGB2BGR)
            else:
                mujoco_image_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create overlay by blending RealSense frame with transparency on MuJoCo frame
            overlay_image = cv2.addWeighted(mujoco_image_bgr, 1 - alpha, realsense_image, alpha, 0)
            
            # Add labels to images
            cv2.putText(realsense_image, 'RealSense RGB', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(mujoco_image_bgr, f'GS Render ({mujoco_scene.camera_name})', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay_image, f'Overlay (Alpha: {alpha:.1f})', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Stack images horizontally for comparison: RealSense | MuJoCo | Overlay
            combined_image = np.hstack((realsense_image, mujoco_image_bgr, overlay_image))
            
            # Display combined image
            cv2.namedWindow('Camera Calibration - RealSense vs GS Render', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Camera Calibration - RealSense vs GS Render', combined_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Save frames
                cv2.imwrite(f'realsense_frame_{frame_counter:04d}.jpg', realsense_image)
                cv2.imwrite(f'gs_render_frame_{frame_counter:04d}.jpg', mujoco_image_bgr)
                print(f"Saved frames {frame_counter:04d}")
                frame_counter += 1
            elif key == ord('r'):  # Reset MuJoCo scene
                mujoco.mj_resetData(mujoco_scene.model, mujoco_scene.data)
                mujoco_scene._initialize_mobile_robot()
                print("Mobile robot scene reset")
            elif key == ord('g'):  # Start grid search calibration
                print("Starting grid search calibration...")
                print("Using current RealSense frame as calibration target.")
                calibration_target = realsense_image.copy()
                
                # Run grid search calibration
                try:
                    success, similarity = mujoco_scene.calibrate_camera_grid_search(
                        calibration_target, 
                        grid_size=5,      # Smaller grid for faster search
                        pos_step=0.05,    # 5cm position steps
                        angle_step=0.1    # 0.1 radian angle steps (~5.7 degrees)
                    )
                    if success:
                        print(f"Grid search calibration completed with similarity score: {similarity:.4f}")
                    else:
                        print("Grid search calibration failed")
                except Exception as e:
                    print(f"Grid search calibration error: {e}")
                    
            elif key == ord('a'):  # Start adaptive grid search calibration
                print("Starting adaptive grid search calibration...")
                print("Using current RealSense frame as calibration target.")
                calibration_target = realsense_image.copy()
                
                # Run adaptive grid search calibration
                try:
                    success, similarity = mujoco_scene.calibrate_camera_adaptive_grid_search(
                        calibration_target,
                        initial_grid_size=5,      # Start with 5x5x5x5x5x5 grid
                        initial_step=0.1,         # Start with larger steps
                        refinement_levels=3       # 3 levels of refinement
                    )
                    if success:
                        print(f"Adaptive grid search calibration completed with similarity score: {similarity:.4f}")
                    else:
                        print("Adaptive grid search calibration failed")
                except Exception as e:
                    print(f"Adaptive grid search calibration error: {e}")
                    
            elif key == ord('o'):  # Reset camera to original
                mujoco_scene.reset_camera_to_original()
                print("Camera reset to original position")
            elif key == ord('+') or key == ord('='):  # Increase transparency (less visible camera)
                alpha = min(1.0, alpha + 0.1)
                print(f"Camera transparency: {alpha:.1f}")
            elif key == ord('-'):  # Decrease transparency (more visible camera)
                alpha = max(0.0, alpha - 0.1)
                print(f"Camera transparency: {alpha:.1f}")
                
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        pipeline.stop()
        print("Camera calibration tool closed")
