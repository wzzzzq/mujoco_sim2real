"""
Real-time apple position visualization in MuJoCo.
Shows the current apple position and allows manual resets to see the randomization.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
from mobile_robot_env import PiperEnv


def batch_visualization(num_resets=20):
    """
    Automatically reset the environment multiple times and show positions.
    """
    print(f"Batch Apple Position Visualization ({num_resets} resets)")
    print("=" * 60)
    
    env = PiperEnv(render_mode="human")
    positions = []
    
    try:
        for i in range(num_resets):
            obs, info = env.reset()
            apple_pos, _ = env._get_body_pose('apple')
            positions.append(apple_pos.copy())
            
            # Calculate distance from center
            center_x, center_y = 0.06, 0.0
            distance = np.sqrt((apple_pos[0] - center_x)**2 + (apple_pos[1] - center_y)**2)
            
            print(f"Reset {i+1:2d}: x={apple_pos[0]:6.3f}, y={apple_pos[1]:6.3f}, z={apple_pos[2]:6.3f}, dist={distance:.3f}")
            
            if hasattr(env, 'handle') and env.handle:
                env.handle.sync()
            
            time.sleep(0.05)  # Pause to see each position
    
    finally:
        # Print statistics
        if positions:
            positions = np.array(positions)
            print("\n" + "=" * 60)
            print("STATISTICS:")
            print(f"Number of samples: {len(positions)}")
            print(f"X: min={positions[:, 0].min():.3f}, max={positions[:, 0].max():.3f}, mean={positions[:, 0].mean():.3f}")
            print(f"Y: min={positions[:, 1].min():.3f}, max={positions[:, 1].max():.3f}, mean={positions[:, 1].mean():.3f}")
            print(f"Z: min={positions[:, 2].min():.3f}, max={positions[:, 2].max():.3f}, mean={positions[:, 2].mean():.3f}")
            
            center_x, center_y = 0.06, 0.0
            distances = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
            print(f"Distance from center: min={distances.min():.3f}, max={distances.max():.3f}, mean={distances.mean():.3f}")
        
        env.close()


def main():
    num_resets = 1000
    batch_visualization(num_resets)


if __name__ == "__main__":
    main()
