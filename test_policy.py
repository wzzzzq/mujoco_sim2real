#!/usr/bin/env python3
"""
Test script to load and run a trained PPO policy with rendering and video recording.

Usage:
    python test_policy.py --checkpoint runs/PiperEnv__ppo_rgb__1__<timestamp>/final_ckpt.pt
    python test_policy.py --checkpoint runs/PiperEnv__ppo_rgb__1__<timestamp>/ckpt_100.pt --episodes 5
    python test_policy.py --checkpoint runs/PiperEnv__ppo_rgb__1__<timestamp>/final_ckpt.pt --deterministic
    
    # Save video of the test process:
    python test_policy.py --save-video
    python test_policy.py --save-video --video-path my_test_video.mp4 --episodes 5
    python test_policy.py --save-video --no-render  # Record video without showing GUI
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from datetime import datetime

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

from single_piper_on_desk_env import PiperEnv


def test_policy(checkpoint_path, num_episodes=3, deterministic=True, max_steps=128, render=True, save_video=False, video_path=None):
    """
    Test a trained policy by loading the checkpoint and running episodes.
    Handles RGB + state observation format used in training.
    Optionally saves video of the test process.
    """
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Running {num_episodes} episodes with {'deterministic' if deterministic else 'stochastic'} policy")
    print(f"Rendering: {'enabled' if render else 'disabled'}")
    if save_video:
        print(f"Video recording: enabled, saving to {video_path}")
    print("-" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment with rendering if requested
    env = PiperEnv(render_mode="human" if render else None)
    
    # Setup video recording if requested
    video_frames = []  # For imageio
    if save_video:
        try:
            # Get RGB observation to determine video dimensions
            temp_obs, _ = env.reset()
            rgb_shape = temp_obs['rgb'].shape
            height, width = rgb_shape[0], rgb_shape[1]
            
            if IMAGEIO_AVAILABLE:
                # Use imageio for video recording
                print(f"✓ Video recording initialized (imageio): {width}x{height}")
            else:
                print("Warning: imageio not available. Disabling video recording.")
                save_video = False
            
            # Reset environment again for actual testing
            env.reset()
        except Exception as e:
            print(f"Warning: Could not initialize video recording: {e}")
            save_video = False
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Get sample observation for agent initialization
    obs, _ = env.reset()
    print(f"Observation format: RGB shape {obs['rgb'].shape}, State shape {obs['state'].shape}")
    
    # Import NatureCNN-based agent from the main PPO file
    from ppo_rgb import NatureCNN, Agent as TrainedAgent
    
    # Create fake envs object for agent initialization
    class FakeEnvs:
        def __init__(self):
            self.single_action_space = env.action_space
    
    fake_envs = FakeEnvs()
    
    # Create a sample observation for network initialization
    sample_obs = {
        'rgb': torch.tensor(obs['rgb']).unsqueeze(0),
        'state': torch.tensor(obs['state']).unsqueeze(0)
    }
    
    # Create agent with NatureCNN (same as training)
    agent = TrainedAgent(envs=fake_envs, sample_obs=sample_obs).to(device)
    agent.load_state_dict(checkpoint)
    agent.eval()
    
    def get_action_from_obs(obs_dict):
        """Convert observation dict to tensor format and get action."""
        converted = {}
        for key, value in obs_dict.items():
            if key == "rgb":
                converted[key] = torch.tensor(value, dtype=torch.uint8, device=device).unsqueeze(0)
            elif key == "state":
                converted[key] = torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            return agent.get_action(converted, deterministic=deterministic).cpu().numpy().squeeze()
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        for step in range(max_steps):
            # Get action from policy
            action = get_action_from_obs(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Record frame for video if enabled
            if save_video and IMAGEIO_AVAILABLE:
                rgb_frame = next_obs['rgb']
                # Store frames for imageio
                video_frames.append(rgb_frame)
            
            episode_reward += reward
            episode_length += 1
            
            # Update observation
            obs = next_obs
            
            # Check if episode is done
            done = terminated or truncated
            
            # Print step info
            if step % 10 == 0 or done:
                print(f"  Step {step:2d}: reward={reward:+6.3f}, total_reward={episode_reward:+7.3f}")
            
            if done:
                print(f"  Episode finished: {'terminated' if terminated else 'truncated'}")
                break
        
        # Episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info["is_success"]:
            success_count += 1
            print(f"  ✓ Episode successful!")
        
        print(f"  Episode {episode + 1} stats: reward={episode_reward:.3f}, length={episode_length}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Episodes run: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min/Max reward: {np.min(episode_rewards):.3f} / {np.max(episode_rewards):.3f}")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    
    # Clean up video recording
    if save_video and IMAGEIO_AVAILABLE and video_frames:
        try:
            # Save video using imageio
            imageio.mimsave(video_path, video_frames, fps=30)
            print(f"\n✓ Video saved to: {video_path}")
        except Exception as e:
            print(f"\nError saving video: {e}")
    
    env.close()
    print("\nDone!")


def find_latest_checkpoint(runs_dir="runs"):
    """
    Find the most recent checkpoint in the runs directory.
    """
    if not os.path.exists(runs_dir):
        return None
    
    # Find all run directories
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith("PiperEnv__ppo_rgb")]
    if not run_dirs:
        return None
    
    # Sort by timestamp (last part of directory name)
    run_dirs.sort(key=lambda x: int(x.split("__")[-1]))
    latest_run = run_dirs[-1]
    
    # Look for final_ckpt.pt first, then the highest numbered checkpoint
    latest_run_path = os.path.join(runs_dir, latest_run)
    
    if os.path.exists(os.path.join(latest_run_path, "final_ckpt.pt")):
        return os.path.join(latest_run_path, "final_ckpt.pt")
    
    # Look for numbered checkpoints
    ckpt_files = [f for f in os.listdir(latest_run_path) if f.startswith("ckpt_") and f.endswith(".pt")]
    if ckpt_files:
        # Sort by checkpoint number
        ckpt_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        return os.path.join(latest_run_path, ckpt_files[-1])
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO policy")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (if not provided, will find latest)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy (no action noise)")
    parser.add_argument("--max-steps", type=int, default=128,
                        help="Maximum steps per episode")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--save-video", action="store_true",
                        help="Save video of the test process")
    parser.add_argument("--video-path", type=str, default=None,
                        help="Path to save video (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Find checkpoint if not provided
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        print("No checkpoint specified, searching for latest...")
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("Error: No checkpoints found in runs/ directory")
            print("Please train a model first or specify a checkpoint path with --checkpoint")
            return
        print(f"Found latest checkpoint: {checkpoint_path}")
    
    # Generate video path if not provided but video saving is requested
    video_path = args.video_path
    if args.save_video and video_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
        video_dir = "test_videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"test_{checkpoint_name}_{timestamp}.mp4")
    
    # Test the policy
    test_policy(
        checkpoint_path=checkpoint_path,
        num_episodes=args.episodes,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        render=not args.no_render,
        save_video=args.save_video,
        video_path=video_path
    )


if __name__ == "__main__":
    main()
