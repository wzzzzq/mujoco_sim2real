"""Simple script to train a RGB PPO policy in simulation with wandb logging

To train with rendering (single environment only):
python train_ppo_rgb.py \
  --ppo.total-timesteps 10000 \
  --ppo.render-training \
  --ppo.num-envs 1 \
  --ppo.num-eval-envs 4 \
  --ppo.learning-rate 1e-4 \
  --ppo.max-grad-norm 0.5 \
  --ppo.num-minibatches 4 \
  --ppo.track

To train without rendering (faster, multiple environments):
python train_ppo_rgb.py \
--ppo.total-timesteps 100000000 \
--ppo.num-envs 100 \
--ppo.num-eval-envs 8 \
--ppo.learning-rate 3e-4 \
--ppo.num-minibatches 16 \
--ppo.track

Note: For large num_envs (>50), reduce num_steps to maintain reasonable batch sizes.
Recommended: batch_size = num_envs * num_steps should be 1000-8000 for optimal performance.
"""

from dataclasses import dataclass, field
import json
from typing import Optional
import tyro

from ppo_rgb import PPOArgs, train

@dataclass
class Args:
    env_id: str = "PiperEnv"
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    ppo: PPOArgs = field(default_factory=PPOArgs)
    """PPO training arguments"""

def main(args: Args):
    args.ppo.env_id = args.env_id
    
    # Enable wandb tracking by default
    if not hasattr(args.ppo, 'track') or args.ppo.track is None:
        args.ppo.track = True
        print("Enabling wandb tracking by default")
    
    # Set project name for Piper environment
    args.ppo.wandb_project_name = "PiperEnv-RGB-PPO"
    
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs = json.load(f)
        args.ppo.env_kwargs = env_kwargs
    else:
        print("No env kwargs json path provided, using default env kwargs with default settings")
    
    train(args=args.ppo)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)