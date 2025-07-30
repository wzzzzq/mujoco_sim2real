from collections import defaultdict
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports - removed, using custom environment
# Import our custom environment
from single_piper_on_desk_env import PiperEnv

@dataclass
class PPOArgs:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PiperEnv-RGB-PPO"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""
    render_training: bool = False
    """if toggled, enable rendering during training (will slow down training significantly)"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_kwargs: dict = field(default_factory=dict)
    """extra environment kwargs to pass to the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout
    NOTE: batch_size = num_envs * num_steps. For large num_envs (>50), consider reducing num_steps to 10-50
    to maintain reasonable batch sizes and update frequencies."""
    num_eval_steps: int = 129
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = None
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)
    def get_features(self, x):
        return self.feature_net(x)
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            # Log with explicit step to ensure proper tracking
            wandb.log({tag: scalar_value, "global_step": step}, step=step)
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        if self.writer:
            self.writer.close()
        if self.log_wandb:
            import wandb
            wandb.finish()

def train(args: PPOArgs):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # Ensure minibatch size is at least 2 to avoid numerical issues
    if args.minibatch_size < 2:
        args.num_minibatches = max(1, args.batch_size // 2)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        print(f"Adjusted num_minibatches to {args.num_minibatches} to ensure minibatch_size >= 2")
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup - use our custom PiperEnv
    def make_env(enable_render=False):
        render_mode = "human" if enable_render else None
        env = PiperEnv(render_mode=render_mode)
        # Add episode tracking wrapper for proper logging
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    # Create vectorized environments (use sync to avoid multiprocessing issues)
    # Enable rendering for training environments if requested (only for single env to avoid multiple windows)
    train_render = args.render_training and args.num_envs == 1
    if args.render_training and args.num_envs > 1:
        print(f"WARNING: Rendering requested but num_envs={args.num_envs} > 1. Rendering disabled to avoid multiple windows.")
        print("To enable rendering, use --ppo.num-envs 1")
        train_render = False
    elif args.render_training:
        print("Rendering enabled for training environment")
    
    eval_envs = gym.vector.SyncVectorEnv([lambda: make_env(False) for _ in range(args.num_eval_envs)])
    envs = gym.vector.SyncVectorEnv([lambda: make_env(train_render) for _ in range(args.num_envs if not args.evaluate else 1)])

    # No need for wrappers since our environment already provides the right format
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        # Video recording disabled for custom environment
        
    # Remove ManiSkill wrappers - not needed for our simple environment
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = 128  # Fixed episode length for our environment
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(num_envs=args.num_envs, env_id="PiperEnv", env_horizon=max_episode_steps)
            config["eval_env_cfg"] = dict(num_envs=args.num_eval_envs, env_id="PiperEnv", env_horizon=max_episode_steps)
            # Add additional config for better tracking
            config["observation_type"] = "RGB + State"
            config["state_dim"] = 13
            config["rgb_shape"] = (128, 128, 3)
            config["action_dim"] = 7
            config["network_type"] = "NatureCNN + State"
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,  # Disable sync to avoid step tracking issues
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "rgb", "robotics", "manipulation", "piper"],
                notes=f"RGB+State PPO training on Piper robot grasping task. Envs: {args.num_envs}, LR: {args.learning_rate}",
                resume="allow"  # Allow resuming runs if they get interrupted
            )
            # Define custom metrics for proper step tracking
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            # Define custom charts for better visualization
            wandb.define_metric("train/rollout_mean_reward", step_metric="global_step")
            wandb.define_metric("train/rollout_success_rate", step_metric="global_step")
            wandb.define_metric("eval/episode_reward_mean", step_metric="global_step")
            wandb.define_metric("eval/success_rate", step_metric="global_step")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    # Convert to appropriate format for dictionary observations
    def convert_obs(obs_dict):
        converted = {}
        for key, value in obs_dict.items():
            if key == "rgb":
                converted[key] = torch.tensor(value, dtype=torch.uint8, device=device)
            elif key == "state":
                converted[key] = torch.tensor(value, dtype=torch.float32, device=device)
        return converted
    
    next_obs = convert_obs(next_obs)
    eval_obs = convert_obs(eval_obs)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        
        # Track rollout-level episode statistics
        rollout_episode_rewards = []
        rollout_episode_successes = []
        rollout_episode_lengths = []
        
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_obs = convert_obs(eval_obs)
            eval_metrics = defaultdict(list)
            num_episodes = 0
            step_count = 0
            for step_i in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = agent.get_action(eval_obs, deterministic=True)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action.cpu().numpy())
                    eval_obs = convert_obs(eval_obs)
                    step_count += args.num_eval_envs
                    
                    # Debug: Check for episode terminations
                    #if step_i < 5 or step_i % 5 == 0:  # Print first few steps and every 20th step
                    print(f"  Eval step {step_i}: terminations={eval_terminations.sum()}, truncations={eval_truncations.sum()}")
                    
                    # Handle episode completion with RecordEpisodeStatistics wrapper
                    if "final_info" in eval_infos:
                        final_infos = eval_infos["final_info"]
                        for info in final_infos:
                            if info and "episode" in info:
                                num_episodes += 1
                                episode_info = info["episode"]
                                eval_metrics["r"].append(float(episode_info["r"]))
                                eval_metrics["l"].append(int(episode_info["l"]))
                                print(f"  Eval episode completed: reward={float(episode_info['r']):.3f}, length={int(episode_info['l'])}")
                                # Extract custom success info if available
                                if "is_success" in info:
                                    eval_metrics["success"].append(float(info["is_success"]))
                            
            print(f"Evaluated {step_count} steps resulting in {num_episodes} episodes")
            
            # Log metrics
            if eval_metrics and logger is not None:
                if "r" in eval_metrics:
                    episode_rewards = eval_metrics["r"]
                    logger.add_scalar("eval/episode_reward_mean", np.mean(episode_rewards), global_step)
                    logger.add_scalar("eval/episode_reward_max", np.max(episode_rewards), global_step)
                    logger.add_scalar("eval/episode_reward_min", np.min(episode_rewards), global_step)
                    print(f"eval_reward_mean={np.mean(episode_rewards):.3f}")
                
                if "success" in eval_metrics:
                    success_rate = np.mean(eval_metrics["success"])
                    logger.add_scalar("eval/success_rate", success_rate, global_step)
                    print(f"eval_success_rate={success_rate:.3f}")
                
                if "l" in eval_metrics:
                    logger.add_scalar("eval/episode_length_mean", np.mean(eval_metrics["l"]), global_step)
            
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs = convert_obs(next_obs)
            # Convert terminations and truncations to tensors before logical_or
            terminations = torch.tensor(terminations, dtype=torch.bool, device=device)
            truncations = torch.tensor(truncations, dtype=torch.bool, device=device)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            
            # Convert rewards - ensure it matches the expected shape
            reward_tensor = torch.tensor(reward, device=device, dtype=torch.float32).reshape(args.num_envs)
            rewards[step] = reward_tensor * args.reward_scale

            if "final_info" in infos:
                final_infos = infos["final_info"]
                for info in final_infos:
                    if info and "episode" in info:
                        episode_info = info["episode"]
                        # Extract scalar values from numpy arrays
                        episode_reward = float(episode_info["r"])
                        episode_length = int(episode_info["l"])
                        
                        # Collect for rollout-level statistics
                        rollout_episode_rewards.append(episode_reward)
                        rollout_episode_lengths.append(episode_length)
                        
                        if logger is not None:
                            logger.add_scalar("train/episode_reward", episode_reward, global_step)
                            logger.add_scalar("train/episode_length", episode_length, global_step)
                            print(f"Train episode: reward={episode_reward:.3f}, length={episode_length}")
                        
                        # Log custom success info if available
                        if "is_success" in info:
                            success_value = float(info["is_success"])
                            rollout_episode_successes.append(success_value)
                            if logger is not None:
                                logger.add_scalar("train/episode_success", success_value, global_step)

                # Convert final observations to tensor format for value computation
                try:
                    if "final_observation" in infos:
                        final_obs_converted = convert_obs(infos["final_observation"])
                        with torch.no_grad():
                            final_vals = agent.get_value(final_obs_converted).view(-1)
                            # Apply final values where episodes terminated
                            termination_mask = torch.logical_or(terminations, truncations)
                            final_values[step, termination_mask] = final_vals[termination_mask]
                except:
                    pass  # Skip if final observation handling fails
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        
        # Log rollout-level statistics
        if rollout_episode_rewards and logger is not None:
            rollout_mean_reward = np.mean(rollout_episode_rewards)
            rollout_max_reward = np.max(rollout_episode_rewards)
            rollout_min_reward = np.min(rollout_episode_rewards)
            logger.add_scalar("train/rollout_mean_reward", rollout_mean_reward, global_step)
            logger.add_scalar("train/rollout_max_reward", rollout_max_reward, global_step)
            logger.add_scalar("train/rollout_min_reward", rollout_min_reward, global_step)
            logger.add_scalar("train/rollout_num_episodes", len(rollout_episode_rewards), global_step)
            print(f"Rollout {iteration}: {len(rollout_episode_rewards)} episodes, mean_reward={rollout_mean_reward:.3f}")
            
            if rollout_episode_successes:
                rollout_success_rate = np.mean(rollout_episode_successes)
                logger.add_scalar("train/rollout_success_rate", rollout_success_rate, global_step)
                print(f"Rollout {iteration}: success_rate={rollout_success_rate:.3f}")
            
            rollout_mean_length = np.mean(rollout_episode_lengths)
            logger.add_scalar("train/rollout_mean_length", rollout_mean_length, global_step)
        else:
            print(f"Rollout {iteration}: No episodes completed in this rollout")
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()
