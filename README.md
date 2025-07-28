# MuJoCo Sim2Real PPO Training

PPO RGB policy training for Piper robot manipulation tasks in MuJoCo simulation.

## Quick Start

### Install Dependencies
```bash
pip install torch gymnasium mujoco wandb tyro
```

### Train Policy

To train with rendering (single environment only):
```python
python train_ppo_rgb.py \
  --ppo.total-timesteps 10000 \
  --ppo.render-training \
  --ppo.num-envs 1 \
  --ppo.num-eval-envs 4 \
  --ppo.learning-rate 1e-4 \
  --ppo.max-grad-norm 0.5 \
  --ppo.num-minibatches 4 \
  --ppo.track
```

To train without rendering (faster, multiple environments):
```python
python train_ppo_rgb.py \
--ppo.total-timesteps 10000000 \
--ppo.num-envs 24 \
--ppo.num-eval-envs 8 \
--ppo.learning-rate 3e-4 \
--ppo.num-minibatches 8 \
--ppo.track
```

### Test Trained Policy
```bash
python test_policy.py # automatically runs the latest policy
```
or

```bash
python test_policy.py --model-path runs/PiperEnv__ppo_rgb__1__<timestamp>/model.pth
```


## Files

- `train_ppo_rgb.py` - Main training script with wandb logging
- `ppo_rgb.py` - PPO implementation for RGB observations
- `single_piper_on_desk_env.py` - Piper environment setup
- `test_policy.py` - Policy evaluation utilities
- `model_assets/` - Robot and environment assets

## Environment

The environment simulates a Piper robot performing manipulation tasks on a desk using RGB visual observations. Training logs are automatically saved to `runs/` and tracked with Weights & Biases.