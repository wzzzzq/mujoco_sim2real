# MuJoCo Sim2Real PPO Training

PPO RGB policy training for Piper robot manipulation tasks in MuJoCo simulation.

## Quick Start

### Install Dependencies
```bash
conda create -n mujoco_sim2real python==3.10.9
conda activate mujoco_sim2real
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# install pytorch 3d from source
cd submodules/
git clone git@github.com:facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
# install diff-plane-guassian
cd ..
cd diff-plane-rasterization/
pip install -e .
```

### Train Policy

To train with rendering (single environment only):
```python
python -m mujoco_sim2real.train_ppo_rgb \
  --ppo.total-timesteps 10000 \
  --ppo.render-training \
  --ppo.num-envs 1 \
  --ppo.num-eval-envs 1 \
  --ppo.learning-rate 1e-4 \
  --ppo.max-grad-norm 0.5 \
  --ppo.num-minibatches 4 \
  --ppo.track
```

To train without rendering (faster, multiple environments):
```python
python -m mujoco_sim2real.train_ppo_rgb \
--ppo.total-timesteps 5000000 \
--ppo.num-envs 100 \
--ppo.num-eval-envs 4 \
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