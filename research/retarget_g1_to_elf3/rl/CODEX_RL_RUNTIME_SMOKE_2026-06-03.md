# CODEX RL Runtime Smoke 2026-06-03

## Scope

Set up and smoke-test a real Gymnasium/SB3 runtime for `research/retarget_g1_to_elf3/rl` on the shared workstation without changing system drivers, system CUDA, `base`, or other users' processes.

## Runtime Boundary

- Conda environment: `/home/chengjunhao/miniforge3/envs/kimodo-elf3-rl`
- Python: `3.11.15`
- GPU visibility used for probes/smoke: `CUDA_VISIBLE_DEVICES=4`
- No `sudo`, no `apt`, no driver changes, no system CUDA changes, no Docker daemon changes.
- Docker GPU fallback was probed and rejected for now because `docker run --gpus ...` fails for this user with `nvidia-container-cli: initialization error: nvml error: insufficient permissions`.

## Installed Runtime

Created with:

```bash
mamba create -n kimodo-elf3-rl python=3.11 pip -y
```

Installed GPU PyTorch inside the conda env:

```bash
/home/chengjunhao/miniforge3/envs/kimodo-elf3-rl/bin/python -m pip install \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
```

Installed RL dependencies from `requirements.txt`, using Tsinghua PyPI mirror after the default PyPI connection produced `IncompleteRead` errors:

```bash
/home/chengjunhao/miniforge3/envs/kimodo-elf3-rl/bin/python -m pip install \
  -r /home/chengjunhao/kimodo/research/retarget_g1_to_elf3/rl/requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Runtime size after install:

```text
/home/chengjunhao/miniforge3/envs/kimodo-elf3-rl: 7.6G
/tmp/kimodo-elf3-rl-smoke: 7.8M
```

## Verification

CUDA probe:

```text
torch 2.11.0+cu128
torch_cuda 12.8
cuda_available True
device_count 1
device0 NVIDIA GeForce RTX 5090
tensor_sum 6.0
```

Import probe:

```text
gymnasium, stable_baselines3, mujoco, yaml, onnx, onnxruntime, cv2 all import from kimodo-elf3-rl
```

Real Gymnasium reset/step:

```text
Loaded 102 reference motion clips
reset_obs (385,) float32
step_obs (385,) float32
step_result finite reward, terminated=False, truncated=False
```

SB3 PPO smoke:

```bash
CUDA_VISIBLE_DEVICES=4 /home/chengjunhao/miniforge3/envs/kimodo-elf3-rl/bin/python \
  -m research.retarget_g1_to_elf3.rl.training.train \
  --config /tmp/kimodo-elf3-rl-smoke/config.yaml \
  --curriculum-phase 1 \
  --timesteps 32 \
  --n-gpus 1
```

Result:

```text
total_timesteps reached 32
final model saved to /tmp/kimodo-elf3-rl-smoke/checkpoints/final_model_phase1.zip
VecNormalize stats saved to /tmp/kimodo-elf3-rl-smoke/checkpoints/vec_normalize_phase1.pkl
```

## Code Fixes Made During Smoke

The first PPO smoke exposed real training-path failures:

- `foot_airborne_penalty` could overflow `exp`, producing infinite rewards.
- VecNormalize then saw non-finite values and policy outputs became NaN.
- MuJoCo could report unstable QACC under exploratory actions.

Fixes:

- `env/rewards.py`: bounded exponential foot penalties and finite-clipped total reward.
- `env/elf3_env.py`: sanitizes non-finite actions; detects non-finite MuJoCo state; restores a finite reference state for terminal observations; reports `termination_reason='simulation_unstable'`.
- `requirements.txt`: added `mujoco` and `PyYAML`, which are required in a clean environment.

## Remaining Risk

The 32-step PPO smoke still printed one MuJoCo warning:

```text
WARNING: Nan, Inf or huge value in QACC at DOF 3. The simulation is unstable.
```

The environment now catches this and terminates the episode without poisoning SB3, but it means the current default PD/action/reward settings are not yet motion-quality safe for long training. The next engineering step should be a short rollout audit of `simulation_unstable` frequency by motion category before any 50k+ step run.
