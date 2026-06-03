# ELF3 RL Multi-Policy Implementation Plan

## Context

ELF3 RL will train a policy set of three independently trained Skill Policies instead of one universal PPO policy. The design is recorded in `DESIGN.md` and ADR `docs/adr/0001-three-skill-policies-with-explicit-routing.md`.

The current codebase already has a working single-policy Gymnasium/SB3 path and a real user-local runtime at `/home/chengjunhao/miniforge3/envs/kimodo-elf3-rl`. That runtime smoke proved the old 385-dimensional observation path can run a 32-step PPO smoke, and it also added important NaN/Inf and MuJoCo instability guards. The multi-policy implementation must preserve those runtime fixes while changing the policy decomposition, observation schema, and training/evaluation entry points.

## Target Architecture

| Area | Target |
| --- | --- |
| Policy shape | Three independent Skill Policies: `locomotion_policy`, `posture_balance_policy`, `upper_body_policy` |
| Routing | Explicit `motion stem -> policy_id` in `config/policy_route.yaml` |
| Observation | Unified `observation(586)` for every policy |
| Action | Unified `action(29)` residual joint target offset |
| Reference horizon | Offsets `[0, 3, 6, 12]`, using root-relative local displacement |
| Training data | 100 classified clips only; root-level legacy baseline clips excluded from training |
| Eval sets | `benchmark_eval` excludes legacy clips; `debug_eval` may include them |
| Runtime safety | Preserve finite reward clipping and `simulation_unstable` termination handling |

## Current Implementation Gap

| Current code | Required change |
| --- | --- |
| `ELF3TrackingEnv` emits `(385,)` observations | Emit `(586,)` observations with Reference Horizon |
| `ReferenceMotionManager` loads root-level NPZ as `uncategorized` | Training path must load classified clips only; debug eval may opt into root clips |
| Training CLI has no `--policy-id` | Add `--policy-id` and policy-specific config selection |
| `config/default.yaml` has one global `reward`, `network`, and `total_timesteps` | Add `policy_route.yaml`, `policy_configs.yaml`, and `eval_sets.yaml` |
| `train.py` writes `final_model_phase{phase}` under one checkpoint dir | Write per-policy checkpoints and VecNormalize stats |
| `evaluate.py` evaluates all clips for one model | Evaluate clips selected by `--policy-id` and `--eval-set` |
| `export_model.py` defaults to `obs_dim=385` | Default to `obs_dim=586` and export per-policy artifacts |

## Execution Plan

### Phase 0: Freeze Runtime Baseline

Goal: avoid losing the other Codex runtime work.

Tasks:

- Keep `/home/chengjunhao/miniforge3/envs/kimodo-elf3-rl` as the reference runtime for smoke tests.
- Preserve `env/rewards.py` finite reward clipping.
- Preserve `env/elf3_env.py` action sanitization and `simulation_unstable` termination handling.
- Treat `CODEX_RL_RUNTIME_SMOKE_2026-06-03.md` as the current runtime baseline.

Validation:

```bash
/home/chengjunhao/miniforge3/envs/kimodo-elf3-rl/bin/python -m py_compile \
  env/reference_motion.py env/rewards.py env/elf3_env.py training/train.py eval/evaluate.py export/export_model.py
```

### Phase 1: Add Policy Configuration Files

Goal: make policy boundaries explicit before changing environment behavior.

Files:

- `config/policy_route.yaml`
- `config/policy_configs.yaml`
- `config/eval_sets.yaml`

Tasks:

- Create `policy_route.yaml` with all 100 classified motion stems mapped to exactly one policy.
- Route all locomotion directory clips and locomotion-heavy compound clips to `locomotion_policy`.
- Route balance, squat, tiptoe, kick, and posture-transition clips to `posture_balance_policy`.
- Route upper-body directory clips and pure upper-body compound sequences to `upper_body_policy`.
- Add per-policy `net_arch`, `reward_weights`, `total_timesteps`, and phase ratios to `policy_configs.yaml`.
- Add `benchmark_eval` and `debug_eval` clip lists to `eval_sets.yaml`.

Validation:

- Route covers exactly 100 training clips.
- No root-level legacy baseline stem appears in training routes.
- Every route points to one of the three known policy ids.
- No training motion stem appears twice.

### Phase 2: Refactor Reference Motion Loading

Goal: make clip loading route-aware and eval-set-aware.

Files:

- `env/reference_motion.py`

Tasks:

- Add loading modes for `train`, `benchmark_eval`, and `debug_eval`.
- Load classified subdirectories by default for training.
- Make root-level NPZ loading opt-in and only for debug eval.
- Add clip filtering by explicit motion stem list.
- Add helper methods for route coverage validation.
- Add a reference lookup that supports frame offsets for horizon construction.

Validation:

```bash
# Expected training count
python - <<'PY'
from pathlib import Path
print(len(list(Path("../generated_elf3_baseline").glob("*/*.npz"))))
PY
```

Expected: `100`.

### Phase 3: Implement 586-Dimensional Observation

Goal: replace single-frame reference observation with Reference Horizon while keeping one interface for all policies.

Files:

- `env/elf3_env.py`

Tasks:

- Add config-driven `horizon_offsets: [0, 3, 6, 12]`.
- Replace the 67-dimensional single reference block with 4 reference points.
- For each horizon point, include:
  - `ref_joint_pos(29)`
  - `ref_joint_vel(29)`
  - `ref_root_delta_local(3)`
  - `ref_root_quat(4)`
  - `ref_foot_contact(2)`
- Compute `ref_root_delta_local` as:

```python
rotate_inv(current_root_quat, ref_root_pos - current_root_pos)
```

- Update `observation_space.shape` and assertions to `(586,)`.
- Keep action sanitization, finite-state detection, and simulation recovery behavior.

Validation:

- Reset/step smoke for each `policy_id` returns `(586,)`.
- No non-finite values in observation, reward, or action.
- `debug_eval` can still render a root-level legacy baseline clip when explicitly requested.

### Phase 4: Add Policy-Aware Training

Goal: make training one policy explicit and reproducible.

Files:

- `training/train.py`
- `training/callbacks.py` if callback metrics need policy ids

Tasks:

- Add `--policy-id`.
- Load `policy_configs.yaml` and select the requested policy.
- Use per-policy `net_arch`, reward weights, total timesteps, and curriculum split.
- Filter the environment's motion set to that policy's routed clips.
- Save outputs under `research/retarget_g1_to_elf3/rl/checkpoints/elf3_rl/{policy_id}/`.
- Save VecNormalize stats per policy and phase.
- Keep single-policy smoke mode through `--timesteps` for fast checks.

Validation:

Run 32-step PPO smoke for each policy in the existing conda runtime:

```bash
CUDA_VISIBLE_DEVICES=4 /home/chengjunhao/miniforge3/envs/kimodo-elf3-rl/bin/python \
  -m research.retarget_g1_to_elf3.rl.training.train \
  --policy-id upper_body_policy \
  --timesteps 32 \
  --n-gpus 1
```

Repeat for `posture_balance_policy` and `locomotion_policy`.

### Phase 5: Add Policy-Aware Evaluation and Export

Goal: keep evaluation and deployment aligned with route decisions.

Files:

- `eval/evaluate.py`
- `export/export_model.py`

Tasks:

- Add `--policy-id` and `--eval-set` to evaluation.
- Evaluate only clips assigned to the selected policy for `benchmark_eval`.
- Allow explicit legacy baseline clips only for `debug_eval`.
- Include `termination_reason` and `simulation_unstable` rate in eval summaries.
- Change export default `obs_dim` to `586`.
- Export one ONNX and one NPZ per policy.

Validation:

- `benchmark_eval` excludes root-level legacy baseline clips.
- `debug_eval --clip walk_forward` can intentionally use legacy baseline only when requested.
- ONNX verification passes against SB3 deterministic `predict()` for each policy.

### Phase 6: Rollout Stability Audit

Goal: measure whether the current PD/action/reward settings are safe before long training.

Tasks:

- Run short random-policy or partially trained rollouts per policy.
- Count `simulation_unstable`, `fall`, `tracking_fail`, and `timeout`.
- Report instability by clip and phase.
- If instability is high, tune action scale, PD gain randomization, reward clipping, or termination thresholds before 50k+ training.

Minimum gate before long training:

- `simulation_unstable` episodes are rare and traceable.
- Observations and rewards remain finite for all tested clips.
- No policy has systematic immediate failure on its phase1 motion set.

### Phase 7: Long Training Order

Train the easiest policy first to validate the full pipeline before consuming large GPU time:

1. `upper_body_policy` at small budget, then full `20M`.
2. `posture_balance_policy` at small budget, then full `35M`.
3. `locomotion_policy` at small budget, then full `50M`.

Each policy must pass `benchmark_eval` and at least one `debug_eval` render before moving to the next long run.

## Initial Route Intent

The first route should follow these boundaries:

- `locomotion_policy`: all `locomotion/` clips plus `stand_to_walk`, `walk_to_stop`, `walk_turn_walk`, `walk_side_step_walk`, `walk_backward_turn`, `turn_and_wave`, and `walk_and_*` compound clips.
- `posture_balance_policy`: all `balance/` clips plus `squat_multiple`, `stand_to_squat`, `squat_to_stand`, `turn_squat_turn`, and `wave_kick_wave`.
- `upper_body_policy`: all `upper_body/` clips plus `wave_sequence`.

This route is intentionally not directory-based. It is based on the dominant balance and coordination demand of each motion.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The existing smoke path is 385-dimensional | Require new 586-dimensional reset/step and PPO smoke before long training |
| MuJoCo instability appears under exploratory actions | Keep `simulation_unstable` termination and add rollout audit gate |
| Route table drifts from generated data | Add route coverage validation before training |
| Policy-specific VecNormalize makes cross-policy comparison noisy | Keep per-policy benchmark reports separate and compare within policy first |
| Future shared-backbone migration becomes attractive | Preserve unified `observation(586) -> action(29)` interface across all policies |

## Non-Goals for This Implementation Pass

- Do not train an automatic policy classifier or learned gate.
- Do not merge the three policies into a shared backbone.
- Do not include root-level legacy baseline clips in training.
- Do not remove runtime safety fixes added during the SB3 smoke test.
- Do not change system CUDA, drivers, Docker GPU setup, or other users' processes.
