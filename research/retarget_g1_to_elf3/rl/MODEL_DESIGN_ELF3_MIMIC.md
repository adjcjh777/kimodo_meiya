# ELF3-Mimic Model Design

This design replaces the current single `MlpPolicy` tracker with a structured
motion-tracking stack inspired by BeyondMimic-style humanoid control, while
staying scoped to the current ELF3 RL environment.

## Current Problem

The current policy is too flat:

- Observation is a 586-dimensional vector with mixed proprioception, reference
  horizon, body state, action history, and contacts.
- Actor and critic both use the same simple MLP over that flat vector.
- The action is a residual joint target around the reference joint pose.
- The reward is mostly joint/reference tracking with penalties.

The latest audit showed a more fundamental issue: `action=0`, which means
direct PD tracking of the reference joint pose, still falls within 4-59 steps on
the phase-1 upper-body clips. A larger network alone will not fix that. The new
model must learn balance and physically feasible tracking, not only memorize
joint references.

## Design Principle

Use a two-level architecture:

1. `ELF3-Mimic Tracker`: robust online RL policy for physically tracking
   retargeted motions.
2. `ELF3-Mimic Composer`: optional offline state-action diffusion model after
   the tracker is stable enough to generate successful rollouts.

The tracker is the immediate target. The diffusion composer should not be built
until reference replay and RL tracking can survive meaningful horizons.

## Tracker V1: Structured Asymmetric Actor-Critic

### Actor Inputs

The actor should not consume an opaque 586-dim vector as one blob. Split it into
semantic streams:

| Stream | Source | Dim | Purpose |
| --- | --- | ---: | --- |
| Proprioception | root height, local root velocity, root angular velocity, gravity projection, joint pos/vel | 68 | balance and body state |
| Phase | sin/cos phase, clip/skill embedding | 2 + embed | motion timing |
| Reference horizon | 4-8 future frames of target data | current 268, later body targets | upcoming motion intent |
| Action history | last two actions | 58 | damping and smoothness |
| Body/contact state | body pos/vel, foot contacts | 182 | whole-body feedback |

The current reference horizon should be changed over time from raw joint/root
tracking to anchor-relative body targets:

- anchor body: pelvis or torso
- tracked bodies: torso, head, hands, feet, knees, elbows
- express non-anchor body targets in the anchor yaw-aligned local frame
- keep height information, but avoid forcing absolute global XY drift

This matches the practical lesson from BeyondMimic: track the style in a local
body/objective frame instead of over-constraining absolute world poses.

### Actor Network

Use specialized encoders and late fusion:

```text
proprio_encoder:      MLP(68 -> 256 -> 256)
ref_temporal_encoder: Transformer/GRU over K reference tokens -> 256
history_encoder:      MLP(58 -> 128)
body_encoder:         MLP(182 -> 256)
skill_encoder:        Embedding(policy_id/clip/category -> 64)

fusion: concat -> MLP(896 -> 512 -> 512)
heads:
  lower_body_balance_head -> 17 dims
  upper_body_tracking_head -> 12 dims
  log_std_head or learned log_std per action group
```

Why split lower and upper body:

- upper-body clips are the first failing target
- legs and waist must prioritize balance, not blindly mimic upper-body
  retarget residuals
- upper-body should track hands/arms while the lower body stabilizes stance

### Action Parameterization

Current action:

```text
target_joint_pos = ref_joint_pos + action * action_scale
```

This makes the reference pose the center of the action. If the reference is not
physically feasible under MuJoCo, the policy starts from a bad target.

Tracker V1 should move to grouped residual setpoints:

```text
lower_body_target = nominal_stance + lower_body_residual
upper_body_target = ref_upper_body + upper_body_residual
waist_target      = blended(nominal_waist, ref_waist, alpha)
```

Group-specific action scales:

| Group | Scale |
| --- | ---: |
| hip/knee/ankle | 0.15-0.25 |
| waist | 0.10-0.20 |
| shoulder/elbow/wrist | 0.25-0.40 |

This makes upper-body expression possible without asking the lower body to
follow dynamically bad retargeted poses.

### Critic Inputs

Use an asymmetric critic. The actor only sees deployable observations; the
critic can receive privileged training-only features:

- full reference body poses relative to anchor
- body-wise pose/twist tracking errors
- foot-bottom heights and contact state
- root pitch/roll/height margins to fall thresholds
- per-clip phase and category
- domain randomization parameters
- termination/failure flags from recent steps

The critic network:

```text
actor_feature     -> 512
privileged_feature -> MLP(priv_obs -> 512)
critic_fusion     -> MLP(1024 -> 512 -> 256 -> value)
```

SB3 does not natively separate actor and critic observations cleanly, so this is
an implementation step, not a config-only change. A minimal implementation can
start with a custom feature extractor over the current Box observation, then
move to a Dict observation with actor/critic branches.

## Reward Redesign

Current reward is too joint-centric for an unstable retargeted humanoid. The
new reward should prioritize physically stable task-space tracking:

### Positive Task Rewards

- anchor height/upright reward
- anchor-relative body position reward for selected target bodies
- body orientation reward for torso/hands/feet
- body linear/angular velocity reward for target bodies
- upper-body end-effector reward for hands and elbows

### Minimal Regularization

- action rate
- joint soft-limit penalty
- self-collision penalty if contact force is available
- energy penalty with a lower coefficient during early curriculum

### Termination

Keep fall termination, but log exact reason:

- root height below threshold
- pitch threshold
- roll threshold
- simulation unstable
- contact/foot deviation if added later

## Training Curriculum

Do not start with all phase-1 upper-body clips directly.

### Stage 0: Reference Feasibility Gate

Before RL training, every phase-1 clip must pass:

```text
action=0 replay survival >= 150 steps
no non-finite qpos/qvel/qacc
root pitch/roll within limits
foot-bottom height within expected band
```

If this fails, repair reference/root/contact data or relax lower-body tracking.

### Stage 1: Balance Warm Start

Train only stance and upper-body static posture:

- fixed phase start
- no domain randomization
- lower-body nominal stance action center
- upper-body target is either neutral or a single slow arm pose
- success metric: `eval/termination_fall < 2/10`, `eval/mean_length > 300`

### Stage 2: Single-Clip Upper Body

Train one clip at a time:

- `raise_both_arms`
- then `wave_right_hand`
- then `wave_left_hand`
- then pointing motions

Use failure-rate adaptive sampling over phase bins. Sample phases that lead to
falls more often, but keep a uniform mixture to avoid forgetting.

### Stage 3: Multi-Clip Upper Body

Enable all five phase-1 upper-body clips. Keep low domain randomization.

### Stage 4: Full Upper Body Policy

Enable the full upper-body policy route and phase randomization.

### Stage 5: Domain Randomization

Only after stable replay/tracking:

- gravity scale
- friction
- PD gain scale
- root velocity perturbation
- action delay

## Diffusion Composer: Later Stage

After the tracker can complete clips, collect rollouts:

```text
dataset item:
  history: previous H state-action pairs
  future: next T state-action trajectory
  condition: skill id, goal velocity/waypoint/clip target
```

Model:

```text
state_action_diffuser:
  transformer_decoder_layers: 6
  heads: 4
  embed_dim: 512
  denoise_steps: 20
  history_steps: 4-8
  future_steps: 16-32
  action_loss_horizon: 8
```

This composer can later support joystick, waypoint, and obstacle guidance. It
should not be used as the low-level controller until the tracker produces stable
expert trajectories.

## Implementation Roadmap

1. Add a reference replay audit script and make it a required pre-training gate.
2. Change action center from full reference joints to grouped nominal/reference
   targets.
3. Add structured actor feature extraction over the existing 586-dim observation.
4. Add exact fall reason logging for root height, pitch, and roll.
5. Add adaptive phase/clip sampling based on failure statistics.
6. Move to asymmetric critic with privileged observations.
7. Only after stable tracking, add offline diffusion distillation.

## First Model to Build

Build `ELF3MimicTrackerV1`, not the diffusion composer.

Target acceptance criteria:

```text
upper_body phase1:
  eval/termination_fall <= 2/10
  eval/mean_length >= 300
  rollout/ep_rew_mean improving over 1M steps
  action=0 or nominal replay does not explode before 150 steps
```

If these fail, the issue is still reference feasibility/control design, not
model capacity.
