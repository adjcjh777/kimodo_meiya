# G1 to ELF3 Retargeting Investigation

## Goal

Retarget Kimodo-generated Unitree G1 29-DOF motion to the Banxing/BXI ELF3 robot.

Source motions:

```text
/home/chengjunhao/kimodo/research/retarget_g1_to_elf3/generated_g1_10s
```

ELF3 assets copied from:

```text
https://github.com/fuqiang-liu/mjlab/tree/elf3_velocity/src/mjlab/asset_zoo/robots/bxi_elf3
```

Local asset path:

```text
research/retarget_g1_to_elf3/assets/bxi_elf3
```

## Confirmed Model Facts

Unitree G1 Kimodo MJCF:

- XML: `kimodo/assets/skeletons/g1skel34/xml/g1.xml`
- `nq = 36`
- qpos layout: `7 root + 29 hinge joints`
- root body: pelvis

ELF3 MJCF:

- XML: `research/retarget_g1_to_elf3/assets/bxi_elf3/xmls/elf3.xml`
- `nq = 36`
- qpos layout: `7 root + 29 hinge joints`
- root body: torso_link
- home/root height in constants: `z = 1.1`

The DOF count matches, but direct playback is still not valid without retargeting because the
root body, link lengths, rest pose, joint names, joint limits, and possible joint signs differ.

## First-Pass Semantic Mapping

The file `joint_map_g1_to_elf3.json` records the first-pass semantic mapping from G1 qpos columns
to ELF3 qpos columns. Signs and offsets are initially conservative placeholders and must be
validated in MuJoCo one joint at a time.

## Important Root Difference

Generated G1 motions in `generated_g1_baseline` have pelvis-root z around `0.74-0.78` for normal standing
and walking motions. ELF3's free root is the torso, with a nominal home z of `1.1`. A direct root
copy would place the ELF3 torso too low. The baseline retarget should therefore apply a root height
offset or recompute root height from ELF3 foot contact geometry.

## Recommended Milestones

1. Build a direct semantic qpos remap baseline:
   - Convert each G1 NPZ to G1 qpos36.
   - Map the 29 G1 hinge values into ELF3 hinge order.
   - Apply a root height offset for ELF3 torso.
   - Clamp to ELF3 joint ranges.
   - Save `qpos_elf3` NPZ.
2. Add an ELF3 MuJoCo visualizer and inspect every generated action.
3. Validate sign and offset per joint by posing one joint at a time.
4. If direct qpos remap is insufficient, move to task-space retargeting:
   - root/torso orientation
   - feet positions and orientations
   - hands positions and orientations
   - knee and elbow bend direction
   - foot contact timing

## Baseline Commands

### One-command workflow

Run the full current workflow from the repository root:

```bash
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh
```

This will:

1. Start the Docker `text-encoder` service.
2. Generate ten 10-second Unitree G1 motions from `research/retarget_g1_to_elf3/prompts_10_g1.json`.
3. Keep the generated G1 NPZ and CSV files under `research/retarget_g1_to_elf3/generated_g1_10s`.
4. Retarget those G1 files to ELF3 NPZ and CSV files under `research/retarget_g1_to_elf3/generated_elf3_10s`.
5. Render ELF3 MP4 videos under `research/retarget_g1_to_elf3/videos_elf3_10s`.
6. Write `research/retarget_g1_to_elf3/videos_elf3_10s.zip` for download.

Useful variants:

```bash
# Retarget and render again without regenerating Kimodo motions.
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh --skip-generate

# Render videos again from existing ELF3 files with a farther camera.
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --skip-generate \
  --skip-retarget \
  --camera-distance-scale 1.6

# Use docker without sudo if the current user has Docker permission.
DOCKER_COMPOSE_CMD="docker compose" \
  research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh
```

### 1. Generate 10-second Unitree G1 motions in Docker

Start the text encoder service once:

```bash
sudo docker compose up -d text-encoder
```

Then generate one 10-second, 30 Hz G1 motion per prompt:

```bash
sudo docker compose run --rm --no-deps demo \
  python research/retarget_g1_to_elf3/generate_g1_prompt_batch.py
```

The G1 outputs are written to:

```text
research/retarget_g1_to_elf3/generated_g1_10s
```

Each action keeps both formats:

```text
<action>.npz
<action>.csv
```

### 2. Retarget G1 motions to ELF3

Generate baseline ELF3 qpos files from the 10-second G1 motions:

```bash
python research/retarget_g1_to_elf3/retarget_g1_to_elf3_baseline.py \
  research/retarget_g1_to_elf3/generated_g1_10s \
  -o research/retarget_g1_to_elf3/generated_elf3_10s
```

Each ELF3 action keeps both formats:

```text
<action>.npz  # contains qpos_elf3 and metadata
<action>.csv  # raw 36-column ELF3 MuJoCo qpos
```

The ELF3 CSV columns follow MuJoCo `model.qpos` order:

```text
root_x, root_y, root_z,
root_quat_w, root_quat_x, root_quat_y, root_quat_z,
waist_y, waist_x, waist_z,
l_hip_y, l_hip_x, l_hip_z, l_knee_y, l_ankle_y, l_ankle_x,
r_hip_y, r_hip_x, r_hip_z, r_knee_y, r_ankle_y, r_ankle_x,
l_shoulder_y, l_shoulder_x, l_shoulder_z, l_elbow_y, l_wrist_x, l_wrist_y, l_wrist_z,
r_shoulder_y, r_shoulder_x, r_shoulder_z, r_elbow_y, r_wrist_x, r_wrist_y, r_wrist_z
```

### 3. Visualize or render videos

Visualize one original G1 file interactively:

```bash
python research/retarget_g1_to_elf3/visualize_mujoco_g1.py \
  research/retarget_g1_to_elf3/generated_g1_10s/walk_forward.npz
```

Visualize one retargeted file interactively:

```bash
python research/retarget_g1_to_elf3/visualize_mujoco_elf3.py \
  research/retarget_g1_to_elf3/generated_elf3_10s/walk_forward.npz
```

Batch render MP4 videos with a farther auto camera:

```bash
python research/retarget_g1_to_elf3/render_elf3_videos.py \
  --input research/retarget_g1_to_elf3/generated_elf3_10s \
  --output-dir research/retarget_g1_to_elf3/videos_elf3_10s \
  --width 1280 \
  --height 720 \
  --fps 30
```

## Baseline Result Snapshot

The first baseline produced 10 ELF3 NPZ files under:

```text
research/retarget_g1_to_elf3/generated_elf3_baseline
```

All 10 files load into the ELF3 MJCF and run `mj_forward` without non-finite body poses.
Observed root z ranges:

```text
normal standing/walking/root-turning actions: roughly 1.03-1.12
squat_down: roughly 0.59-1.10
```

Joint limit clipping occurred in a few actions:

```text
squat_down: waist_y_joint, r_knee_y_joint
turn_left: waist_x_joint
wave_left_hand: l_wrist_z_joint
wave_right_hand: r_wrist_z_joint
```

This is expected for the direct qpos baseline and should be inspected visually before choosing
whether to tune signs/offsets, reduce action amplitude, or switch specific limbs to task-space IK.
