# ELF3 RL 训练数据需求

## 数据总量

**100 条 NPZ 文件**，按动作类别分为 4 个子目录：

```
research/retarget_g1_to_elf3/generated_elf3_baseline/
├── locomotion/       # 步态类 ~40 条
├── upper_body/       # 上肢类 ~30 条
├── compound/         # 复合动作 ~20 条
└── balance/          # 平衡类 ~10 条
```

---

## 数据格式要求

每条 NPZ 必须包含以下字段（由 `retarget_g1_to_elf3_baseline.py` 自动生成）：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `qpos_elf3` | (T, 36) | float32 | ELF3 完整 qpos（7 根 + 29 关节） |
| `joint_pos` | (T, 29) | float32 | 29 个关节角度 |
| `joint_vel` | (T, 29) | float32 | 29 个关节角速度 |
| `body_pos_w` | (T, 30, 3) | float32 | 30 个刚体世界坐标 |
| `body_quat_w` | (T, 30, 4) | float32 | 30 个刚体四元数 |
| `body_lin_vel_w` | (T, 30, 3) | float32 | 刚体线速度 |
| `body_ang_vel_w` | (T, 30, 3) | float32 | 刚体角速度 |
| `foot_contacts` | (T, 4) | bool | 脚部接触（左脚前/后、右脚前/后） |
| `fps` | (1,) | float64 | 帧率（30.0） |
| `joint_names` | (29,) | str | 关节名称列表 |
| `body_names` | (30,) | str | 刚体名称列表 |

RL 环境加载时会把 `foot_contacts` 的 4 个接触点归并为左右脚 2 维观测；源 NPZ 仍应保留 (T, 4)，用于更细粒度的离线质量检查。

**时长建议**：
- 步态类：8~10 秒（240~300 帧，至少 2 个完整步态周期）
- 上肢类：4~6 秒（120~180 帧）
- 复合动作：10~15 秒（300~450 帧）
- 平衡类：6~8 秒（180~240 帧）

---

## 具体动作清单

### 1. locomotion（步态类，40 条）

**基础步态**（已有，保留）：
1. `walk_forward` — 正常向前行走
2. `walk_backward` — 向后行走
3. `side_step_left` — 左侧步
4. `side_step_right` — 右侧步
5. `turn_left` — 左转
6. `turn_right` — 右转

**速度变体**（新增）：
7. `walk_forward_slow` — 慢速前行
8. `walk_forward_fast` — 快速前行
9. `side_step_left_fast` — 快速左侧步
10. `side_step_right_fast` — 快速右侧步

**方向变体**（新增）：
11. `walk_diagonal_left` — 左前斜向行走
12. `walk_diagonal_right` — 右前斜向行走
13. `turn_left_sharp` — 急左转（90°）
14. `turn_right_sharp` — 急右转（90°）

**步幅变体**（新增）：
15. `walk_forward_small_steps` — 小步行走
16. `walk_forward_long_steps` — 大步行走
17. `side_step_left_wide` — 宽幅左侧步
18. `side_step_right_wide` — 宽幅右侧步

**重复生成**（每条生成 2 次，增加随机性）：
19~38. 上述 1~10 各再生成一次（命名加 `_v2` 后缀）

**总计**：40 条

---

### 2. upper_body（上肢类，30 条）

**基础上肢**（已有，保留）：
1. `wave_left_hand` — 左手挥手
2. `wave_right_hand` — 右手挥手
3. `raise_both_arms` — 双臂举起

**单臂变体**（新增）：
4. `point_forward_left` — 左手指前方
5. `point_forward_right` — 右手指前方
6. `point_up_left` — 左手指上方
7. `point_up_right` — 右手指上方
8. `reach_forward_left` — 左手前伸
9. `reach_forward_right` — 右手前伸
10. `reach_sideways_left` — 左手侧伸
11. `reach_sideways_right` — 右手侧伸

**双臂变体**（新增）：
12. `clap_hands` — 拍手
13. `arms_cross` — 双臂交叉
14. `arms_open_wide` — 双臂张开
15. `hands_on_hips` — 双手叉腰
16. `arms_behind_back` — 双手背后

**动态上肢**（新增）：
17. `wave_both_hands` — 双手同时挥手
18. `punch_left` — 左拳出击
19. `punch_right` — 右拳出击
20. `push_forward` — 双手前推

**重复生成**（每条生成 2 次）：
21~30. 上述 1~10 各再生成一次（命名加 `_v2` 后缀）

**总计**：30 条

---

### 3. compound（复合动作，20 条）

**步态 + 上肢**（新增）：
1. `walk_and_wave_left` — 行走同时左手挥手
2. `walk_and_wave_right` — 行走同时右手挥手
3. `walk_and_point` — 行走同时指前方
4. `walk_and_carry` — 行走同时双手捧物（双臂前伸）

**连续动作**（新增）：
5. `squat_multiple` — 连续蹲起（3 次）
6. `wave_sequence` — 连续挥手（左→右→左）
7. `turn_and_wave` — 转身 + 挥手
8. `walk_turn_walk` — 前行→转身→前行

**过渡动作**（新增）：
9. `stand_to_walk` — 站立→行走
10. `walk_to_stop` — 行走→停止
11. `stand_to_squat` — 站立→蹲下
12. `squat_to_stand` — 蹲下→站立

**复杂组合**（新增）：
13. `walk_side_step_walk` — 前行→侧步→前行
14. `turn_squat_turn` — 转身→蹲下→转身
15. `wave_kick_wave` — 挥手→踢腿→挥手
16. `walk_backward_turn` — 后退→转身

**重复生成**（每条生成 2 次）：
17~20. 上述 1~4 各再生成一次（命名加 `_v2` 后缀）

**总计**：20 条

---

### 4. balance（平衡类，10 条）

**单脚站立**（新增）：
1. `single_leg_stand_left` — 左脚单立
2. `single_leg_stand_right` — 右脚单立
3. `single_leg_stand_left_arms_out` — 左脚单立 + 双臂平举
4. `single_leg_stand_right_arms_out` — 右脚单立 + 双臂平举

**踮脚**（新增）：
5. `tiptoe_stand` — 双脚踮脚站立
6. `tiptoe_walk` — 踮脚行走
7. `tiptoe_left` — 左脚踮脚
8. `tiptoe_right` — 右脚踮脚

**重心转移**（新增）：
9. `weight_shift_left_right` — 重心左右转移
10. `lean_forward_backward` — 身体前后倾

**总计**：10 条

---

## 生成方法

### 步骤 1：创建 prompts 文件

为每个类别创建独立的 prompts JSON 文件：

```bash
# locomotion prompts（40 条）
research/retarget_g1_to_elf3/prompts_locomotion.json

# upper_body prompts（30 条）
research/retarget_g1_to_elf3/prompts_upper_body.json

# compound prompts（20 条）
research/retarget_g1_to_elf3/prompts_compound.json

# balance prompts（10 条）
research/retarget_g1_to_elf3/prompts_balance.json
```

每个文件格式与现有 `prompts_10_g1.json` 相同：

```json
[
  {
    "action": "walk_forward_slow",
    "prompt": "a humanoid robot walks forward slowly with careful steps.",
    "duration": 10.0
  },
  ...
]
```

### 步骤 2：批量生成 G1 动作

对每个 prompts 文件运行生成脚本：

```bash
# 生成 locomotion
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts research/retarget_g1_to_elf3/prompts_locomotion.json \
  --g1-dir research/retarget_g1_to_elf3/generated_g1_locomotion \
  --elf3-dir research/retarget_g1_to_elf3/generated_elf3_locomotion \
  --skip-render --no-zip

# 生成 upper_body
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts research/retarget_g1_to_elf3/prompts_upper_body.json \
  --g1-dir research/retarget_g1_to_elf3/generated_g1_upper_body \
  --elf3-dir research/retarget_g1_to_elf3/generated_elf3_upper_body \
  --skip-render --no-zip

# 生成 compound
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts research/retarget_g1_to_elf3/prompts_compound.json \
  --g1-dir research/retarget_g1_to_elf3/generated_g1_compound \
  --elf3-dir research/retarget_g1_to_elf3/generated_elf3_compound \
  --skip-render --no-zip

# 生成 balance
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts research/retarget_g1_to_elf3/prompts_balance.json \
  --g1-dir research/retarget_g1_to_elf3/generated_g1_balance \
  --elf3-dir research/retarget_g1_to_elf3/generated_elf3_balance \
  --skip-render --no-zip
```

### 步骤 3：整理到统一目录

将生成的 ELF3 NPZ 按类别移动到 `generated_elf3_baseline/` 子目录：

```bash
# 创建子目录
mkdir -p research/retarget_g1_to_elf3/generated_elf3_baseline/{locomotion,upper_body,compound,balance}

# 移动文件
mv research/retarget_g1_to_elf3/generated_elf3_locomotion/*.npz \
   research/retarget_g1_to_elf3/generated_elf3_baseline/locomotion/

mv research/retarget_g1_to_elf3/generated_elf3_upper_body/*.npz \
   research/retarget_g1_to_elf3/generated_elf3_baseline/upper_body/

mv research/retarget_g1_to_elf3/generated_elf3_compound/*.npz \
   research/retarget_g1_to_elf3/generated_elf3_baseline/compound/

mv research/retarget_g1_to_elf3/generated_elf3_balance/*.npz \
   research/retarget_g1_to_elf3/generated_elf3_baseline/balance/
```

### 步骤 4：验证数据完整性

运行验证脚本检查所有 NPZ 是否包含必需字段：

```bash
python -c "
import glob
import numpy as np

required_keys = {'qpos_elf3', 'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
                 'body_lin_vel_w', 'body_ang_vel_w', 'foot_contacts', 'fps',
                 'joint_names', 'body_names'}

npz_files = glob.glob('research/retarget_g1_to_elf3/generated_elf3_baseline/**/*.npz', recursive=True)
print(f'Total NPZ files: {len(npz_files)}')

for path in npz_files:
    d = np.load(path, allow_pickle=False)
    missing = required_keys - set(d.files)
    if missing:
        print(f'❌ {path}: missing {missing}')
    else:
        print(f'✓ {path}')
"
```

---

## 数据质量检查

生成完成后，建议抽样检查以下动作：

1. **步态类**：`walk_forward`, `side_step_left`, `turn_right`
   - 检查脚部是否穿地
   - 检查步态是否自然（无抖动）
   - 检查 `foot_contacts` 是否与脚步匹配

2. **上肢类**：`wave_left_hand`, `raise_both_arms`, `point_forward_left`
   - 检查手臂是否穿模（穿过身体）
   - 检查动作幅度是否合理

3. **复合类**：`walk_and_wave_left`, `squat_multiple`
   - 检查动作连贯性
   - 检查是否有不自然的跳变

4. **平衡类**：`single_leg_stand_left`, `tiptoe_stand`
   - 检查重心是否稳定
   - 检查是否有过度倾斜

可以用 `visualize_mujoco_elf3.py` 或 `render_elf3_videos.py` 查看。

---

## 预期时间

- 生成 G1 动作：约 40 分钟（100 条 × 24 秒/条）
- G1→ELF3 重定向：约 20 分钟（100 条 × 12 秒/条）
- 总计：约 1 小时

建议在 `tmux` 或 `screen` 中运行，避免 SSH 断开。
