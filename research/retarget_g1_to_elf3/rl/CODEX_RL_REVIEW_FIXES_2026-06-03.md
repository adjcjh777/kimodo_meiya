# CODEX RL Review Fixes 2026-06-03

## 范围

本次只修改 `research/retarget_g1_to_elf3/rl` 目录。用户给出的远端资产页 `http://172.18.6.8:8929/robot/mjlab/-/tree/master/src/mjlab/asset_zoo` 当前返回 GitLab 登录页，因此代码校验以本地已拉取的 ELF3 资产为准：

- XML: `research/retarget_g1_to_elf3/assets/bxi_elf3/xmls/elf3_complie.xml`
- 模型实测: `nq=36, nv=35, nu=29, nbody=31`
- NPZ 实测: `qpos_elf3=(T,36)`, `joint_pos=(T,29)`, `foot_contacts=(T,4)`

## 关键问题与修复

1. `ELF3TrackingEnv.step()` 和 `_get_obs()` 调用 `get_reference(self.phase)`，但参考动作管理器实际需要 clip。已改为 `get_reference(self.current_clip, phase)`，并让 `reset(options={"clip": clip})` 真正生效。

2. 环境没有初始化 `action_scale`，首次 step 会直接报错。已从 `config/default.yaml` 的 `action.scale` 读取。

3. qvel 维度错误。原代码把 qpos 当 qvel 模板，得到 36 维 qvel；真实自由根模型是 `nv=35`。已改为用 `mujoco.mj_differentiatePos` 生成 `(T,35)` qvel，并保留无 XML 时的后备算法。

4. 相位推进单位错误。原逻辑 `control_dt / (frames * sim_dt)` 会让 8 秒动作十几步就循环完。已改为 `control_dt / clip.duration`，其中 `control_dt = dt * substeps`。

5. body 索引错位。NPZ 的 `body_pos_w/body_names` 不含 world body，但原环境用含 world 的 MuJoCo body id 索引，末端奖励和脚奖励会偏一个 body。已改为维护去掉 world 后的 body-data index。

6. 脚接触语义不一致。源 NPZ 是 4 个脚底接触点，观测空间设计是左右脚 2 维。已在加载时归并为左右脚接触，同时保留文档说明源数据仍是 `(T,4)`。

7. 当前 body velocity 取错列。MuJoCo `data.cvel` 前 3 维是角速度，后 3 维是线速度；原环境把角速度当线速度。已改为 `data.cvel[1:31, 3:6]`。

8. reset 会在随机化时修改原始参考帧 view。已复制 qpos/qvel 后再加扰动，避免污染 `MotionClip`。

9. 平滑奖励永远接近 0。原 step 在算 reward 前已经把 `last_action` 覆盖为当前 action。已显式保存 `previous_action` 用于 reward，同时下一帧观测仍记录当前 action。

10. reward/跟踪误差相位滞后一拍。控制目标使用 step 开始相位，仿真后 reward 和 tracking error 改为对齐 step 结束后的新相位参考。

11. 训练脚本读取不存在的配置键。原代码读 `training.n_envs/checkpoint_freq/eval_freq`，YAML 实际是 `training.num_envs` 和 `logging.*`。已修正，并按并行环境数换算 SB3 callback 频率。

12. eval callback 使用训练 env 原地评估，并且 `done` 是数组时会崩。已改为单独 eval env，并在评估前同步 `VecNormalize.obs_rms`。

13. 导出脚本仍默认 341 维。已改为 385 维，并 clamp ONNX deterministic action 到 `[-1, 1]`，匹配 SB3 `predict()` 的 Box action 行为。

14. 文档存在旧接口和错误命令。已修正 `DESIGN.md`、`IMPLEMENTATION_PLAN.md`、`DATA_REQUIREMENTS.md` 中的 341/203 维、`--output-g1/--output-elf3`、`render_demo.py`、`get_reference(phase)` 等残留。

15. `rl` 目录缺依赖清单。已新增 `requirements.txt`，列出 Gymnasium、Stable-Baselines3、TensorBoard、ONNX、OpenCV 等 RL/导出依赖。

## 修改文件

- `env/elf3_env.py`
- `env/reference_motion.py`
- `env/rewards.py`
- `env/__init__.py`
- `training/train.py`
- `training/callbacks.py`
- `training/gpu_selector.py`
- `eval/evaluate.py`
- `export/export_model.py`
- `config/default.yaml`
- `DESIGN.md`
- `IMPLEMENTATION_PLAN.md`
- `DATA_REQUIREMENTS.md`
- `requirements.txt`

## 验证

已通过：

```bash
python -m py_compile __init__.py env/__init__.py env/reference_motion.py env/rewards.py env/elf3_env.py training/train.py training/gpu_selector.py training/callbacks.py eval/evaluate.py export/export_model.py
```

参考动作加载 smoke test：

```text
Loaded 102 reference motion clips:
  locomotion: 40 clips
  upper_body: 30 clips
  compound: 20 clips
  balance: 10 clips
clip locomotion side_step_left (300, 36) (300, 35) (300, 2)
ref (36,) (35,) (2,)
```

在未安装 `gymnasium` 的当前环境中，用临时 Gymnasium stub 覆盖 MuJoCo/NPZ/reset/step 路径：

```text
reset (385,) balance single_leg_stand_left_arms_out 0.03
step (385,) 0.3870581325108367 False False 0.011546379228892302 0.00375
```

未完整运行 SB3 训练，因为当前 Python 环境缺少 `gymnasium`、`stable_baselines3`、`torch`、`onnx`、`onnxruntime`。依赖已记录在 `requirements.txt`，但本次没有修改系统环境或执行 pip 安装。
