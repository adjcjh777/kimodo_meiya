# Kimodo G1 到 ELF3 动作生成与重定向使用手册

这份文档面向在自己 A6000 服务器上部署的同事。目标是从 Kimodo Docker 环境开始，
批量生成宇树 G1 动作，并重定向成半醒/BXI ELF3 可用的动作文件。

当前工作流的核心输出是动作数据：

```text
G1 原始动作:  research/retarget_g1_to_elf3/generated_g1_10s/*.npz
G1 原始 CSV:   research/retarget_g1_to_elf3/generated_g1_10s/*.csv
ELF3 动作:     research/retarget_g1_to_elf3/generated_elf3_10s/*.npz
ELF3 CSV:      research/retarget_g1_to_elf3/generated_elf3_10s/*.csv
```

默认总脚本会额外录制视频和打包 zip。你如果只需要 `.npz` / `.csv` 动作文件，
请使用 `--skip-render --no-zip`。

当前工作流会在生成和重定向阶段直接写出同事/MJLab 风格的 `.npz` 字段，
不需要再额外跑格式转换脚本。`align_npz_to_mjlab_schema.py` 只保留给历史 `.npz`
手动补齐字段使用，默认工作流不会调用它。

## 1. 先明确权限和交付方式

GitLab 链接只能提供代码，不能直接访问 5090 服务器上的 Docker 镜像、模型缓存和 checkpoint。

如果你还没有 5090 服务器账号，目前只能在自己的 A6000 机器上从互联网下载依赖。
等你拿到 5090 服务器账号并获得文件读取权限后，可以直接从 5090 服务器复制以下大文件，
这样能省掉大量公网下载时间。

5090 服务器当前依赖位置：

```text
仓库目录:
/home/chengjunhao/kimodo

Kimodo G1 checkpoint:
/home/chengjunhao/kimodo/checkpoints/Kimodo-G1-RP-v1
大小约 1.1G

Hugging Face 缓存:
/home/chengjunhao/.cache/huggingface
大小约 6.2G

ModelScope Llama-3-8B-Instruct 缓存:
/home/chengjunhao/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct
大小约 31G

kimodo-viser 外部仓库副本:
/home/chengjunhao/kimodo/kimodo-viser
大小约 937M

Docker 镜像:
kimodo:1.0
镜像 ID: 15c757a88e15
大小约 29.8G
Docker 实际存储由 Docker 管理，通常在 /var/lib/docker；建议用 docker save/docker load 迁移。
```

### 可选项 A：在 5090 服务器上使用已经拉起的服务

适用场景：你已经拿到 5090 服务器账号，希望直接复用这台机器上的 `kimodo:1.0` 镜像、
checkpoint、模型缓存，以及已经启动的 `text-encoder` 服务。

这种方式需要服务器侧授权。至少需要：

1. SSH 登录 5090 服务器的账号。
2. GitLab 仓库或 `/home/chengjunhao/kimodo` 的读取权限。
3. 输出目录写权限，用于写入生成的 `.npz` / `.csv`。
4. Docker daemon 权限，才能查看/复用/启动容器。

如果由管理员授权 Docker，常见方式是把你的 Linux 用户加入 `docker` 组：

```bash
sudo usermod -aG docker <your_user>
```

执行后需要重新登录 SSH，新的组权限才会生效。

注意：`docker` 组权限很大，基本等价于给了服务器高权限能力。只有在服务器负责人确认可以授权时，
才应该加入 `docker` 组。

你登录 5090 后可以检查当前服务：

```bash
cd /home/chengjunhao/kimodo
docker ps | grep -E 'text-encoder|demo|kimodo'
curl http://localhost:9550/
```

如果 `text-encoder` 已经是 healthy 状态，可以直接跑工作流。因为当前 `docker-compose.yaml`
里用了 `${HOME}` 挂载 Hugging Face 和 ModelScope 缓存，所以你用自己的 Linux 用户运行时，
必须显式让 compose 使用 `/home/chengjunhao` 作为 `HOME`，否则会去找你自己 home 下面的空缓存。

推荐命令：

```bash
cd /home/chengjunhao/kimodo

DOCKER_COMPOSE_CMD="env HOME=/home/chengjunhao docker compose" \
  research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh --skip-render --no-zip
```

如果 Docker 仍然需要 sudo：

```bash
cd /home/chengjunhao/kimodo

DOCKER_COMPOSE_CMD="sudo env HOME=/home/chengjunhao docker compose" \
  research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh --skip-render --no-zip
```

如果不想授予 Docker 权限，则目前不能让同事完整自助运行这条工作流。只能采用其中一种方式：

- 由 `chengjunhao` 账号代跑工作流。
- 以后额外封装一个受控的 Web/API/队列服务，只暴露 prompt 提交和结果下载，不暴露 Docker。

### 可选项 B：在自己的 A6000 机器上拉起自己的服务

适用场景：你不使用 5090 服务器上已经拉起的容器，也不需要服务器侧给 Docker 权限；
你在自己的 A6000 机器上独立部署一套 Kimodo Docker 服务。

这种方式不需要 5090 服务器授权，但需要你自己准备：

1. GitLab 仓库访问权限。
2. 自己机器上的 Docker 和 NVIDIA Container Toolkit。
3. 自己机器上的 Docker 权限，或使用 `sudo docker ...`。
4. `kimodo-viser/`。
5. `checkpoints/`、Hugging Face 缓存、ModelScope/Llama 缓存。
6. `kimodo:1.0` 镜像，或者从 Dockerfile 自己构建。

如果还没有 5090 账号，就只能从互联网下载这些大文件。下载和构建时间主要花在：

- NVIDIA PyTorch 基础镜像。
- PyTorch CUDA wheel。
- Python 依赖和 `kimodo-viser`。
- Kimodo G1 checkpoint。
- LLM2Vec / Llama-3-8B-Instruct 相关模型。

在自己的 A6000 机器上独立部署时，按下面顺序走：

```bash
git clone <gitlab-url> kimodo
cd kimodo
git checkout kimodo-init-5090

git clone https://github.com/nv-tlabs/kimodo-viser.git
```

单卡 A6000 需要先把 `docker-compose.yaml` 中两个服务的 GPU 都改为 0：

```text
text-encoder: CUDA_VISIBLE_DEVICES=0
demo:         CUDA_VISIBLE_DEVICES=0
```

然后构建或导入镜像：

```bash
sudo docker compose build
```

准备好 checkpoint 和模型缓存后，运行只输出 `.npz` / `.csv` 的工作流：

```bash
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh --skip-render --no-zip
```

这种模式的优点是权限边界清晰，不需要访问 5090 服务器的 Docker daemon。
缺点是首次部署会经历完整的镜像构建和模型下载，耗时较长，也更容易遇到网络、Hugging Face token、
模型授权和缓存路径问题。

## 2. A6000 机器前置要求

请先确认 A6000 机器满足这些条件：

```bash
nvidia-smi
docker --version
docker compose version
```

需要具备：

- NVIDIA 驱动可用，`nvidia-smi` 能看到 A6000。
- Docker Engine 和 Docker Compose plugin 可用。
- NVIDIA Container Toolkit 可用，容器里能访问 GPU。
- 当前用户能运行 Docker。没有 docker 组权限时，需要用 `sudo docker ...`。
- 磁盘空间建议至少预留 80G；从互联网构建和下载模型时建议预留更多。

快速测试 Docker GPU：

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

如果这个命令失败，先解决 Docker GPU 权限或 NVIDIA Container Toolkit，再继续。

## 3. 获取代码

从 GitLab 拉仓库并切到当前工作分支：

```bash
git clone <gitlab-url> kimodo
cd kimodo
git checkout kimodo-init-5090
```

当前 Dockerfile 需要本地存在 `kimodo-viser/`。这个目录不在主仓库里，需要单独准备。

从互联网拉：

```bash
git clone https://github.com/nv-tlabs/kimodo-viser.git
```

如果你已经有 5090 服务器账号，可以从 5090 服务器复制：

```bash
rsync -aP <user>@<5090-server>:/home/chengjunhao/kimodo/kimodo-viser ./kimodo-viser
```

没有 `kimodo-viser/` 时，`docker compose build` 会在 `COPY kimodo-viser /workspace/kimodo-viser`
这一步失败。

## 4. 准备模型和缓存

### 推荐方式：从 5090 服务器复制

等你拿到 5090 账号和读取权限后，优先复制这些目录到你自己的 A6000 机器。
以下命令需要在 A6000 机器的 `kimodo` 仓库根目录执行：

```bash
rsync -aP <user>@<5090-server>:/home/chengjunhao/kimodo/checkpoints ./checkpoints

mkdir -p ~/.cache
rsync -aP <user>@<5090-server>:/home/chengjunhao/.cache/huggingface ~/.cache/

mkdir -p ~/.cache/modelscope/hub/models/LLM-Research
rsync -aP \
  <user>@<5090-server>:/home/chengjunhao/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct \
  ~/.cache/modelscope/hub/models/LLM-Research/
```

复制 Docker 镜像有两种方式。

方式 A：在 5090 服务器导出，然后拷贝 tar 包：

```bash
docker save kimodo:1.0 | gzip > /tmp/kimodo_1.0.tar.gz
scp <user>@<5090-server>:/tmp/kimodo_1.0.tar.gz .
gunzip -c kimodo_1.0.tar.gz | sudo docker load
```

方式 B：如果 SSH 和 Docker 权限都可用，可以直接流式复制：

```bash
ssh <user>@<5090-server> 'docker save kimodo:1.0' | sudo docker load
```

如果远端 Docker 需要 sudo，就把远端命令改成：

```bash
ssh <user>@<5090-server> 'sudo docker save kimodo:1.0' | sudo docker load
```

### 备选方式：从互联网下载

如果暂时没有 5090 账号，只能联网准备依赖：

- Docker build 会拉 `nvcr.io/nvidia/pytorch:24.10-py3`，体积很大。
- Dockerfile 会重新安装 PyTorch CUDA 12.8 wheel，下载时间较长。
- `docker_requirements.txt` 会安装 Kimodo、MotionCorrection、viser 等依赖。
- Kimodo G1 checkpoint 来自 Hugging Face：`nvidia/Kimodo-G1-RP-v1`。
- 文本编码器会用 LLM2Vec：
  - `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`
  - `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`
- Llama-3-8B-Instruct 相关权重可能涉及 Hugging Face/ModelScope 访问、账号授权或网络问题。

注意：当前 `docker-compose.yaml` 为了复用本机缓存，设置了：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
LOCAL_CACHE=true
CHECKPOINT_DIR=/workspace/checkpoints
```

这意味着默认更适合“本地已经有 checkpoint 和模型缓存”的模式。
如果你选择从互联网首次下载，需要临时关闭离线模式，或者先把模型缓存准备好再启动服务。

从互联网首次下载时，建议先在宿主机登录 Hugging Face：

```bash
hf auth login
```

然后临时修改 `docker-compose.yaml`：

```yaml
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0
LOCAL_CACHE=false
```

首次下载跑通后，再改回离线/本地缓存模式，避免后续每次运行都访问外网。
如果 Hugging Face 下载 Llama/LLM2Vec 失败，通常是网络、token、模型授权或 gated model
访问权限问题；这时优先等 5090 账号开通后直接复制缓存。

## 5. A6000 单卡机器的 GPU 配置

当前 `docker-compose.yaml` 是按 5090 双卡环境写的：

```text
text-encoder: CUDA_VISIBLE_DEVICES=1
demo:         CUDA_VISIBLE_DEVICES=0
```

如果你的 A6000 机器只有一张 GPU，必须把两个服务都改成 `0`，否则 `text-encoder`
可能因为找不到 GPU 1 而启动失败。

修改位置在 `docker-compose.yaml`：

```yaml
text-encoder:
  environment:
    - CUDA_VISIBLE_DEVICES=0

demo:
  environment:
    - CUDA_VISIBLE_DEVICES=0
```

如果你的机器有多张 GPU，可以按实际情况分配。例如 text encoder 用 1、motion generation 用 0。
A6000 48G 显存通常可以先尝试两个服务都放在同一张卡上；如果 OOM，再拆到两张卡。

## 6. 构建 Docker 镜像

如果你已经通过 `docker load` 导入了 `kimodo:1.0`，可以先跳过 build，直接检查：

```bash
sudo docker images | grep kimodo
```

如果没有镜像，执行构建：

```bash
sudo docker compose build
```

首次构建会很慢，主要耗时在：

- 拉 NVIDIA PyTorch 基础镜像。
- 下载 PyTorch CUDA wheel。
- 安装 Python 依赖。
- 编译或安装 MotionCorrection。
- 安装本地 `kimodo-viser`。

构建完成后应该能看到：

```bash
sudo docker images | grep kimodo
```

预期有：

```text
kimodo   1.0
```

## 7. 只生成 npz/csv 的推荐命令

同事目前只需要动作文件，不需要视频，所以推荐运行：

```bash
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh --skip-render --no-zip
```

这条命令会做：

1. 启动 `text-encoder` Docker 服务。
2. 读取 `research/retarget_g1_to_elf3/prompts_10_g1.json`。
3. 通过 Kimodo CLI 批量生成 10 条 G1 动作。
4. 每条动作默认 10 秒、30Hz、300 帧。
5. 保存带 MJLab 字段和真实 G1 关节/刚体名的 G1 `.npz`，同时保留 `.csv`。
6. 将 G1 动作重定向到 ELF3。
7. 保存带 MJLab 字段和真实 ELF3 关节/刚体名的 ELF3 `.npz`，同时保留 `.csv`。
8. 不录制视频，不生成 zip。

输出位置：

```text
G1:
research/retarget_g1_to_elf3/generated_g1_10s

ELF3:
research/retarget_g1_to_elf3/generated_elf3_10s
```

如果当前用户没有 docker 组权限，脚本默认会使用：

```text
sudo docker compose
```

如果当前用户已经有 Docker 权限，可以这样跑：

```bash
DOCKER_COMPOSE_CMD="docker compose" \
  research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh --skip-render --no-zip
```

建议在 `tmux` 或 `screen` 里跑，避免 SSH 断开导致任务中断：

```bash
tmux new -s kimodo_g1_elf3
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh --skip-render --no-zip
```

## 8. Prompt 文件怎么改

Prompt 文件：

```text
research/retarget_g1_to_elf3/prompts_10_g1.json
```

每条动作格式：

```json
{
  "action": "walk_forward",
  "prompt": "a humanoid robot walks forward at a steady pace.",
  "duration": 10.0
}
```

字段含义：

- `action`：输出文件名，建议只用英文小写、数字和下划线。
- `prompt`：英文动作描述。
- `duration`：动作时长，当前默认要求是 `10.0` 秒。

如果只想生成自己的 prompt 文件，可以指定路径：

```bash
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts path/to/prompts.json \
  --skip-render \
  --no-zip
```

## 9. 分步运行命令

如果完整脚本失败，建议分步排查。

### 9.1 启动 text-encoder

```bash
sudo docker compose up -d text-encoder
```

检查状态：

```bash
sudo docker compose ps
curl http://localhost:9550/
```

查看日志：

```bash
sudo docker logs -f text-encoder
```

text encoder 首次加载 LLM2Vec/Llama 相关模型会比较慢，当前脚本最多等待 600 秒。

### 9.2 生成 G1 动作

```bash
sudo docker compose run --rm --no-deps demo \
  python research/retarget_g1_to_elf3/generate_g1_prompt_batch.py --fps 30
```

只打印命令、不真正生成：

```bash
python research/retarget_g1_to_elf3/generate_g1_prompt_batch.py --dry-run
```

### 9.3 G1 重定向到 ELF3

```bash
python research/retarget_g1_to_elf3/retarget_g1_to_elf3_baseline.py \
  research/retarget_g1_to_elf3/generated_g1_10s \
  -o research/retarget_g1_to_elf3/generated_elf3_10s
```

### 9.4 可选：录制视频

默认总脚本会录制视频；如果只要 npz/csv，不需要跑这一步。

需要检查动作效果时再运行：

```bash
python research/retarget_g1_to_elf3/render_elf3_videos.py \
  --input research/retarget_g1_to_elf3/generated_elf3_10s \
  --output-dir research/retarget_g1_to_elf3/videos_elf3_10s \
  --width 1280 \
  --height 720 \
  --fps 30
```

视频输出：

```text
research/retarget_g1_to_elf3/videos_elf3_10s/*.mp4
```

录制脚本默认会给 ELF3 场景动态加入 MuJoCo 默认风格的深色 skybox、深色棋盘地面和固定补光。
这样 MP4 里能看到脚相对地面的高度、穿地和滑步问题。
如果只想录原始 MJCF 场景，可以加：

```bash
python research/retarget_g1_to_elf3/render_elf3_videos.py \
  --input research/retarget_g1_to_elf3/generated_elf3_10s \
  --output-dir research/retarget_g1_to_elf3/videos_elf3_raw \
  --no-ground
```

调试时只想快速看前几帧，可以加 `--max-frames 30`。

重定向脚本默认还会做脚底贴地校正：根据 ELF3 左右脚 collision capsule 的最低点，逐帧调整 root z，
让最低脚底落在地面高度 `z=0`。如果只想保留原始 root z，可以在重定向阶段加：

```bash
python research/retarget_g1_to_elf3/retarget_g1_to_elf3_baseline.py \
  research/retarget_g1_to_elf3/generated_g1_10s \
  -o research/retarget_g1_to_elf3/generated_elf3_10s \
  --no-foot-ground-correction
```

## 10. 查看动作

查看原始 G1：

```bash
python research/retarget_g1_to_elf3/visualize_mujoco_g1.py \
  research/retarget_g1_to_elf3/generated_g1_10s/walk_forward.npz
```

查看重定向后的 ELF3：

```bash
python research/retarget_g1_to_elf3/visualize_mujoco_elf3.py \
  research/retarget_g1_to_elf3/generated_elf3_10s/walk_forward.npz
```

MuJoCo viewer 常用交互：

```text
左键拖动: 旋转视角
右键拖动: 平移视角
滚轮:     缩放
Space:    暂停/继续
ESC:      退出
```

如果服务器没有图形界面，优先用 `render_elf3_videos.py` 离屏录制 MP4。

## 11. 输出文件说明

G1 输出：

```text
research/retarget_g1_to_elf3/generated_g1_10s/<action>.npz
research/retarget_g1_to_elf3/generated_g1_10s/<action>.csv
```

ELF3 输出：

```text
research/retarget_g1_to_elf3/generated_elf3_10s/<action>.npz
research/retarget_g1_to_elf3/generated_elf3_10s/<action>.csv
```

G1 和 ELF3 `.npz` 都会直接包含同事/MJLab 风格字段：

```text
fps:             [1]，帧率，例如 30.0
joint_pos:       [T, 29]，29 个关节位置
joint_vel:       [T, 29]，29 个关节速度
body_pos_w:      [T, 30, 3]，去掉 world 后的 30 个刚体世界坐标
body_quat_w:     [T, 30, 4]，去掉 world 后的 30 个刚体世界四元数
body_lin_vel_w:  [T, 30, 3]，刚体世界线速度
body_ang_vel_w:  [T, 30, 3]，刚体世界角速度
joint_names:     [29]，MJCF 里的真实关节名，顺序与 joint_pos 对齐
body_names:      [30]，MJCF 里的真实刚体名，顺序与 body_pos_w/body_quat_w 对齐
```

G1 `.npz` 额外保留：

```text
qpos_g1:          [T, 36]，G1 MuJoCo qpos
qpos_g1_columns:  [36]，G1 qpos 列名
qpos_columns:     [36]，与 qpos_g1_columns 相同，便于通用读取
```

ELF3 `.npz` 额外保留：

```text
qpos_elf3:          [T, 36]，ELF3 MuJoCo qpos
qpos_g1:            [T, 36]，来源 G1 qpos
qpos_elf3_columns:  [36]，ELF3 qpos 列名
qpos_columns:       [36]，与 qpos_elf3_columns 相同，便于通用读取
source_file:        来源 G1 文件
joint_map:          使用的 G1->ELF3 关节映射
```

ELF3 CSV 是 36 列 MuJoCo qpos，列顺序：

```text
root_x, root_y, root_z,
root_quat_w, root_quat_x, root_quat_y, root_quat_z,
waist_y, waist_x, waist_z,
l_hip_y, l_hip_x, l_hip_z, l_knee_y, l_ankle_y, l_ankle_x,
r_hip_y, r_hip_x, r_hip_z, r_knee_y, r_ankle_y, r_ankle_x,
l_shoulder_y, l_shoulder_x, l_shoulder_z, l_elbow_y, l_wrist_x, l_wrist_y, l_wrist_z,
r_shoulder_y, r_shoulder_x, r_shoulder_z, r_elbow_y, r_wrist_x, r_wrist_y, r_wrist_z
```

如果手上已经有旧格式 `.npz`，可以手动补齐字段：

```bash
python research/retarget_g1_to_elf3/align_npz_to_mjlab_schema.py \
  research/retarget_g1_to_elf3/generated_elf3_10s \
  --robot elf3 \
  --fps 30
```

这一步只用于历史数据迁移；默认 `run_g1_to_elf3_workflow.sh` 不会调用它。

## 12. 常见问题和踩坑记录

### 12.1 `ModuleNotFoundError: No module named 'torch'`

这是在宿主机 base/conda 环境里直接跑生成脚本导致的。Kimodo 生成动作需要 PyTorch。
不要用普通 base 环境生成动作，使用 Docker：

```bash
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh --skip-render --no-zip
```

宿主机 Python 主要用于重定向和渲染，不建议在宿主机直接跑 Kimodo 生成。

### 12.2 `omtrack` 虚拟环境

`omtrack` 是另一个项目的虚拟环境，不要用它跑 Kimodo。

### 12.3 A6000 单卡找不到 GPU 1

如果日志里出现 CUDA device 相关错误，先检查 `docker-compose.yaml` 里的：

```text
CUDA_VISIBLE_DEVICES=1
```

单卡机器改成：

```text
CUDA_VISIBLE_DEVICES=0
```

`text-encoder` 和 `demo` 两个服务都要检查。

### 12.4 `text-encoder` 一直不 healthy

检查日志：

```bash
sudo docker logs -f text-encoder
```

常见原因：

- 模型缓存缺失，但 compose 开了离线模式。
- Hugging Face 或 ModelScope 权限/网络问题。
- GPU 号配置错误。
- 显存不足。
- 9550 端口被占用。

### 12.5 checkpoint 缺失

如果缺少：

```text
checkpoints/Kimodo-G1-RP-v1/config.yaml
checkpoints/Kimodo-G1-RP-v1/model.safetensors
```

生成阶段会找不到 Kimodo G1 模型。优先从 5090 服务器复制 `checkpoints/`。

### 12.6 `kimodo-viser` 缺失

Docker build 如果报：

```text
COPY kimodo-viser /workspace/kimodo-viser
```

说明仓库根目录没有 `kimodo-viser/`。解决：

```bash
git clone https://github.com/nv-tlabs/kimodo-viser.git
```

或者从 5090 服务器复制 `/home/chengjunhao/kimodo/kimodo-viser`。

### 12.7 端口冲突

`text-encoder` 默认占用：

```text
9550
```

`demo` 默认占用：

```text
7860
```

如果 7860 冲突，可以临时改：

```bash
SERVER_PORT=7861 sudo docker compose up -d demo
```

如果 9550 冲突，需要改 `docker-compose.yaml` 里的端口映射和 `TEXT_ENCODER_URL`。

### 12.8 只想重新重定向，不想重新生成

已有 G1 输出时：

```bash
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --skip-generate \
  --skip-render \
  --no-zip
```

### 12.9 只想重新录视频

已有 ELF3 输出时：

```bash
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --skip-generate \
  --skip-retarget
```

### 12.10 Docker 权限问题

如果出现：

```text
permission denied while trying to connect to the Docker daemon socket
```

可以先用 `sudo docker ...`。如果希望免 sudo，需要管理员把当前用户加入 docker 组。
注意：docker 组权限很大，接近 root 权限，只给可信用户。

## 13. 当前重定向方法的限制

当前 G1 到 ELF3 是第一版直接语义 qpos 映射：

```text
research/retarget_g1_to_elf3/joint_map_g1_to_elf3.json
```

它会：

- 将 G1 29 个 hinge qpos 按语义映射到 ELF3。
- 给 ELF3 root z 加初始高度偏移。
- 按 ELF3 joint range 做 clamp。

这不是完整的 task-space IK 重定向。后续如果发现脚滑、手脚空间位置不准、膝肘方向不自然，
需要继续做 foot/hand/root 的 task-space retargeting。
