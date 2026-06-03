#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# 设置 Docker Compose 命令（不使用 sudo，因为 Docker 已拉起）
export DOCKER_COMPOSE_CMD="docker compose"

echo "=========================================="
echo "ELF3 RL 训练数据批量生成"
echo "=========================================="
echo ""

# 1. 创建目录结构
echo "[1/6] 创建目录结构..."
mkdir -p research/retarget_g1_to_elf3/generated_elf3_baseline/{locomotion,upper_body,compound,balance}
mkdir -p research/retarget_g1_to_elf3/generated_g1_{locomotion,upper_body,compound,balance}
echo "✓ 目录创建完成"
echo ""

# 2. 生成 locomotion (40 条)
echo "[2/6] 生成 Locomotion 动作 (40 条)..."
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts research/retarget_g1_to_elf3/prompts_locomotion.json \
  --g1-dir research/retarget_g1_to_elf3/generated_g1_locomotion \
  --elf3-dir research/retarget_g1_to_elf3/generated_elf3_locomotion \
  --skip-render --no-zip
echo "✓ Locomotion 生成完成"
echo ""

# 3. 生成 upper_body (30 条)
echo "[3/6] 生成 Upper Body 动作 (30 条)..."
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts research/retarget_g1_to_elf3/prompts_upper_body.json \
  --g1-dir research/retarget_g1_to_elf3/generated_g1_upper_body \
  --elf3-dir research/retarget_g1_to_elf3/generated_elf3_upper_body \
  --skip-render --no-zip
echo "✓ Upper Body 生成完成"
echo ""

# 4. 生成 compound (20 条)
echo "[4/6] 生成 Compound 动作 (20 条)..."
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts research/retarget_g1_to_elf3/prompts_compound.json \
  --g1-dir research/retarget_g1_to_elf3/generated_g1_compound \
  --elf3-dir research/retarget_g1_to_elf3/generated_elf3_compound \
  --skip-render --no-zip
echo "✓ Compound 生成完成"
echo ""

# 5. 生成 balance (10 条)
echo "[5/6] 生成 Balance 动作 (10 条)..."
research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh \
  --prompts research/retarget_g1_to_elf3/prompts_balance.json \
  --g1-dir research/retarget_g1_to_elf3/generated_g1_balance \
  --elf3-dir research/retarget_g1_to_elf3/generated_elf3_balance \
  --skip-render --no-zip
echo "✓ Balance 生成完成"
echo ""

# 6. 移动文件到 baseline 目录
echo "[6/6] 整理文件到 baseline 目录..."

mv research/retarget_g1_to_elf3/generated_elf3_locomotion/*.npz \
   research/retarget_g1_to_elf3/generated_elf3_baseline/locomotion/ 2>/dev/null || true

mv research/retarget_g1_to_elf3/generated_elf3_upper_body/*.npz \
   research/retarget_g1_to_elf3/generated_elf3_baseline/upper_body/ 2>/dev/null || true

mv research/retarget_g1_to_elf3/generated_elf3_compound/*.npz \
   research/retarget_g1_to_elf3/generated_elf3_baseline/compound/ 2>/dev/null || true

mv research/retarget_g1_to_elf3/generated_elf3_balance/*.npz \
   research/retarget_g1_to_elf3/generated_elf3_baseline/balance/ 2>/dev/null || true

echo "✓ 文件整理完成"
echo ""

# 统计结果
echo "=========================================="
echo "生成统计"
echo "=========================================="
for category in locomotion upper_body compound balance; do
  count=$(ls research/retarget_g1_to_elf3/generated_elf3_baseline/${category}/*.npz 2>/dev/null | wc -l)
  echo "  ${category}: ${count} 条"
done
total=$(find research/retarget_g1_to_elf3/generated_elf3_baseline -name "*.npz" | wc -l)
echo "  总计: ${total} 条"
echo ""

# 验证数据完整性
echo "=========================================="
echo "验证数据完整性"
echo "=========================================="
python3 <<'PYEOF'
import numpy as np
from pathlib import Path

baseline_dir = Path("research/retarget_g1_to_elf3/generated_elf3_baseline")

total_valid = 0
total_invalid = 0

for category in ["locomotion", "upper_body", "compound", "balance"]:
    category_dir = baseline_dir / category
    npz_files = list(category_dir.glob("*.npz"))

    print(f"\n{category.upper()} ({len(npz_files)} files)")

    for npz_file in sorted(npz_files):
        try:
            data = np.load(npz_file)
            required_keys = [
                "qpos_elf3", "joint_pos", "joint_vel",
                "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
                "foot_contacts", "fps"
            ]
            missing = [k for k in required_keys if k not in data]

            if missing:
                print(f"  ✗ {npz_file.name}: missing {missing}")
                total_invalid += 1
            else:
                frames = data["qpos_elf3"].shape[0]
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                duration = frames / fps
                print(f"  ✓ {npz_file.name}: {frames} frames, {duration:.1f}s")
                total_valid += 1
        except Exception as e:
            print(f"  ✗ {npz_file.name}: {e}")
            total_invalid += 1

print(f"\n验证结果: {total_valid} 有效, {total_invalid} 无效")
PYEOF

echo ""
echo "=========================================="
echo "批量生成完成！"
echo "=========================================="
echo ""
echo "数据位置: research/retarget_g1_to_elf3/generated_elf3_baseline/"
echo "  ├── locomotion/"
echo "  ├── upper_body/"
echo "  ├── compound/"
echo "  └── balance/"
echo ""
