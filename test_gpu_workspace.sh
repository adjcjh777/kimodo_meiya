#!/bin/bash
# 测试覆盖 /workspace 的不同方式

echo "=== Test 1: 挂载 kimodo 到 /workspace 但不设 -w ==="
docker run --rm --runtime=nvidia \
  -v $HOME/kimodo:/workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 2: 不挂载，只改 WORKDIR ==="
docker run --rm --runtime=nvidia \
  -w /tmp \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 3: 用 --gpus 代替 runtime:nvidia ==="
docker run --rm --gpus all \
  -v $HOME/kimodo:/workspace \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 4: 挂载到 /workspace 但加 --privileged ==="
docker run --rm --runtime=nvidia --privileged \
  -v $HOME/kimodo:/workspace \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
