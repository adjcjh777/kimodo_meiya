#!/bin/bash
# 进一步排查挂载问题

echo "=== Test 1: 挂载到 /workspace2 ==="
docker run --rm --runtime=nvidia \
  -v $HOME/kimodo:/workspace2 \
  -w /workspace2 \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 2: 查看镜像中 /workspace 内容 ==="
docker run --rm --runtime=nvidia \
  kimodo:1.0 ls -la /workspace/

echo ""
echo "=== Test 3: 查看镜像中 /workspace/.cache ==="
docker run --rm --runtime=nvidia \
  kimodo:1.0 ls -laR /workspace/.cache/ 2>/dev/null || echo "no .cache"

echo ""
echo "=== Test 4: 挂载但排除 .cache（用 tmpfs）==="
docker run --rm --runtime=nvidia \
  -v $HOME/kimodo:/workspace \
  --tmpfs /workspace/.cache \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
