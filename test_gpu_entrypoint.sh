#!/bin/bash
# 检查 entrypoint 和 /workspace 中的隐藏文件

echo "=== Test 1: 查看 entrypoint 脚本 ==="
docker run --rm --runtime=nvidia \
  kimodo:1.0 cat /usr/local/bin/docker-entrypoint

echo ""
echo "=== Test 2: 查看 /workspace 下所有隐藏文件 ==="
docker run --rm --runtime=nvidia \
  kimodo:1.0 find /workspace -maxdepth 2 -name ".*" -not -path "/workspace/.*" 2>/dev/null

echo ""
echo "=== Test 3: 覆盖 entrypoint 测试 ==="
docker run --rm --runtime=nvidia \
  -v $HOME/kimodo:/workspace \
  -w /workspace \
  --entrypoint python \
  kimodo:1.0 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 4: 检查 /workspace 下的 .nvidia 或 cuda 相关文件 ==="
docker run --rm --runtime=nvidia \
  kimodo:1.0 find /workspace -name "*cuda*" -o -name "*nvidia*" -o -name "*.so" 2>/dev/null | head -20
