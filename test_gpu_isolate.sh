#!/bin/bash
# 排查 kimodo 目录中什么文件导致 CUDA 失败

echo "=== Test 1: 挂载到不同路径（不覆盖 /workspace）==="
docker run --rm --runtime=nvidia \
  -v $HOME/kimodo:/data/kimodo \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 2: 挂载空目录 ==="
tmpdir=$(mktemp -d)
docker run --rm --runtime=nvidia \
  -v $tmpdir:/workspace \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
rm -rf $tmpdir

echo ""
echo "=== Test 3: 检查 kimodo 目录中的可疑文件 ==="
echo "--- .env files ---"
find $HOME/kimodo -maxdepth 2 -name ".env*" -o -name "*.cfg" -o -name "*.ini" -o -name "sitecustomize.py" 2>/dev/null
echo "--- ld configs ---"
find $HOME/kimodo -maxdepth 2 -name "*.conf" -o -name "ld.so*" 2>/dev/null
echo "--- hidden files in root ---"
ls -la $HOME/kimodo/.* 2>/dev/null | head -20
