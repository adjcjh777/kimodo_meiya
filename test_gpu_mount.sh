#!/bin/bash
# 逐个测试挂载目录

echo "=== Mount 1: 仅挂载 kimodo ==="
docker run --rm --runtime=nvidia \
  -v $HOME/kimodo:/workspace \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Mount 2: 仅挂载 huggingface cache ==="
docker run --rm --runtime=nvidia \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Mount 3: 仅挂载 modelscope ==="
docker run --rm --runtime=nvidia \
  -v $HOME/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct:/workspace/llama-3-8b-instruct:ro \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Mount 4: kimodo + huggingface ==="
docker run --rm --runtime=nvidia \
  -v $HOME/kimodo:/workspace \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
