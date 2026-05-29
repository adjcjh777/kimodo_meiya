#!/bin/bash
# 逐步测试，找出哪个参数导致 CUDA 失败

echo "=== Test 1: +NVIDIA_VISIBLE_DEVICES=all ==="
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 2: +volume mounts ==="
docker run --rm --runtime=nvidia \
  -v $HOME/kimodo:/workspace \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  -v $HOME/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct:/workspace/llama-3-8b-instruct:ro \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 3: +env vars ==="
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/workspace/.cache/huggingface \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Test 4: all together ==="
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v $HOME/kimodo:/workspace \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  -v $HOME/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct:/workspace/llama-3-8b-instruct:ro \
  -w /workspace \
  kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
