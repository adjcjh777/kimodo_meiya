#!/bin/bash
# 不设 CUDA_VISIBLE_DEVICES，直接用所有 GPU
# 用法: bash test_gpu_load3.sh

docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v $HOME/kimodo:/workspace \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  -v $HOME/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct:/workspace/llama-3-8b-instruct:ro \
  -w /workspace \
  kimodo:1.0 python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}:', torch.cuda.get_device_name(i))
    print('Loading small tensor to cuda:1...')
    t = torch.tensor([1.0]).cuda(1)
    print(f'cuda:1 works: {t}')
    print('SUCCESS')
else:
    print('FAILED: CUDA not available')
"
