#!/bin/bash
# 测试 GPU 直接加载模型
# 用法: bash test_gpu_load.sh

docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v $HOME/kimodo:/workspace \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  -v $HOME/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct:/workspace/llama-3-8b-instruct:ro \
  -w /workspace \
  kimodo:1.0 python -c "
from kimodo.model.llm2vec.llm2vec import LLM2Vec
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('Loading model directly to GPU...')
model = LLM2Vec.from_pretrained(
    base_model_name_or_path='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
    peft_model_name_or_path='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
)
print(f'GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
print('SUCCESS')
"
