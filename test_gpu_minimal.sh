#!/bin/bash
# 最简单的 CUDA 测试（之前成功过的命令）
docker run --rm --runtime=nvidia kimodo:1.0 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}')"
