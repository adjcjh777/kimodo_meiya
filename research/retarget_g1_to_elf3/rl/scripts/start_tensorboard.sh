#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chengjunhao/kimodo"
PYTHON_ENV="/home/chengjunhao/miniforge3/envs/kimodo-elf3-rl"
LOGDIR="${REPO_ROOT}/research/retarget_g1_to_elf3/rl/runs/elf3_rl"
HOST="${TENSORBOARD_HOST:-0.0.0.0}"
PORT="${TENSORBOARD_PORT:-6006}"

if command -v ss >/dev/null 2>&1; then
  while ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; do
    PORT=$((PORT + 1))
  done
fi

echo "TensorBoard logdir: ${LOGDIR}"
echo "TensorBoard URL: http://${HOST}:${PORT}/"

exec "${PYTHON_ENV}/bin/tensorboard" \
  --logdir "${LOGDIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --reload_interval 15
