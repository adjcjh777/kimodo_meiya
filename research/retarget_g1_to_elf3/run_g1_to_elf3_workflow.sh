#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DOCKER_COMPOSE_CMD="${DOCKER_COMPOSE_CMD:-sudo docker compose}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROMPTS_FILE="${PROMPTS_FILE:-research/retarget_g1_to_elf3/prompts_10_g1.json}"
G1_DIR="${G1_DIR:-research/retarget_g1_to_elf3/generated_g1_10s}"
ELF3_DIR="${ELF3_DIR:-research/retarget_g1_to_elf3/generated_elf3_10s}"
VIDEO_DIR="${VIDEO_DIR:-research/retarget_g1_to_elf3/videos_elf3_10s}"

G1_MODEL="${G1_MODEL:-kimodo-g1-rp-v1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
SEED="${SEED:-42}"

WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
CAMERA="${CAMERA:-auto}"
CAMERA_DISTANCE_SCALE="${CAMERA_DISTANCE_SCALE:-1.35}"

TEXT_ENCODER_HEALTH_URL="${TEXT_ENCODER_HEALTH_URL:-http://localhost:9550/}"
TEXT_ENCODER_WAIT_SECONDS="${TEXT_ENCODER_WAIT_SECONDS:-600}"

SKIP_GENERATE=0
SKIP_RETARGET=0
SKIP_RENDER=0
ZIP_VIDEOS=1

usage() {
  cat <<'EOF'
Usage:
  research/retarget_g1_to_elf3/run_g1_to_elf3_workflow.sh [options]

Runs the current G1-to-ELF3 workflow:
  1. Generate 10-second Unitree G1 motions in Docker.
  2. Retarget G1 NPZ files to ELF3 NPZ and CSV.
  3. Render ELF3 videos with a farther auto camera.

Options:
  --skip-generate              Reuse existing G1 outputs.
  --skip-retarget              Reuse existing ELF3 outputs.
  --skip-render                Do not render videos.
  --no-zip                     Do not zip the rendered video directory.
  --prompts PATH               Prompt JSON file.
  --g1-dir DIR                 G1 output/input directory.
  --elf3-dir DIR               ELF3 output/input directory.
  --video-dir DIR              MP4 output directory.
  --model NAME                 Kimodo G1 model name.
  --diffusion-steps N          Diffusion steps for generation.
  --seed N                     Seed for generation.
  --width N                    Render width.
  --height N                   Render height.
  --fps N                      Render FPS.
  --camera NAME                MJCF camera name, or "auto".
  --camera-distance-scale X    Auto camera distance scale.
  --docker-compose-cmd CMD     Docker compose command, e.g. "docker compose".
  --python PATH                Host Python used for retarget/render.
  -h, --help                   Show this help.

Environment variables with the same uppercase names can also override defaults,
for example G1_DIR=... or DOCKER_COMPOSE_CMD="docker compose".
EOF
}

require_value() {
  local opt="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    echo "Missing value for ${opt}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-generate)
      SKIP_GENERATE=1
      shift
      ;;
    --skip-retarget)
      SKIP_RETARGET=1
      shift
      ;;
    --skip-render)
      SKIP_RENDER=1
      shift
      ;;
    --no-zip)
      ZIP_VIDEOS=0
      shift
      ;;
    --prompts)
      require_value "$1" "${2:-}"
      PROMPTS_FILE="$2"
      shift 2
      ;;
    --g1-dir)
      require_value "$1" "${2:-}"
      G1_DIR="$2"
      shift 2
      ;;
    --elf3-dir)
      require_value "$1" "${2:-}"
      ELF3_DIR="$2"
      shift 2
      ;;
    --video-dir)
      require_value "$1" "${2:-}"
      VIDEO_DIR="$2"
      shift 2
      ;;
    --model)
      require_value "$1" "${2:-}"
      G1_MODEL="$2"
      shift 2
      ;;
    --diffusion-steps)
      require_value "$1" "${2:-}"
      DIFFUSION_STEPS="$2"
      shift 2
      ;;
    --seed)
      require_value "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --width)
      require_value "$1" "${2:-}"
      WIDTH="$2"
      shift 2
      ;;
    --height)
      require_value "$1" "${2:-}"
      HEIGHT="$2"
      shift 2
      ;;
    --fps)
      require_value "$1" "${2:-}"
      FPS="$2"
      shift 2
      ;;
    --camera)
      require_value "$1" "${2:-}"
      CAMERA="$2"
      shift 2
      ;;
    --camera-distance-scale)
      require_value "$1" "${2:-}"
      CAMERA_DISTANCE_SCALE="$2"
      shift 2
      ;;
    --docker-compose-cmd)
      require_value "$1" "${2:-}"
      DOCKER_COMPOSE_CMD="$2"
      shift 2
      ;;
    --python)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

to_container_path() {
  local path="$1"
  if [[ "${path}" == "${REPO_ROOT}" ]]; then
    printf '%s\n' "/workspace"
  elif [[ "${path}" == "${REPO_ROOT}/"* ]]; then
    printf '/workspace/%s\n' "${path#"${REPO_ROOT}/"}"
  else
    printf '%s\n' "${path}"
  fi
}

docker_compose() {
  local -a compose_parts
  read -r -a compose_parts <<< "${DOCKER_COMPOSE_CMD}"
  "${compose_parts[@]}" "$@"
}

wait_for_text_encoder() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is not available on host; skipping text-encoder health wait."
    return
  fi

  local deadline=$((SECONDS + TEXT_ENCODER_WAIT_SECONDS))
  until curl -fsS "${TEXT_ENCODER_HEALTH_URL}" >/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${TEXT_ENCODER_HEALTH_URL}" >&2
      exit 1
    fi
    sleep 5
  done
}

zip_video_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo "Video directory does not exist, skip zip: ${dir}"
    return
  fi
  if ! command -v zip >/dev/null 2>&1; then
    echo "zip is not available on host; videos are still in ${dir}"
    return
  fi

  local parent
  local base
  parent="$(cd "$(dirname "${dir}")" && pwd)"
  base="$(basename "${dir}")"
  (
    cd "${parent}"
    zip -qr "${base}.zip" "${base}"
  )
  echo "saved ${parent}/${base}.zip"
}

echo "Workflow configuration:"
echo "  prompts: ${PROMPTS_FILE}"
echo "  G1 output: ${G1_DIR}"
echo "  ELF3 output: ${ELF3_DIR}"
echo "  video output: ${VIDEO_DIR}"
echo "  render: ${WIDTH}x${HEIGHT} @ ${FPS} fps, camera=${CAMERA}, distance_scale=${CAMERA_DISTANCE_SCALE}"

if [[ "${SKIP_GENERATE}" -eq 0 ]]; then
  echo
  echo "==> Starting Docker text-encoder"
  docker_compose up -d text-encoder
  wait_for_text_encoder

  echo
  echo "==> Generating Unitree G1 motions"
  docker_compose run --rm --no-deps demo \
    python research/retarget_g1_to_elf3/generate_g1_prompt_batch.py \
      --prompts "$(to_container_path "${PROMPTS_FILE}")" \
      --output-dir "$(to_container_path "${G1_DIR}")" \
      --model "${G1_MODEL}" \
      --diffusion-steps "${DIFFUSION_STEPS}" \
      --seed "${SEED}"
else
  echo
  echo "==> Skipping G1 generation"
fi

if [[ "${SKIP_RETARGET}" -eq 0 ]]; then
  echo
  echo "==> Retargeting G1 motions to ELF3"
  "${PYTHON_BIN}" research/retarget_g1_to_elf3/retarget_g1_to_elf3_baseline.py \
    "${G1_DIR}" \
    -o "${ELF3_DIR}"
else
  echo
  echo "==> Skipping ELF3 retargeting"
fi

if [[ "${SKIP_RENDER}" -eq 0 ]]; then
  echo
  echo "==> Rendering ELF3 videos"
  "${PYTHON_BIN}" research/retarget_g1_to_elf3/render_elf3_videos.py \
    --input "${ELF3_DIR}" \
    --output-dir "${VIDEO_DIR}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --camera "${CAMERA}" \
    --camera-distance-scale "${CAMERA_DISTANCE_SCALE}"

  if [[ "${ZIP_VIDEOS}" -eq 1 ]]; then
    zip_video_dir "${VIDEO_DIR}"
  fi
else
  echo
  echo "==> Skipping video rendering"
fi

echo
echo "Workflow complete."
