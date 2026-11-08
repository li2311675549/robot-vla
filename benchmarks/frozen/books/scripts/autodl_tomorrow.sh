#!/usr/bin/env bash
# Deploy alongside the offline verifier and frozen official manifest on AutoDL.
set -euo pipefail
usage() {
  printf '%s\n' 'Usage: bash autodl_tomorrow.sh --preflight-only | --authorized-training' \
    'The training option is an explicit authorization for the bounded 30-step smoke only.'
}
case "${1:-}" in
  --help) usage; exit 0 ;;
  --preflight-only|--authorized-training) mode="$1" ;;
  *) usage; exit 2 ;;
esac
if [[ $# != 1 ]]; then usage; exit 2; fi
task_root="${ROBOT_EXAM_BUNDLE:-/root/autodl-tmp/robot_exam_training}"
source "$task_root/server_paths_v2.sh"
prep_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_tag="$(date -u +%Y%m%dT%H%M%S_%N)"
run_dir="$task_root/runs/gpu_start_$run_tag"
mkdir -p "$task_root/runs"
mkdir "$run_dir"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
unset JAX_PLATFORMS
"$ROBOT_EXAM_PYTHON" "$prep_root/verify_official_pi05_assets.py" \
  --manifest "$prep_root/pi05_official_assets_20261030.json" \
  --cache "$OPENPI_DATA_HOME" --output "$run_dir/assets.json"
ROBOT_EXAM_SERVER_PATHS="$task_root/server_paths_v2.sh" bash "$task_root/next_gpu_preflight_v2.sh"
"$ROBOT_EXAM_PYTHON" "$task_root/robot_exam/scripts/validate_openpi_ur5_mapping.py" \
  --openpi-root "$ROBOT_EXAM_OPENPI_ROOT" --manifest "$task_root/dataset_manifest" \
  --lerobot-cache "$ROBOT_EXAM_LEROBOT_CACHE" --repo-id local/robot_exam_dual_release_train \
  --wrist-camera --converted-only --output "$run_dir/native_train.json"
if [[ "$mode" == --preflight-only ]]; then
  printf 'GPU/data/assets preflight finished; no model training started. Results: %s\n' "$run_dir"
  exit 0
fi
export ROBOT_EXAM_TRAIN_AUTHORIZED=1
export ROBOT_EXAM_TRAIN_SETTINGS="$task_root/robot_exam/configs/pi05_dual_fruit_smoke_v1.json"
export ROBOT_EXAM_TRAIN_RESULT="$run_dir/training_result.json"
bash "$task_root/robot_exam/scripts/autodl_train_ur5_lora.sh" \
  --repo-id local/robot_exam_dual_release_train \
  --checkpoints "$task_root/checkpoints/smoke_$run_tag" 2>&1 | tee "$run_dir/training.log"
printf 'Bounded smoke attempt finished. Review actual metrics before any pilot: %s\n' "$run_dir"
