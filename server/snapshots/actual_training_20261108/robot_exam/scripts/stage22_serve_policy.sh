#!/usr/bin/env bash
# Future authorized inference only. Never run during no-card preparation.
set -euo pipefail
test "${ROBOT_EXAM_MODEL_LOAD_AUTHORIZED:-0}" = 1 || { echo 'Refusing: explicit model/GPU inference authorization required'; exit 2; }
stage_root="${ROBOT_EXAM_STAGE22_ROOT:-/root/autodl-tmp/robot_exam_stage22}"
base_root="${ROBOT_EXAM_BASE_ROOT:-/root/autodl-tmp/robot_exam_training}"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
step="${1:-9999}"
[[ "$step" =~ ^[0-9]+$ ]] || { echo 'Checkpoint label must be numeric'; exit 2; }
adapter="${2:-strict}"
adapter_args=()
case "$adapter" in
  strict) ;;
  bounded_projection_v1) adapter_args=(--project-gripper);;
  *) echo 'Use strict or explicitly versioned bounded_projection_v1'; exit 2;;
esac
training_python="$base_root/openpi/.venv/bin/python"
checkpoint="$stage_root/checkpoints/pi05_original_ur5_lora_v1/multifruit_stage22_batch4_v2/$step"
test -f "$checkpoint/_CHECKPOINT_METADATA"
test -f "$checkpoint/params/_METADATA"
test -f "$checkpoint/assets/original_ur5/norm_stats.json"
exec 9>"$stage_root/training.lock"
flock -n 9 || { echo 'Training or another policy service is active'; exit 2; }
export PYTHONPATH="$project_root/src:$base_root/openpi/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export HF_DATASETS_CACHE="$stage_root/cache/hf_datasets" XDG_CACHE_HOME="$stage_root/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.75
unset JAX_PLATFORMS
export CUDA_VISIBLE_DEVICES=0
attempt="$stage_root/results/serve_${step}_${adapter}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$attempt"
exec "$training_python" "$project_root/scripts/run_training_disk_guard.py" \
  --base-guard "$project_root/scripts/run_resource_guarded.py" \
  --disk-root "$stage_root" --disk-root / --min-free-gib 25 -- \
  --memory-accounting cgroup --max-memory-mib 81920 --timeout 10800 \
  --samples "$attempt/memory.jsonl" --output "$attempt/guard.json" --log "$attempt/serve.log" -- \
  "$training_python" "$project_root/scripts/serve_openpi_original_ur5.py" \
  --openpi-root "$base_root/openpi" --assets "$stage_root/dataset/assets" \
  --checkpoint "$checkpoint" --repo-id local/robot_exam_stage21_train \
  --tokenizer "$base_root/cache/openpi/big_vision/paligemma_tokenizer.model" \
  --lerobot-cache "$stage_root/dataset/lerobot" --port 8000 --wrist-camera \
  --websocket-compression deflate --evaluation-noise-seeds "${adapter_args[@]}" --serve
