#!/usr/bin/env bash
# Run only after explicit user authorization, on the intended training machine.
set -euo pipefail
: "${ROBOT_EXAM_TRAIN_AUTHORIZED:?Set to 1 only after explicit training authorization}"
test "$ROBOT_EXAM_TRAIN_AUTHORIZED" = 1
: "${ROBOT_EXAM_OPENPI_ROOT:?Local pinned OpenPI source root required}"
: "${ROBOT_EXAM_ASSETS:?Local train-only original_ur5 statistics directory required}"
: "${ROBOT_EXAM_WEIGHTS:?Local explicitly authorized pi05_base params directory required}"
: "${ROBOT_EXAM_PYTHON:?Python from the prepared OpenPI environment required}"
: "${ROBOT_EXAM_TOKENIZER:?Already downloaded local tokenizer required}"
: "${ROBOT_EXAM_LEROBOT_CACHE:?Verified local LeRobot cache required}"
: "${ROBOT_EXAM_TRAIN_SETTINGS:?Explicit smoke or pilot settings JSON required}"
: "${ROBOT_EXAM_TRAIN_RESULT:?New immutable training attempt JSON path required}"
test -d "$ROBOT_EXAM_WEIGHTS"
test -f "$ROBOT_EXAM_ASSETS/original_ur5/norm_stats.json"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_root/src:$ROBOT_EXAM_OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
exec "$ROBOT_EXAM_PYTHON" "$project_root/scripts/openpi_ur5_train.py" \
  --openpi-root "$ROBOT_EXAM_OPENPI_ROOT" --assets "$ROBOT_EXAM_ASSETS" \
  --weights "$ROBOT_EXAM_WEIGHTS" --output "$ROBOT_EXAM_TRAIN_RESULT" \
  --settings "$ROBOT_EXAM_TRAIN_SETTINGS" --lerobot-cache "$ROBOT_EXAM_LEROBOT_CACHE" \
  --tokenizer "$ROBOT_EXAM_TOKENIZER" --wrist-camera --authorized-training "$@"
