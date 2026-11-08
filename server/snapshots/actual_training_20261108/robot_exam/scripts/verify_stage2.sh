#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
record_args=()
if [[ "${1:-}" == --record ]]; then
  record_args=(--record)
elif [[ $# -gt 0 ]]; then
  echo 'Usage: bash scripts/verify_stage2.sh [--record]' >&2
  exit 2
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p outputs/stage2/acceptance
.venv/bin/python scripts/build_stage2.py | tee outputs/stage2/acceptance/build.log
.venv/bin/python -m pytest -v --junitxml=outputs/stage2/acceptance/junit.xml 2>&1 | tee outputs/stage2/acceptance/pytest.log
.venv/bin/python scripts/validate_base.py "${record_args[@]}" | tee outputs/stage2/acceptance/gate1.log
for variant in box small wide cylinder; do
  if [[ "$variant" == box ]]; then
    .venv/bin/python scripts/validate_grasp.py --variant "$variant" "${record_args[@]}" | tee "outputs/stage2/acceptance/gate2_${variant}.log"
  else
    .venv/bin/python scripts/validate_grasp.py --variant "$variant" | tee "outputs/stage2/acceptance/gate2_${variant}.log"
  fi
done
.venv/bin/python scripts/validate_camera.py | tee outputs/stage2/acceptance/gate3.log
.venv/bin/python scripts/demo_mobile_manipulator.py "${record_args[@]}" | tee outputs/stage2/acceptance/gate4.log
.venv/bin/python scripts/collect_stage2_results.py
