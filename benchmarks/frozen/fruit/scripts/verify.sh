#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/acceptance
export MUJOCO_GL="${MUJOCO_GL:-egl}"
.venv/bin/python -m pip check | tee outputs/acceptance/pip_check.txt
.venv/bin/python scripts/build_ur5.py | tee outputs/acceptance/build.txt
for scene in minimal ur5 robotiq; do
  .venv/bin/python scripts/smoke_test.py --scene "$scene" | tee "outputs/acceptance/smoke_${scene}.log"
done
.venv/bin/python -m pytest -v -m 'not stage2' --junitxml=outputs/acceptance/junit.xml 2>&1 | tee outputs/acceptance/pytest.log
for scene in minimal ur5 robotiq; do
  .venv/bin/python scripts/smoke_render.py --scene "$scene" --video | tee "outputs/acceptance/render_${scene}.log"
done
echo 'PASS: physics, model integrity, control, RGB-D, PNG and MP4. GUI is a separate check.'
