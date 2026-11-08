#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p outputs/stage34/acceptance
.venv/bin/python scripts/build_tasks.py --task all --seed 0 | tee outputs/stage34/acceptance/build.log
.venv/bin/python -m pytest -v --junitxml=outputs/stage34/acceptance/junit.xml 2>&1 | tee outputs/stage34/acceptance/pytest.log
.venv/bin/python scripts/validate_navigation.py --output outputs/stage34/acceptance/navigation | tee outputs/stage34/acceptance/navigation.log
.venv/bin/python scripts/validate_passive_skills.py --output outputs/stage34/acceptance/passive_skills | tee outputs/stage34/acceptance/passive_skills.log
# Full tasks have strict independent success predicates. Failures stay nonzero.
# Run formal task evaluation separately; it records every failure and summary.
