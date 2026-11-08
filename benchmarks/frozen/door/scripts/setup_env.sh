#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse only the interpreter; never install into an existing Conda environment.
if [[ ! -x .venv/bin/python ]]; then
  candidates=("${PYTHON310:-}" "$HOME/anaconda3/envs/lab/bin/python" /usr/bin/python3.10)
  selected=""
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys, ensurepip; assert sys.version_info[:2] == (3, 10)' 2>/dev/null; then
      selected="$candidate"
      break
    fi
  done
  if [[ -z "$selected" ]]; then
    echo 'Set PYTHON310 to a Python 3.10 interpreter with venv/ensurepip support.' >&2
    exit 1
  fi
  "$selected" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,10); assert sys.prefix != sys.base_prefix'
lock=requirements.lock.txt
[[ -f "$lock" ]] || lock=requirements.in
mkdir -p outputs/setup
du -sk .venv > outputs/setup/venv_before_kib.txt
.venv/bin/python -m pip install --no-cache-dir -r "$lock"
.venv/bin/python -m pip check
du -sk .venv > outputs/setup/venv_after_kib.txt
.venv/bin/python -m pip freeze > outputs/setup/installed.txt
echo 'Environment ready. Run: bash scripts/verify.sh'
