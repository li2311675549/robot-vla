#!/usr/bin/env bash
set -euo pipefail
source "${ROBOT_EXAM_SERVER_PATHS:-/root/autodl-tmp/robot_exam_training/server_paths.sh}"
unset JAX_PLATFORMS
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p "$ROBOT_EXAM_BUNDLE/gpu_checks"
tag="$(date -u +%Y%m%dT%H%M%SZ)"
export ROBOT_EXAM_GPU_CHECK_RESULT="$ROBOT_EXAM_BUNDLE/gpu_checks/bfloat16_$tag.json"
"$ROBOT_EXAM_PYTHON" "$ROBOT_EXAM_BUNDLE/robot_exam/scripts/autodl_server_preflight.py" --require-gpu --disk-root "$ROBOT_EXAM_BUNDLE" --output "$ROBOT_EXAM_BUNDLE/gpu_checks/environment_$tag.json"
"$ROBOT_EXAM_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
if not any(device.platform=="gpu" for device in jax.devices()):
    raise RuntimeError("GPU mode is required; CPU fallback forbidden")
x=jnp.eye(128,dtype=jnp.bfloat16)
y=jax.jit(lambda a:a@a)(x)
jax.block_until_ready(y)
np.testing.assert_allclose(np.asarray(y,dtype=np.float32),np.eye(128,dtype=np.float32))
result={"status":"PASS_GPU_BFLOAT16_MATMUL_ONLY","devices":[str(d) for d in jax.devices()],"model_loaded":False,"training_performed":False}
with Path(os.environ["ROBOT_EXAM_GPU_CHECK_RESULT"]).open("x") as stream:
    stream.write(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
PY
