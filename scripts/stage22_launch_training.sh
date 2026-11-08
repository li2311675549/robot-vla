#!/usr/bin/env bash
# Future manual entrypoint. No invocation of this file during no-card preparation.
set -euo pipefail
test "${ROBOT_EXAM_TRAIN_AUTHORIZED:-0}" = 1 || { echo 'Refusing: explicit GPU training authorization required'; exit 2; }
stage_root="${ROBOT_EXAM_STAGE22_ROOT:-/root/autodl-tmp/robot_exam_stage22}"
legacy_root="${ROBOT_EXAM_BASE_ROOT:-/root/autodl-tmp/robot_exam_training}"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
training_python="$legacy_root/openpi/.venv/bin/python"
mode="${1:-train}"
pause_args=()
case "$mode" in
  train) settings="$project_root/configs/pi05_multifruit_stage22_batch4_v2.json"; resume_args=();;
  train-first-half) settings="$project_root/configs/pi05_multifruit_stage22_batch4_v2.json"; resume_args=(); pause_args=(--stop-after-checkpoint 5000);;
  resume) settings="$project_root/configs/pi05_multifruit_stage22_batch4_v2.json"; resume_args=(--resume);;
  smoke) settings="$project_root/configs/pi05_multifruit_stage22_smoke_batch4_v2.json"; resume_args=();;
  resume-smoke) settings="$project_root/configs/pi05_multifruit_stage22_smoke_batch4_v2.json"; resume_args=(--resume);;
  reload-smoke) settings="$project_root/configs/pi05_multifruit_stage22_smoke_batch4_v2.json"; resume_args=(--resume);;
  *) echo 'Use train, train-first-half, resume, smoke, resume-smoke or reload-smoke'; exit 2;;
esac
mkdir -p "$stage_root/results" "$stage_root/checkpoints" "$stage_root/cache/jax"
exec 9>"$stage_root/training.lock"
flock -n 9 || { echo 'Another Stage22 training launcher holds the lock'; exit 2; }
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export HF_DATASETS_CACHE="$stage_root/cache/hf_datasets"
export HF_LEROBOT_HOME="$stage_root/dataset/lerobot"
export HF_HOME="$stage_root/cache/huggingface" UV_CACHE_DIR="$stage_root/cache/uv"
export XDG_CACHE_HOME="$stage_root/cache" JAX_COMPILATION_CACHE_DIR="$stage_root/cache/jax"
export PYTHONPATH="$project_root/src:$legacy_root/openpi/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.75
unset JAX_PLATFORMS
export CUDA_VISIBLE_DEVICES=0
"$training_python" - "$stage_root" "$mode" "$settings" <<'PY'
import json, shutil, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1])
if sys.argv[2]=='reload-smoke':
 settings=json.loads(Path(sys.argv[3]).read_text())
 final=root/'checkpoints/pi05_original_ur5_lora_v1'/settings['run_name']/str(settings['steps']-1)
 if not all((final/p).is_file() for p in ('_CHECKPOINT_METADATA','params/_METADATA','train_state/_METADATA')):
  raise SystemExit('Zero-update reload requires a complete final smoke checkpoint; use resume-smoke for recovery')
p=root/'results/no_gpu_readiness_final.json'
r=json.loads(p.read_text())
if r.get('status')!='PASS_NO_GPU_PREPARATION':raise SystemExit('No completed Stage22 readiness certificate')
# Bind the future launch to the actual prepared source/data and upstream trainer.
import hashlib, importlib.util
code=root/'code'
spec=importlib.util.spec_from_file_location('verify_release',code/'scripts/verify_stage22_transfer.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for label,base in [('dataset',root/'dataset'),('code',code)]:
 manifest=root/'results'/r.get(f'{label}_manifest_file',f'{label}_transfer_manifest_v1.json')
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!=r[f'{label}_manifest_sha256']:
  raise SystemExit(f'{label} manifest differs from the prepared release')
 module.verify(base,manifest)
for filename,expected in r['upstream_sha256'].items():
 path=Path(r['openpi_root'])/filename
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise SystemExit('Pinned OpenPI source changed')
free=shutil.disk_usage(root).free/1024**3
from robot_exam.vla.checkpoint_cache import checkpoint_disk_budget
active_settings=json.loads(Path(sys.argv[3]).read_text())
active_checkpoints=root/'checkpoints/pi05_original_ur5_lora_v1'/active_settings['run_name']
disk_budget=checkpoint_disk_budget(root/'checkpoints',active_checkpoint_dir=active_checkpoints)
if free<disk_budget['required_free_gib']:raise SystemExit(f'Checkpoint peak budget{disk_budget}, have{free:.2f}GiB')
print(json.dumps({'checkpoint_disk_budget':disk_budget,'free_gib':free}))
limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
if limit!='max' and int(limit)<86*1024**3:raise SystemExit('GPU mode must provide at least86GiB cgroup RAM for80GiB guard +6GiB margin')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
if len(gpu)!=1:raise SystemExit('This launch contract requires exactly one GPU')
total,available=map(int,gpu[0].split(','))
if total<80000 or available<75000:raise SystemExit('Require80,000MiB GPU total and75,000MiB free; re-review smaller cards')
import jax, jax.numpy as jnp, numpy as np
if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise SystemExit('JAX GPU backend unavailable; no CPU fallback')
x=jnp.eye(128,dtype=jnp.bfloat16)
y=jax.jit(lambda a:a@a)(x)
jax.block_until_ready(y)
np.testing.assert_allclose(np.asarray(y,dtype=np.float32),np.eye(128,dtype=np.float32))
print(json.dumps({'resource_gate':'PASS','disk_free_gib':free,'gpu_total_mib':total,'gpu_free_mib':available,
                  'jax_gpu_bfloat16_matmul':True,'model_loaded':False}))
PY
attempt="$(date -u +%Y%m%dT%H%M%SZ)_${mode}"
attempt_root="$stage_root/results/$attempt"
mkdir "$attempt_root"
exec "$training_python" "$project_root/scripts/run_training_disk_guard.py" \
  --base-guard "$project_root/scripts/run_resource_guarded.py" \
  --disk-root "$stage_root" --disk-root / --min-free-gib 25 -- \
  --memory-accounting cgroup --max-memory-mib 81920 --timeout 18000 \
  --samples "$attempt_root/memory.jsonl" --output "$attempt_root/guard.json" \
  --log "$attempt_root/train.log" -- \
  "$training_python" "$project_root/scripts/openpi_ur5_train.py" \
  --openpi-root "$legacy_root/openpi" --mixture-root "$stage_root/dataset" \
  --repo-id local/robot_exam_stage21_train --assets "$stage_root/dataset/assets" \
  --weights "$legacy_root/cache/openpi/openpi-assets/checkpoints/pi05_base/params" \
  --tokenizer "$legacy_root/cache/openpi/big_vision/paligemma_tokenizer.model" \
  --lerobot-cache "$stage_root/dataset/lerobot" --settings "$settings" \
  --checkpoints "$stage_root/checkpoints" --output "$attempt_root/training.json" \
  --wrist-camera --authorized-training --checkpoint-cache-advice "${resume_args[@]}" "${pause_args[@]}"
