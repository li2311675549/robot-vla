#!/usr/bin/env python3
"""Check the prepared training environment, without loading a model or using network."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disk-root', type=Path, required=True)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve server preflight attempts')
    report = {'python': sys.version, 'platform': platform.platform(), 'checks': {},
              'model_loaded': False, 'training_performed': False, 'network_used': False}
    failures = []
    if sys.version_info[:2] != (3, 11):
        failures.append('Use a separate Python 3.11 OpenPI environment')
    report['packages'] = {}
    for name in ('jax', 'jaxlib', 'flax', 'orbax-checkpoint', 'numpy', 'torch', 'datasets',
                 'lerobot', 'sentencepiece', 'transformers', 'optax', 'wandb'):
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f'Missing package {name}')
    try:
        import jax
        report['devices'] = [{'device': str(d), 'platform': d.platform} for d in jax.devices()]
        report['checks']['jax_compute'] = int(jax.numpy.asarray([1, 2]).sum()) == 3
        if args.require_gpu and not any(d.platform == 'gpu' for d in jax.devices()):
            failures.append('JAX GPU unavailable; refuse CPU training fallback')
    except Exception as error:
        failures.append(f'JAX check: {type(error).__name__}: {error}')
    if shutil.which('nvidia-smi'):
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        report['nvidia_smi'] = {'exit_code': result.returncode, 'stdout': result.stdout.strip(),
                                'stderr': result.stderr.strip()}
    report['free_disk_GiB'] = shutil.disk_usage(args.disk_root).free/1024**3
    if report['free_disk_GiB'] < 25:
        failures.append('Less than 25GiB free; weights/checkpoints need additional measured budget')
    report.update(status='FAIL' if failures else ('PASS_GPU_ENVIRONMENT_ONLY' if args.require_gpu else 'PASS_LOCAL_CPU_ENVIRONMENT_ONLY'),
                  failures=failures, gpu_required=args.require_gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
