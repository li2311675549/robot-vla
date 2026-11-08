#!/usr/bin/env python3
"""Validate all merged episodes with the native loader using disposable RAM cache."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--python',type=Path,default=Path('/home/li/vla/openpi/.venv/bin/python'))
    p.add_argument('--openpi-root',type=Path,default=Path('/home/li/vla/openpi'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous native validation')
    if shutil.disk_usage('/dev/shm').free<5*1024**3:raise RuntimeError('Require5GiB existing temporary shared-memory headroom')
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    if available<6*1024**3:raise RuntimeError('Require6GiB available RAM; no system changes')
    prepared=a.prepared.resolve();reports={};peak=0
    with tempfile.TemporaryDirectory(prefix='robot_exam_native_',dir='/dev/shm') as temporary:
        cache=Path(temporary)
        env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_DATASETS_CACHE=str(cache),
                 JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        for split in ('train','validation','test'):
            result=prepared/f'native_{split}_ram.json';log=prepared/f'native_{split}_ram.log'
            if result.exists() or log.exists():raise FileExistsError('Preserve prior split validation')
            cmd=[str(a.python),str(Path(__file__).resolve().parent/'validate_openpi_ur5_mapping.py'),
                '--openpi-root',str(a.openpi_root),'--manifest',str(prepared/'manifest'),
                '--lerobot-cache',str(prepared/'cache'),'--repo-id',f'local/robot_exam_green_timing_full_{split}',
                '--converted-only','--wrist-camera','--output',str(result)]
            start=time.monotonic()
            with log.open('x') as stream:
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                while child.poll() is None:
                    used=sum(f.stat().st_size for f in cache.rglob('*') if f.is_file());peak=max(peak,used)
                    if used>4*1024**3 or time.monotonic()-start>1800:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise RuntimeError('Bound native RAM cache/time budget')
                    time.sleep(1)
            if child.returncode:raise RuntimeError(f'Native{split} validation failed; inspect{log}')
            reports[split]=json.loads(result.read_text())
        peak=max(peak,sum(f.stat().st_size for f in cache.rglob('*') if f.is_file()))
    report={'status':'PASS_NATIVE_ALL_THREE_SPLITS','splits':{k:v['native_lerobot_training_loader'] for k,v in reports.items()},
            'temporary_ram_cache_peak_bytes':peak,'temporary_cache_removed':True,'disk_source_data_modified':False,
            'model_loaded':False,'training_started':False}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
