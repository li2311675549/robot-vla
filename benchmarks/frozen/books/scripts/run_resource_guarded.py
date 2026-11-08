#!/usr/bin/env python3
"""Bound a preparation subprocess without changing cgroups or system settings."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

def accounted_bytes(mode, rss, cgroup_current=Path('/sys/fs/cgroup/memory.current')):
    """RSS is useful diagnostically, but CUDA mappings need container accounting.

    This only reads the kernel's existing counters; no cgroup limit is changed.
    Refuse missing/invalid counters instead of silently disabling the guard.
    """
    if mode == 'rss':
        return rss
    if mode != 'cgroup':
        raise ValueError('Unknown memory accounting mode')
    value = int(cgroup_current.read_text().strip())
    if value < 0:
        raise ValueError('Invalid cgroup memory counter')
    return value


def main():
    import psutil  # Server preparation dependency; not required for local counter tests.
    p=argparse.ArgumentParser()
    p.add_argument('--max-rss-mib','--max-memory-mib',dest='max_rss_mib',type=int,default=1100)
    p.add_argument('--memory-accounting',choices=['rss','cgroup'],default='rss')
    p.add_argument('--samples',type=Path,help='Optional once-per-second memory telemetry')
    p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or a.max_rss_mib<64 or a.timeout<=0:p.error('A command and positive resource budget are required')
    if a.output.exists() or a.log.exists() or (a.samples and a.samples.exists()):raise FileExistsError('Preserve existing preparation attempts')
    accounted_bytes(a.memory_accounting,0)  # Validate mode/counter before starting any work.
    a.output.parent.mkdir(parents=True,exist_ok=True);a.log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();peak=0;peak_accounted=0;reason=None;last_sample=-1.
    with a.log.open('x') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            root=psutil.Process(proc.pid)
            while proc.poll() is None:
                rss=0
                try:processes=[root,*root.children(recursive=True)]
                except psutil.NoSuchProcess:processes=[]
                for process in processes:
                    try:rss+=process.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss)
                used=accounted_bytes(a.memory_accounting,rss)
                peak_accounted=max(peak_accounted,used)
                elapsed=time.monotonic()-started
                if a.samples and elapsed-last_sample>=1.:
                    with a.samples.open('a') as stream:
                        stream.write(json.dumps({'elapsed_seconds':elapsed,'rss_mib':rss/1024**2,
                            'accounted_memory_mib':used/1024**2,'accounting':a.memory_accounting})+'\n')
                    last_sample=elapsed
                if used>a.max_rss_mib*1024**2:reason=a.memory_accounting+'_budget'
                elif time.monotonic()-started>a.timeout:reason='wall_time_budget'
                if reason:break
                time.sleep(.1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    result={'status':'PASS_SUBPROCESS_ONLY' if proc.returncode==0 and not reason else 'STOPPED_OR_FAILED',
            'returncode':proc.returncode,'stop_reason':reason,'peak_rss_mib':peak/1024**2,
            'wall_seconds':time.monotonic()-started,
            'rss_budget_mib':a.max_rss_mib if a.memory_accounting=='rss' else None,
            'memory_budget_mib':a.max_rss_mib,'memory_accounting':a.memory_accounting,
            'peak_accounted_memory_mib':peak_accounted/1024**2,
            'command':command,'log':str(a.log),'model_loaded_by_guard':False,
            'training_performed_by_guard':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status']!='PASS_SUBPROCESS_ONLY':raise SystemExit(1)


if __name__=='__main__':main()
