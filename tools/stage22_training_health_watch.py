#!/usr/bin/env python3
"""Run-scoped health watch. Never starts/restarts training or changes resources."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time


def decision(*, guard, last_metric, idle_seconds, files_unchanged, request_stop=False):
    if guard is not None:
        return 'FINISHED' if guard.get('returncode') == 0 else 'ALERT_GUARD_STOPPED'
    if request_stop:
        return 'STOP_REQUESTED'
    if not files_unchanged:
        return 'STOP_FROZEN_INPUT_CHANGED'
    if last_metric is not None and not all(math.isfinite(float(last_metric[k])) for k in ('loss', 'grad_norm')):
        return 'STOP_NONFINITE_METRIC'
    if idle_seconds > 900:
        return 'STOP_NO_OPTIMIZER_PROGRESS_15MIN'
    return 'RUNNING'


def terminate_owned_training(attempt, script):
    """SIGTERM only the exact child command/output and its private session.

    The existing parent guard stays alive to reap the child and write its
    terminal result. No other GPU/process/account is touched.
    """
    expected=str(Path(attempt).resolve()/'training.json');script=str(Path(script).resolve());sent=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            raw=path.read_bytes();args=[s.decode() for s in raw.split(b'\0') if s]
            if len(args)<2 or args[1]!=script or '--output' not in args:continue
            if args[args.index('--output')+1]!=expected:continue
            pid=int(path.parent.name)
            # Recheck identity immediately before signalling; never signal a
            # reused PID or a shared shell/process group.
            if path.read_bytes()!=raw:continue
            if os.getpgid(pid)!=pid:raise RuntimeError('Refusing non-private training process group')
            os.killpg(pid,signal.SIGTERM);sent.append(pid)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return sent


def read_json(path):
    # Training/guard reports use a short non-atomic write. Do not mistake its
    # transient empty/partial contents for a permanent watcher failure.
    for attempt in range(4):
        if not path.is_file():return None
        try:return json.loads(path.read_text())
        except FileNotFoundError:return None
        except json.JSONDecodeError:
            if attempt==3:raise
            time.sleep(.05)


def atomic_json(path, record):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--stage-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=float,default=30);p.add_argument('--max-scans',type=int)
    a=p.parse_args()
    if not 5<=a.interval<=60:p.error('Use a bounded5..60second watch interval')
    attempt=a.attempt.resolve();stage=a.stage_root.resolve()
    if not attempt.is_relative_to(stage/'results'):raise ValueError('Attempt outside the declared stage')
    if a.output.exists():raise FileExistsError('Keep previous health watch records')
    a.output.mkdir(parents=True)
    with (attempt/'health_watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        training=read_json(attempt/'training.json')
        if not training:raise FileNotFoundError('Attach only after the real training report exists')
        metrics=attempt/'training_metrics/train_metrics.jsonl'
        settings=stage/'code/configs/pi05_multifruit_stage22_batch4_v2.json'
        norm=stage/'dataset/assets/original_ur5/norm_stats.json'
        frozen={settings:hashlib.sha256(settings.read_bytes()).hexdigest(),norm:training['norm_stats_sha256']}
        checkpoint=Path(training['checkpoint_dir']).resolve()
        if not checkpoint.is_relative_to(stage/'checkpoints'):raise ValueError('Checkpoint outside stage')
        offset=0;last=None;max_step=0;scan=0;previous_step=None;started=time.time()
        while True:
            try:
                guard=read_json(attempt/'guard.json')
                if metrics.exists():
                    with metrics.open('rb') as stream:
                        stream.seek(offset);data=stream.read()
                    # Leave an in-flight final line for the next scan.
                    end=data.rfind(b'\n')+1
                    for line in data[:end].splitlines():
                        row=json.loads(line);step=int(row['optimizer_step'])
                        if step<=max_step:raise ValueError('Non-increasing optimizer step audit')
                        max_step=step;last=row
                    offset+=end
                idle=time.time()-(metrics.stat().st_mtime if metrics.exists() else started)
                unchanged=all(hashlib.sha256(path.read_bytes()).hexdigest()==sha for path,sha in frozen.items())
                complete=[]
                for folder in checkpoint.iterdir():
                    try:
                        if folder.name.isdigit() and all((folder/name).is_file() for name in
                                ('_CHECKPOINT_METADATA','params/_METADATA','train_state/_METADATA','assets/original_ur5/norm_stats.json')):
                            checkpoint_norm=read_json(folder/'assets/original_ur5/norm_stats.json')
                            if checkpoint_norm is None:continue  # normal retention removed this old save
                            if checkpoint_norm!=read_json(norm):unchanged=False
                            complete.append(int(folder.name))
                    except FileNotFoundError:
                        continue  # Retention may remove an older complete checkpoint during the scan.
                state=decision(guard=guard,last_metric=last,idle_seconds=idle,files_unchanged=unchanged,
                               request_stop=(attempt/'REQUEST_STOP').exists())
                record={'state':state,'time':time.time(),'attempt':str(attempt),'last_metric':last,
                        'max_optimizer_step':max_step,'idle_seconds':idle,'frozen_inputs_unchanged':unchanged,
                        'training_restarted_by_watch':False,'scope':'This attempt only; existing80GiB/25GiB guards unchanged'}
                record['complete_checkpoint_labels']=sorted(complete)
                record['planned_total_steps']=training['settings']['steps']
                if guard is not None:
                    record['training_terminal_status']=read_json(attempt/'training.json').get('status')
                if state.startswith('STOP_'):
                    record['signalled_owned_child_pids']=terminate_owned_training(attempt,stage/'code/scripts/openpi_ur5_train.py')
                if guard is not None:record['guard']=guard
                atomic_json(a.output/'status.json',record)
                if max_step!=previous_step or state!='RUNNING':
                    with (a.output/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    previous_step=max_step
                scan+=1
                if state!='RUNNING':return
                if a.max_scans is not None and scan>=a.max_scans:
                    record.update(state='BOUNDED_PROBE_FINISHED_TRAINING_UNCHANGED');atomic_json(a.output/'status.json',record);return
                time.sleep(a.interval)
            except Exception as exc:
                # A watcher error must not kill an unknown process or restart
                # a job. The independent existing resource guard stays active.
                atomic_json(a.output/'status.json',{'state':'WATCH_ERROR_TRAINING_UNCHANGED',
                    'error':str(exc),'type':type(exc).__name__,'training_restarted_by_watch':False})
                raise


if __name__=='__main__':main()
