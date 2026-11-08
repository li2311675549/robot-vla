#!/usr/bin/env python3
"""Freeze a protocol and run every reserved seed once, with resumable evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from _bootstrap import ROOT


def hashes(protocol):
    paths=[ROOT/'scripts/build_tasks.py',ROOT/'scripts/build_stage2.py',ROOT/'scripts/run_task.py',
           ROOT/'configs/task_defaults.json',ROOT/protocol['config']]
    # Include skills (notably physical book pushing) and the shared step
    # boundary, so an indirect dependency cannot silently change mid-eval.
    paths.extend((ROOT/'src/robot_exam').rglob('*.py'))
    paths.extend([ROOT/'configs/model_sources.lock.json',ROOT/'assets/ur5/ur5.xml',
                  ROOT/'assets/robotiq_2f85/2f85.xml'])
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def summarize(records):
    successful=[x for x in records if x.get('success')]
    errors={}
    for row in records:
        if not row.get('success'):
            key=row.get('root_failure',row.get('failure_category','unknown'))
            errors[key]=errors.get(key,0)+1
    return {'episodes':len(records),'complete_successes':len(successful),
        'complete_success_rate':len(successful)/len(records) if records else None,
        'successful_sim_seconds_mean':sum(x['sim_seconds'] for x in successful)/len(successful) if successful else None,
        'successful_wall_seconds_mean':sum(x['wall_seconds'] for x in successful)/len(successful) if successful else None,
        'successful_sim_seconds_worst':max((x['sim_seconds'] for x in successful),default=None),
        'failures':errors,'recoveries':sum(x.get('num_retries',0) for x in records),
        'phase_successes':{key:sum(bool(r.get('phase_metrics',{}).get(key)) for r in records)
                           for key in ('opened','entered','closed_after_entry')}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());out=a.output.resolve()
    if protocol['split']!='held_out_formal' or len(protocol['seeds'])!=10 or len(set(protocol['seeds']))!=10:
        raise ValueError('Formal protocol requires ten unique reserved seeds')
    if out.exists() and not a.resume:raise FileExistsError('Refusing to overwrite evaluation')
    out.mkdir(parents=True,exist_ok=True)
    fingerprint=hashes(protocol);freeze=out/'freeze.json'
    if a.resume:
        saved=json.loads(freeze.read_text())
        if saved['source_hashes']!=fingerprint or saved['protocol']!=protocol:raise RuntimeError('Frozen evaluation source/protocol changed')
    else:
        changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).splitlines()
        if any(path in fingerprint for path in changed):raise RuntimeError('Commit task source/config before formal evaluation')
        freeze.write_text(json.dumps({'protocol':protocol,'source_hashes':fingerprint,
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'started_unix':time.time()},indent=2)+'\n')
    records=[]
    settings=json.loads((ROOT/'configs/task_defaults.json').read_text())
    settings.update(json.loads((ROOT/protocol['config']).read_text()))
    for seed in protocol['seeds']:
        if hashes(protocol)!=fingerprint:raise RuntimeError('Task source changed during frozen evaluation')
        folder=out/str(seed);path=folder/'result.json'
        if path.exists():result=json.loads(path.read_text())
        elif folder.exists():
            # An interrupted episode remains a failed attempt, not a free retry.
            result={'seed':seed,'success':False,'failure_category':'process_interrupted','sim_seconds':None,'wall_seconds':None}
        else:
            folder.mkdir()
            command=[sys.executable,str(ROOT/'scripts/run_task.py'),'--task',protocol['task'],
                '--seed',str(seed),'--mode',protocol['mode'],'--config',str(ROOT/protocol['config']),
                '--output',str(folder),'--door-profile',protocol.get('door_profile','fixed_v1'),
                '--book-profile',protocol.get('book_profile','booklet')]
            if seed in protocol.get('record_seeds',[]):command.append('--record')
            with (folder/'process.log').open('w') as log:
                try:
                    completed=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,
                        timeout=settings['task_wall_seconds_limit']+60)
                    code=completed.returncode
                except subprocess.TimeoutExpired:code=None
            result=json.loads(path.read_text()) if path.exists() else {
                'seed':seed,'success':False,'failure_category':'process_timeout_or_crash','sim_seconds':None,'wall_seconds':None}
            result['process_returncode']=code
        text=str(result.get('error','')).lower()
        root_failure=('dock' if 'docking failed' in text or 'drift after final turn' in text else result.get('failure_category'))
        solver=folder/'MUJOCO_LOG.TXT'
        records.append({**result,'root_failure':root_failure,
            'solver_linesearch_messages':solver.read_text().count('Linesearch objective is not convex') if solver.exists() else 0,
            'evidence':str(path)})
        (out/'episodes.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
        (out/'summary.json').write_text(json.dumps({'protocol':protocol,'summary':summarize(records)},indent=2)+'\n')
        print(seed,result.get('success'),result.get('sim_seconds'),root_failure,flush=True)
    with (out/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','success','completed','sim_seconds','wall_seconds','root_failure','num_retries','solver_linesearch_messages'],extrasaction='ignore')
        writer.writeheader();writer.writerows(records)


if __name__=='__main__':main()
