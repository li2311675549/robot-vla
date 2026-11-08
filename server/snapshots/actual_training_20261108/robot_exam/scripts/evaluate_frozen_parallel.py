#!/usr/bin/env python3
"""Bounded parallel execution of the existing frozen ten-episode protocol.

Each physics process owns an isolated working directory. Results remain in
protocol order; interrupted attempts count as failures, never free retries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from _bootstrap import ROOT
from evaluate_frozen import hashes, summarize


def previous_attempt(folder,seed):
    path=folder/'result.json'
    if path.exists():return json.loads(path.read_text())
    if folder.exists():
        result={'seed':seed,'success':False,'failure_category':'process_interrupted',
                'sim_seconds':None,'wall_seconds':None,'note':'Interrupted formal attempt retained; not retried'}
        path.write_text(json.dumps(result,indent=2)+'\n')
        return result
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());seeds=protocol['seeds']
    if protocol['split']!='held_out_formal' or len(seeds)!=10 or len(set(seeds))!=10:raise ValueError('Ten unique reserved seeds required')
    if not 1<=a.workers<=3:raise ValueError('One to three isolated physics processes allowed')
    out=a.output.resolve();fingerprint=hashes(protocol)
    runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if out.exists() and not a.resume:raise FileExistsError('Preserve prior evaluation')
    out.mkdir(parents=True,exist_ok=True);freeze=out/'freeze.json'
    if a.resume:
        x=json.loads(freeze.read_text())
        if x['source_hashes']!=fingerprint or x['protocol']!=protocol or x['runner_sha256']!=runner_hash:raise RuntimeError('Frozen inputs changed')
    else:
        changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).splitlines()
        if any(path in fingerprint for path in changed):raise RuntimeError('Commit control/configuration before evaluation')
        freeze.write_text(json.dumps({'protocol':protocol,'source_hashes':fingerprint,'runner_sha256':runner_hash,
            'workers':a.workers,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'started_unix':time.time()},indent=2)+'\n')
    settings=json.loads((ROOT/'configs/task_defaults.json').read_text());settings.update(json.loads((ROOT/protocol['config']).read_text()))
    def execute(seed):
        if hashes(protocol)!=fingerprint:raise RuntimeError('Frozen source changed')
        folder=out/str(seed);result=previous_attempt(folder,seed)
        if result is None:
            folder.mkdir();command=[sys.executable,str(ROOT/'scripts/run_task.py'),'--task',protocol['task'],
                '--seed',str(seed),'--mode',protocol['mode'],'--config',str(ROOT/protocol['config']),
                '--output',str(folder),'--door-profile',protocol.get('door_profile','fixed_v1'),
                '--book-profile',protocol.get('book_profile','booklet')]
            if seed in protocol.get('record_seeds',[]):command.append('--record')
            with (folder/'process.log').open('w') as log:
                try:code=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,
                    timeout=settings['task_wall_seconds_limit']+60).returncode
                except subprocess.TimeoutExpired:code=None
            path=folder/'result.json'
            result=json.loads(path.read_text()) if path.exists() else {'seed':seed,'success':False,
                'failure_category':'process_timeout_or_crash','sim_seconds':None,'wall_seconds':None}
            result['process_returncode']=code
            if not path.exists():path.write_text(json.dumps(result,indent=2)+'\n')
        if hashes(protocol)!=fingerprint:raise RuntimeError('Frozen source changed during an episode')
        log=folder/'MUJOCO_LOG.TXT'
        result.update(evidence=str(folder/'result.json'),root_failure=result.get('failure_category'),
            solver_linesearch_messages=log.read_text().count('Linesearch objective is not convex') if log.exists() else 0)
        return result
    completed={}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(execute,seed):seed for seed in seeds}
        for future in as_completed(futures):
            result=future.result();completed[futures[future]]=result
            records=[completed[seed] for seed in seeds if seed in completed]
            (out/'episodes.json').write_text(json.dumps(records,indent=2)+'\n')
            (out/'summary.json').write_text(json.dumps({'protocol':protocol,'summary':summarize(records)},indent=2)+'\n')
            print(result['seed'],result.get('success'),result.get('sim_seconds'),result.get('failure_category'),flush=True)
    (out/'source_integrity.json').write_text(json.dumps({'source_hash_mismatches':[
        path for path,value in hashes(protocol).items() if fingerprint.get(path)!=value],
        'episodes':len(completed),'all_attempts_counted':len(completed)==10},indent=2)+'\n')


if __name__=='__main__':main()
