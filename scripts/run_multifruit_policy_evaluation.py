#!/usr/bin/env python3
"""Run every frozen phase case, retain failures, separate validation and test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('validation','test'),required=True)
    p.add_argument('--phase',choices=('grasp','place'),default='grasp')
    p.add_argument('--expected-checkpoint-label',required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--allow-frozen-test',action='store_true')
    p.add_argument('--record',action='store_true');a=p.parse_args()
    if a.split=='test' and not a.allow_frozen_test:p.error('Select a checkpoint on validation before unlocking test')
    if a.output.exists():raise FileExistsError('Retain every matrix attempt')
    protocol=json.loads(a.protocol.read_text());rows=protocol['matrices'][f'{a.split}/{a.phase}']
    a.output.mkdir(parents=True);results=[];started=time.monotonic()
    for i,row in enumerate(rows):
        case=a.output/f"{i:03d}_{row['seed']}_{row['category']}";work=a.output/'work'/str(i);work.mkdir(parents=True)
        fixture=a.dataset_root/row['fixture'];error=None;rc=None
        try:
            for name,expected in row['sha256'].items():
                if hashlib.sha256((fixture/name).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Frozen fixture payload differs')
            command=[sys.executable,str(ROOT/'scripts/evaluate_multifruit_policy_phase.py'),
                '--fixture',str(fixture.resolve()),'--output',str(case.resolve()),'--policy','openpi',
                '--host',a.host,'--port',str(a.port),'--expected-checkpoint-label',a.expected_checkpoint_label,
                '--expected-normalization-sha256',protocol['normalization_sha256'],
                '--policy-noise-seed',str(row.get('policy_noise_seed',protocol.get('policy_noise_seed',0))),
                '--seconds',str(protocol['phase_budget_seconds'])]
            if a.record:command+=['--record']
            with (work/'process.log').open('x') as log:
                rc=subprocess.run(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=650,env=os.environ.copy()).returncode
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
        path=case/'result.json'
        result=json.loads(path.read_text()) if path.exists() else {'success':False,'status':'FAIL',
            'failure_category':'process_or_initialization_error','error':error}
        results.append({'fixture':row,'returncode':rc,'infrastructure_error':error,'result':result})
        summary={'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
            'checkpoint_label':a.expected_checkpoint_label,'split':a.split,'phase':a.phase,
            'attempted':len(results),'planned':len(rows),'status':'RUNNING',
            'scope':protocol['scope'],'wall_seconds':time.monotonic()-started,
            'results':results,'by_class':{}}
        for category in ('apple','banana','orange'):
            group=[r for r in results if r['fixture']['category']==category]
            summary['by_class'][category]={'attempted':len(group),'successes':sum(bool(r['result']['success']) for r in group),
                'all_attempt_success_rate':sum(bool(r['result']['success']) for r in group)/len(group) if group else None}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(i,row['category'],result['status'],flush=True)
    summary['status']='COMPLETE';(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
