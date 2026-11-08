#!/usr/bin/env python3
"""Consecutive independent attempts; never replaces failed seeds with successes."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['all','fruit','door','drawer'],default='all');p.add_argument('--seeds',type=Path,default=ROOT/'configs/final_seeds.json')
    p.add_argument('--development',action='store_true');p.add_argument('--mode',choices=['vision','oracle']);p.add_argument('--headless',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();seeds=json.loads(a.seeds.read_text());out=a.output or ROOT/'outputs/stage34/evaluation'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=True)
    records=[]
    for task in (['fruit','door','drawer'] if a.task=='all' else [a.task]):
        mode=a.mode or ('oracle' if task=='door' else 'vision')
        for seed in seeds['development' if a.development else task]:
            episode=out/task/str(seed);episode.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(ROOT/'scripts/run_task.py'),'--task',task,'--mode',mode,'--seed',str(seed),'--output',str(episode)]
            with (episode/'process.log').open('w') as log:
                try:proc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1850)
                except subprocess.TimeoutExpired:proc=None
            path=episode/'result.json'
            if path.exists():r=json.loads(path.read_text())
            else:r={'task':task,'seed':seed,'success':False,'status':'FAIL','failure_category':'process_timeout_or_crash','sim_seconds':None,'wall_seconds':None}
            r['process_returncode']=None if proc is None else proc.returncode
            records.append(r);(out/'episodes.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
            print(task,seed,r['status'],r.get('failure_category'),r.get('sim_seconds'),flush=True)
    summary={}
    for task in set(r['task'] for r in records):
        group=[r for r in records if r['task']==task];success=[r for r in group if r.get('success')]
        categories={k:sum(r.get('failure_category')==k for r in group) for k in set(r.get('failure_category') for r in group) if k}
        attempts=[x for r in group for x in r.get('grasp_attempts',[])]
        summary[task]={'episodes':len(group),'complete_successes':len(success),'complete_success_rate':len(success)/len(group),
                       'successful_sim_seconds_mean':float(np.mean([r['sim_seconds'] for r in success])) if success else None,
                       'successful_sim_seconds_worst':max((r['sim_seconds'] for r in success),default=None),
                       'all_attempt_sim_seconds_mean':float(np.mean([r['sim_seconds'] for r in group if r.get('sim_seconds') is not None])),
                       'failure_categories':categories,'recoveries':sum(r.get('num_retries',0) for r in group),
                       'single_grasp_attempts':len(attempts),'single_grasp_success_rate':sum(x['success'] for x in attempts)/len(attempts) if attempts else None,
                       'policy_mode':group[0].get('policy_mode'),'protocol':'development' if a.development else 'frozen independent consecutive complete-task attempts'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,indent=2))
    with (out/'episodes.csv').open('w',newline='') as f:
        fields=['task','seed','success','sim_seconds','wall_seconds','failure_stage','failure_category','num_retries'];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(records)


if __name__=='__main__':main()
