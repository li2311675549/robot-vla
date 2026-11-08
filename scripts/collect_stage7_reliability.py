#!/usr/bin/env python3
"""Collect saved physical evidence; do not run or replace any task episode."""
import argparse
import json
from pathlib import Path
import subprocess

from _bootstrap import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output path to preserve historical acceptance')
    records=[]
    for seed,suffix in [(412,'regression'),(413,'dev'),(414,'dev')]:
        folder=ROOT/f'outputs/stage7/fruit_seed{seed}_current_tristation_{suffix}'
        result=json.loads((folder/'result.json').read_text())
        settings=json.loads((folder/'config.json').read_text())
        solver=folder/'MUJOCO_LOG.TXT'
        records.append({k:result.get(k) for k in ('seed','success','completed','sim_seconds','wall_seconds','failure_category','error','num_retries','warnings')})
        records[-1].update(evidence=str(folder.relative_to(ROOT)/'result.json'),
                           task_sim_seconds_limit=settings['task_sim_seconds_limit'],
                           collision_events=len(result['collision_events']),
                           solver_linesearch_messages=solver.read_text().count('Linesearch objective is not convex') if solver.exists() else 0)
    successes=[r for r in records if r['success']]
    summary={'stage':7,'status':'PARTIAL','protocol':'development, same code and config; not held-out formal evaluation',
             'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'fruit':{'records':records,'complete_successes':len(successes),'episodes':len(records),
                      'complete_development_success_rate':len(successes)/len(records),
                      'successful_sim_seconds_mean':sum(r['sim_seconds'] for r in successes)/len(successes),
                      'successful_wall_seconds_mean':sum(r['wall_seconds'] for r in successes)/len(successes),
                      'successes_within_internal_1600_seconds':sum(r['success'] and r['sim_seconds']<=1600 for r in records),
                      'formal_10_seed_evaluation':'NOT_RUN'},
             'dock':json.loads((ROOT/'outputs/stage7/dock_pairs_v2_strict/summary.json').read_text()),
             'dock_independent_validation':'NOT_RUN',
             'pytest':{'passed':62,'failed':0,'evidence':'outputs/stage7/pytest_stage7_reliability_final.log'},
             'door_and_book_new_full_regressions':'NOT_RUN',
             'vla_training':'NOT_RUN'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
