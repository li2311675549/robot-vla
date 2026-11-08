#!/usr/bin/env python3
"""Local planner ablation using frozen physical snapshots, never live resets.

The candidate planner is injected only into this diagnostic process. No
production default changes until independent physical evidence exists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from _bootstrap import ROOT
from benchmark_dock_stage7 import trial
from benchmark_dock_capture import isolated_log_directory
from robot_exam.control.base import wrap_angle
from robot_exam.navigation.local_dock import connect
import robot_exam.control.local_dock as controller


def timed_plan(start,goal,strategy='preapproach',valid=lambda p:True,max_nodes=12000):
    start,goal=np.asarray(start),np.asarray(goal);clock=time.monotonic()
    direction=np.array([np.cos(goal[2]),np.sin(goal[2])]);paths=[]
    direct=connect(start,goal,valid)
    if direct is not None:paths.append(direct)
    for distance in (.05,.10,.15,.22,.30,.45,.60):
        for sign in (-1,1):
            pre=np.r_[goal[:2]+sign*distance*direction,goal[2]]
            first,last=connect(start,pre,valid),connect(pre,goal,valid)
            if first is not None and last is not None:paths.append(first+last[1:])
    if not paths:raise RuntimeError('No safe direct or goal-axis connection')
    costs=[]
    for path in paths:
        length=sum(np.linalg.norm(np.subtract(b[:2],a[:2])) for a,b in zip(path[:-1],path[1:]))
        turns=sum(abs(wrap_angle(b[2]-a[2])) for a,b in zip(path[:-1],path[1:]))
        # Fixed aggregate speeds from previous DEV logs, not heldout fitting.
        costs.append((length/.0748268+turns/.0301733,path,length,turns))
    cost,path,length,turns=min(costs,key=lambda x:x[0])
    return dict(strategy='timed_connectors_diagnostic_v1',poses=path,expanded_nodes=0,
        planning_wall_seconds=time.monotonic()-clock,estimated_seconds=float(cost),
        path_length_m=float(length),turn_total_rad=float(turns),
        candidate_count=len(costs),kinematic_model='forward/reverse and in-place yaw only')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=int,nargs='+',default=[2,5,8,9])
    p.add_argument('--kinds',nargs='+',choices=['empty','loaded'],default=['empty','loaded'])
    p.add_argument('--timeout',type=float,default=45.)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/stage7/dock_pairs_v2_strict')
    a=p.parse_args();out=a.output.resolve();source=a.source.resolve()
    if out.exists():raise FileExistsError('Preserve previous pairs')
    out.mkdir(parents=True);original=controller.plan_dock
    report=dict(status='RUNNING',timeout_sim_seconds=a.timeout,
        note='Production45s protocol is separate from former strict25s. Both policies use exactly the same budget, snapshot, tolerance and executor.',
        tolerance='23mm/.035rad/.02mps/.03radps/2s plus independent2s',rows=[])
    save=lambda:(out/'summary.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        for kind in a.kinds:
            for index in a.cases:
                src=source/kind/f'case_{index:02d}';raw=np.load(src/'initial_state.npz')
                snapshot=json.loads((src/'fixture.json').read_text());snapshot.update({k:raw[k] for k in raw.files})
                snapshot['state_spec']=int(snapshot['state_spec'])
                digest=hashlib.sha256((src/'initial_state.npz').read_bytes()).hexdigest()
                for name,planner in [('preapproach',original),('timed',timed_plan)]:
                    controller.plan_dock=planner;folder=out/kind/f'case_{index:02d}'/name
                    with isolated_log_directory(folder):
                        result=trial(src/'scene.xml',snapshot,name,folder,kind,timeout=a.timeout,
                            config_overrides=dict(local_dock_strategy='preapproach',local_dock_tracking_version='axis_bounded_v5',
                                payload_precision_twist_slew=True,payload_twist_slew_categories=['apple','banana','orange'],
                                local_dock_capture_goal_during_primitives=True))
                    result.update(case_index=index,same_initial_state_sha256=digest)
                    report['rows'].append(result);save();print(kind,index,name,result['strict_success_after_2s'],flush=True)
        report['status']='COMPLETE_ALL_ATTEMPTS_RETAINED'
        report['counts']={f'{kind}/{name}':dict(attempts=len(rows),strict_pass=sum(r['strict_success_after_2s'] for r in rows))
            for kind in a.kinds for name in ('preapproach','timed')
            for rows in [[r for r in report['rows'] if r['kind']==kind and r['strategy']==name]]}
        save()
    finally:controller.plan_dock=original


if __name__=='__main__':main()
