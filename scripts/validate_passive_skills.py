#!/usr/bin/env python3
"""Isolated parked manipulation tests; these are not full 5m task attempts."""
import argparse,json
from pathlib import Path
from _bootstrap import ROOT
from build_tasks import build
import mujoco,numpy as np
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.tasks.manipulation import drawer_operation
from robot_exam.tasks.door import open_segment


def run_case(case,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    task='door' if case=='door_outer_pull' else 'drawer'
    path,meta=build(task,0,output/'scene.xml');s=Simulation(path)
    # Independent skill fixture initialization, before the first physical step.
    start=meta['opening_base'] if task=='door' else meta['cabinet_dock']
    s.data.qpos[:2]=start[:2];s.data.qpos[3:7]=[np.cos(start[2]/2),0,0,np.sin(start[2]/2)]
    mujoco.mj_forward(s.model,s.data)
    r=TaskRuntime(s,output=output)
    try:
        r.hold(.6)
        if task=='door':
            actual=open_segment(r,meta,.1)
            result={'success':abs(actual-.1)<.07,'actual_opening_rad':actual,'bilateral':r.bilateral('door_leaf')}
        else:
            layer=int(case[-1]);opened=drawer_operation(r,layer,.26,meta)
            r.stow();closed=drawer_operation(r,layer,0.,meta);r.hold(1)
            q=float(s.data.qpos[s.model.jnt_qposadr[s.model.joint(f'drawer_slide_{layer}').id]])
            result={'success':opened['actual']>.23 and abs(q)<.012,'opened':opened,'closed':closed,'settled_slide_m':q}
    except Exception as e:result={'success':False,'error':str(e),'category':getattr(e,'category','exception')}
    finally:r.save()
    result.update({'status':'PASS' if result['success'] else 'FAIL','scope':'isolated parked physical skill, no navigation','sim_seconds':s.data.time,
                   'warnings':s.data.warning.number.tolist()})
    (output/'result.json').write_text(json.dumps(result,indent=2));print(case,result,flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/stage34/passive_skills');a=p.parse_args()
    results={name:run_case(name,a.output/name) for name in ['door_outer_pull','drawer_0','drawer_1','drawer_2']}
    (a.output/'summary.json').write_text(json.dumps(results,indent=2))
    if not all(r['success'] for r in results.values()):raise SystemExit(1)


if __name__=='__main__':main()
