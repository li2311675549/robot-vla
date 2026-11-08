#!/usr/bin/env python3
"""Real five-metre navigation/open/entry development check; excludes closing."""
import argparse
import json
from pathlib import Path
import time
import traceback

from build_tasks import build
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.door import open_and_enter


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=1901)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous physical trials')
    a.output.mkdir(parents=True)
    path,meta=build('door',a.seed,a.output/'scene.xml',door_profile='aligned_handles_v3')
    sim=Simulation(path);r=TaskRuntime(sim,output=a.output,sim_limit=2500,wall_limit=1800)
    r.config={'door_initial_dock_timeout_seconds':45}
    result={'scope':'OPEN_AND_ENTRY_ONLY: normal five-metre initial scene; not complete open-enter-close success','seed':a.seed}
    started=time.monotonic()
    try:
        angles=open_and_enter(r,meta)
        result.update(opened_and_entered=True,angles=angles,final_base=r.base.pose().tolist())
    except Exception as error:
        result.update(opened_and_entered=False,error=str(error),failure_category=getattr(error,'category','exception'),failure_stage=r.stage,traceback=traceback.format_exc())
    finally:r.base.command(0,0);r.save()
    result.update(sim_seconds=sim.data.time,wall_seconds=time.monotonic()-started,
                  warnings=sim.data.warning.number.tolist(),collisions=r.collision_events)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='traceback'},indent=2))
    if not result['opened_and_entered']:raise SystemExit(1)


if __name__=='__main__':main()
