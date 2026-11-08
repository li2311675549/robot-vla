#!/usr/bin/env python3
"""Local physical fixture: true grasp followed by one coordinated door arc."""
import argparse
import json
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

from build_tasks import build
import mujoco
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.door_contact import arc_station,run_contact_phase
from robot_exam.evaluation.recording import VideoRecorder


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['open','close'],required=True);p.add_argument('--seed',type=int,default=2101)
    p.add_argument('--record',action='store_true');p.add_argument('--speed',type=float,default=.020)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);inner=a.phase=='close'
    angle=1.56 if inner else 0.;target=0. if inner else 1.56
    scene,meta=build('door',a.seed,a.output/'scene.xml',door_profile='aligned_handles_v3')
    model=mujoco.MjModel.from_xml_path(str(scene));tree=ET.parse(scene);key=tree.find('keyframe/key')
    q=np.fromstring(key.get('qpos'),sep=' ');base=arc_station(meta,angle,inner)
    q[:2]=base[:2];q[3:7]=[np.cos(base[2]/2),0,0,np.sin(base[2]/2)];q[model.jnt_qposadr[model.joint('door_hinge').id]]=angle
    key.set('qpos',' '.join(map(str,q)));tree.write(scene,encoding='unicode',xml_declaration=True)
    sim=Simulation(scene);camera=mujoco.MjvCamera();camera.lookat[:]=[-.25,-.3,.8];camera.distance=4.5;camera.elevation=-65;camera.azimuth=145
    rec=VideoRecorder(a.output/'video',camera=camera,fps=5) if a.record else None
    r=TaskRuntime(sim,output=a.output,callback=rec,sim_limit=400,wall_limit=1200);r.config={'door_contact_angle_speed_rad_s':a.speed}
    result={'scope':'LOCAL_ARC_ONLY; time-zero base/door fixture; grasp itself uses real contacts. Not a five-metre task.','seed':a.seed,'phase':a.phase}
    started=time.monotonic()
    try:
        r.hold(.6)
        result.update(run_contact_phase(r,meta,target,inner))
        result['success']=result['completed']
    except Exception as e:result.update(success=False,error=str(e),traceback=traceback.format_exc(),failure_category=getattr(e,'category','exception'),**getattr(e,'partial',{}))
    finally:
        r.base.command(0,0);r.save()
        if rec:result['video']=rec.close()
    result.update(sim_seconds=float(sim.data.time),wall_seconds=time.monotonic()-started,warnings=sim.data.warning.number.tolist(),collisions=r.collision_events)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('samples','traceback')},indent=2))
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
