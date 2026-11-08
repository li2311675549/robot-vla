#!/usr/bin/env python3
"""Eight physical wheel-driven navigation cases, including failure injection."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from _bootstrap import ROOT
import mujoco
import numpy as np
from build_stage2 import scene_root
from build_tasks import robot, add
from robot_exam.simulation import Simulation, UR5_JOINTS
from robot_exam.navigation.grid import OccupancyGrid, NoPath
from robot_exam.tasks.runtime import TaskRuntime, State


CASES = {
    'straight_5m': ([-5,0,0], [[0,0,0]], []),
    'turn_90': ([-2,-1,0], [[0,-1,0],[0,1,np.pi/2]], []),
    'obstacle': ([-2,0,0], [[2,0,0]], [[-.3,.3,-.4,.4]]),
    'initial_yaw': ([-2,0,1.2], [[0,0,0]], []),
    'accurate_stop': ([-1,0,0], [[0,0,0]], []),
    'replan': ([-2,0,0], [[2,0,0]], [[-.3,.3,-.4,.4]]),
    'open_door_passage': ([-2,0,0], [[2,0,0]], [[-.1,.1,.8,3],[-.1,.1,-3,-.8],[-1.6,0,-.85,-.73]]),
    'return_home': ([-2,-1,0], [[0,0,0],[-2,-1,0]], []),
}


def run_case(name, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    start,goals,boxes=CASES[name]
    root=scene_root('navigation_'+name);robot(root,start)
    for i,box in enumerate(boxes):
        x0,x1,y0,y1=box
        add(root.find('worldbody'),'geom',name=f'obstacle_{i}',type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} .5',size=f'{(x1-x0)/2} {(y1-y0)/2} .5',rgba='.4 .4 .5 1')
    for element in root.findall('.//mesh'):
        if element.get('file'):element.set('file',str((ROOT/'scenes'/element.get('file')).resolve()))
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    q=model.qpos0.copy();ctrl=np.zeros(model.nu)
    for joint,value in zip(UR5_JOINTS,TaskRuntime.STOW):
        q[model.jnt_qposadr[model.joint(joint).id]]=value;ctrl[model.actuator(joint+'_position').id]=value
    add(add(root,'keyframe'),'key',name='home',qpos=' '.join(map(str,q)),ctrl=' '.join(map(str,ctrl)))
    path=output/'scene.xml';ET.ElementTree(root).write(path)
    sim=Simulation(path);r=TaskRuntime(sim,output=output)
    class RefreshingGrid(OccupancyGrid):
        calls=0
        def plan(self,*args):
            self.calls+=1
            if self.calls==1:raise NoPath('deliberate stale-map fault injection')
            return super().plan(*args)
    grid=(RefreshingGrid if name=='replan' else OccupancyGrid)(bounds=(-6,4,-4,4))
    for box in boxes:grid.rectangle(*box)
    r.hold(.6);results=[]
    try:
        for goal in goals:
            r.transition(State.NAVIGATE,'validation_'+name)
            results.append(r.navigate(goal,grid,timeout=200))
        before=r.base.pose().copy();r.hold(1)
        stop_drift=float(np.linalg.norm(r.base.pose()[:2]-before[:2]))
        success=all(x['position_error_m']<.09 and x['heading_error_rad']<.08 for x in results) and stop_drift<.01 and not r.collision_events and not sim.data.warning.number.any()
        result={'status':'PASS' if success else 'FAIL','case':name,'legs':results,'stop_drift_m':stop_drift,
                'wheel_angle_norm_rad':float(np.linalg.norm(sim.data.qpos[7:11])),'warnings':sim.data.warning.number.tolist(),
                'recovery_scope':'fault-injected stale map, bounded plan retry' if name=='replan' else None}
    except Exception as exc:result={'status':'FAIL','case':name,'error':str(exc),'legs':results}
    finally:r.save()
    (output/'result.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/stage34/navigation');a=p.parse_args()
    results=[run_case(name,a.output/name) for name in CASES]
    summary={'passed':sum(r['status']=='PASS' for r in results),'total':len(results),'results':results}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    if summary['passed']!=len(results):raise SystemExit(1)


if __name__=='__main__':main()
