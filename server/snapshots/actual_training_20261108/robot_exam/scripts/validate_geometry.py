#!/usr/bin/env python3
"""Independent zero-time reachability fixtures; never counts as physical success."""
import json
from pathlib import Path
from _bootstrap import ROOT
from build_tasks import build
import mujoco,numpy as np
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController
from robot_exam.tasks.door import handle_position,rotation_z


def main():
    out=ROOT/'outputs/stage34/geometry';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for task in ['fruit','door','drawer']:
        path,meta=build(task,0,out/(task+'.xml'));s=Simulation(path)
        def check(label,base,position,R,seed=None):
            # Fresh planning-only MjData for each fixture; no live execution.
            d=mujoco.MjData(s.model);mujoco.mj_resetDataKeyframe(s.model,d,0)
            d.qpos[:2]=base[:2];d.qpos[3:7]=[np.cos(base[2]/2),0,0,np.sin(base[2]/2)];mujoco.mj_forward(s.model,d)
            from types import SimpleNamespace
            arm=ArmController(SimpleNamespace(model=s.model,data=d))
            try:
                q,metrics=arm.solve(position,R,seed=seed)
                row={'task':task,'target':label,'ik_reachable':True,**metrics,'q':q.tolist()}
            except Exception as e:row={'task':task,'target':label,'ik_reachable':False,'error':str(e)}
            records.append(row);assert d.time==0
        if task=='fruit':
            for obj in meta['objects']:
                p=np.array(obj['initial']);p[2]-=.01 if obj['category']=='apple' else 0
                check(obj['body'],meta['dock'],p,np.diag([1,-1,-1]))
            for cat,entry in meta['bins'].items():
                for dy in [-.032,.032]:check('bin_'+cat, [0,entry['center'][1],0],[.55,entry['center'][1]+dy,.789],np.diag([1,-1,-1]))
        elif task=='drawer':
            R=np.array([[0,0,1],[1,0,0],[0,1,0.]])
            for layer,z in enumerate(meta['drawer_bottoms']):
                for opening in [0,.27]:
                    base=meta['cabinet_dock'] if opening==0 else [-.35 if layer==2 else -.3,.5,0]
                    check(f'handle_{layer}_{opening}',base,[.525-opening,.5,z+.165],R,seed=[0,-1.5,2,-2,1.57,0])
        else:
            for angle in [0,.5,1.,1.56]:
                base=meta['closing_base'];p=handle_position(meta,angle,True);f=p-np.r_[base[:2],p[2]];f[2]=0;f/=np.linalg.norm(f)
                R=np.column_stack([[0,0,1],[f[1],-f[0],0],f])
                check(f'inner_handle_{angle}',base,p,R,seed=[0,-1.5,2,-2,1.57,0])
    result={'scope':'kinematic planning only, no mj_step and no physical task success claimed','targets':len(records),
            'ik_reachable':sum(r['ik_reachable'] for r in records),'records':records}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
