#!/usr/bin/env python3
"""Read-only RGB-D instance diagnostics at actual saved failure checkpoints."""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.perception.objects import observe_objects


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve diagnostics')
    a.output.mkdir(parents=True);fsm=json.loads((a.episode/'fsm.json').read_text());index=max(i for i,x in enumerate(fsm) if x['reason']=='multi_candidate_grasp')
    state=np.load(a.episode/'checkpoints'/f'{index+1:04d}.npz');sim=Simulation(a.episode/'scene.xml')
    mujoco.mj_setState(sim.model,sim.data,state['integration_state'],int(state['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    camera=RGBDCamera(sim)
    try:
        rgb,depth=camera.capture();iio.imwrite(a.output/'rgb.png',rgb);np.savez_compressed(a.output/'depth.npz',depth=depth)
        meta=json.loads((a.episode/'scene.json').read_text());groups={}
        for flag in (False,True):
            detections=observe_objects(camera,rgb,depth,split_round_instances=flag);rows=[]
            for detection in detections:
                objects=[o for o in meta['objects'] if o['category']==detection['category']]
                obj=min(objects,key=lambda o:np.linalg.norm(sim.data.xpos[sim.model.body(o['body']).id,:2]-detection['center'][:2]))
                true=sim.data.xpos[sim.model.body(obj['body']).id]
                rows.append(dict(detection=detection,nearest_body_for_offline_comparison=obj['body'],
                                 xy_origin_error_mm=1000*float(np.linalg.norm(true[:2]-detection['center'][:2]))))
            groups['split' if flag else 'legacy']=rows
    finally:camera.close()
    result=dict(scope='Same actual RGB/depth; body truth used only for offline diagnostic association, not detector inputs. Body-origin error is not necessarily the optimum grasp-point error for asymmetric geometry.',episode=str(a.episode),checkpoint=f'{index+1:04d}.npz',results=groups)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
