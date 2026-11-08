#!/usr/bin/env python3
"""Paired passive book drop tests, NOT complete robot manipulation episodes."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from build_tasks import build
from robot_exam.simulation import Simulation
from robot_exam.tasks.scoring import body_bounds


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier contact tests')
    args.output.mkdir(parents=True);results=[]
    for profile in ('legacy_v1','wood_support_v2'):
        folder=args.output/profile;folder.mkdir()
        path,metadata=build('drawer',200004,folder/'scene.xml',book_profile='distinct_v2',book_contact_profile=profile)
        for rank in range(3):
            for clearance in (.10,.23):
                sim=Simulation(path);model,data=sim.model,sim.data;name=f'book_{rank}'
                body=model.body(name).id;adr=int(model.jnt_qposadr[model.body_jntadr[body]])
                slide=int(model.jnt_qposadr[model.joint(f'drawer_slide_{rank}').id])
                # Explicit independent test initialization, never a live task.
                data.qpos[slide]=.27
                data.qpos[adr:adr+3]=[.45,.5,metadata['drawer_bottoms'][rank]+.008+metadata['objects'][next(i for i,o in enumerate(metadata['objects']) if o['rank']==rank)]['size'][2]/2+clearance]
                data.qpos[adr+3:adr+7]=[np.sqrt(.5),0,0,np.sqrt(.5)]
                mujoco.mj_forward(model,data)
                maximum_penetration=0.;trace=[]
                for step in range(1000):
                    sim.step(data.ctrl)
                    for contact in data.contact:
                        geoms=[model.geom(int(g)).name or '' for g in contact.geom]
                        if f'drawer_{rank}_bottom' in geoms and f'book_{rank}_cover' in geoms:
                            maximum_penetration=max(maximum_penetration,-float(contact.dist))
                    if step%5==0:trace.append([float(data.time),*data.xpos[body].tolist()])
                lower,upper=body_bounds(sim,name);bottom=metadata['drawer_bottoms'][rank]
                dof=int(model.jnt_dofadr[model.body_jntadr[body]])
                supported=bool(lower[2]>bottom-.003 and upper[2]<bottom+.14 and
                               np.all(lower[:2]>[.318,.31]) and np.all(upper[:2]<[.59,.69]) and
                               np.linalg.norm(data.qvel[dof:dof+3])<.02)
                row={'profile':profile,'rank':rank,'drop_clearance_m':clearance,'supported':supported,
                     'maximum_bottom_penetration_m':maximum_penetration,'final_lower':lower.tolist(),
                     'final_upper':upper.tolist(),'warnings':data.warning.number.tolist()}
                results.append(row)
                (folder/f'rank{rank}_drop{round(clearance*1000)}mm.json').write_text(json.dumps({'result':row,'trace':trace},indent=2)+'\n')
                print(json.dumps(row),flush=True)
    result={'scope':'Independent passive support test. Initialization sets book/drop and drawer open states; not autonomous grasp/place or full-task success.',
            'physics_step_seconds':.002,'changed_parameters':'Only book cover and drawer bottom solref20ms ->4ms; original friction, mass, geometry and force limits retained.',
            'results':results}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
