#!/usr/bin/env python3
"""Physical closing from an explicitly open time-zero fixture, not a full task."""
import argparse
import json
from pathlib import Path
import time
import traceback
import faulthandler
import xml.etree.ElementTree as ET

from build_tasks import build
import mujoco
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.door_latch import closing_station,close_latch
from robot_exam.evaluation.recording import VideoRecorder


def main():
    faulthandler.enable();faulthandler.dump_traceback_later(30,repeat=True)
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1901);p.add_argument('--record',action='store_true')
    p.add_argument('--angle',type=float,default=1.56)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous physical probes')
    a.output.mkdir(parents=True)
    path,meta=build('door',a.seed,a.output/'scene.xml',door_profile='aligned_handles_v3')
    tree=ET.parse(path);key=tree.find('keyframe/key');model=mujoco.MjModel.from_xml_path(str(path))
    q=np.fromstring(key.get('qpos'),sep=' ');base=closing_station(meta,a.angle)
    q[:2]=base[:2];q[3:7]=[np.cos(base[2]/2),0,0,np.sin(base[2]/2)]
    q[model.jnt_qposadr[model.joint('door_hinge').id]]=a.angle
    key.set('qpos',' '.join(map(str,q)));tree.write(path,encoding='unicode',xml_declaration=True)
    sim=Simulation(path)
    camera=mujoco.MjvCamera();camera.lookat[:]=[-.3,0.,.9];camera.distance=3.8;camera.azimuth=125;camera.elevation=-22
    recorder=VideoRecorder(a.output/'video',camera=camera,fps=5) if a.record else None
    r=TaskRuntime(sim,output=a.output,callback=recorder,sim_limit=1200,wall_limit=1800)
    r.config={'arm_endpoint_convergence_seconds':8.}
    result={'scope':'LOCAL_CLOSE_ONLY: initialized open door and base work pose; excluded from full five-metre success rates',
            'initial_angle':a.angle,'initial_base':base.tolist(),'seed':a.seed}
    started=time.monotonic()
    try:
        r.hold(.6)
        result.update(close_latch(r,meta,already_at_station=True))
        result['scope']='LOCAL_CLOSE_ONLY: initialized open door, not a full five-metre task'
    except BaseException as error:
        result.update(success=False,error=str(error),traceback=traceback.format_exc(),
                      failure_category=getattr(error,'category','exception'),failure_stage=r.stage)
        result['tip_position']=r.arm.pose()[0].tolist()
        result['tip_rotation']=r.arm.pose()[1].tolist()
        result['handle_position']=sim.data.geom_xpos[sim.model.geom('inner_handle').id].tolist()
        result['door_contacts']=r.contacts('door_leaf')
        result['all_contacts']=[{'geoms':[sim.model.geom(int(g)).name for g in c.geom],
                                'bodies':[sim.model.body(int(sim.model.geom_bodyid[g])).name for g in c.geom],
                                'distance':float(c.dist)} for c in sim.data.contact]
    finally:
        r.base.command(0,0);r.save()
        state=np.empty(mujoco.mj_stateSize(sim.model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(sim.model,sim.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez_compressed(a.output/'final_state.npz',integration_state=state,
                            state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
                            arm_reference=r.arm.reference,qgoal=r.qgoal)
        if recorder:result['video']=recorder.close()
    result.update(sim_seconds=sim.data.time,wall_seconds=time.monotonic()-started,
                  warnings=sim.data.warning.number.tolist(),collisions=r.collision_events,
                  final_angle=float(sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint('door_hinge').id]]))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({k:v for k,v in result.items() if k not in ('traceback','video','close_trace')},indent=2))
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
