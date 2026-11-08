#!/usr/bin/env python3
"""Compare physical release heights from the same real held-book checkpoint."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.tasks.drawer import _insert_side_book
from robot_exam.tasks.scoring import drawer_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--clearances',type=float,nargs='+',default=[.23,.08,.05]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--near-front',action='store_true');p.add_argument('--reposition-x',type=float)
    p.add_argument('--release-contact-trace',action='store_true')
    p.add_argument('--book-contact-profile',choices=('source','legacy_v1','wood_support_v2'),default='source')
    p.add_argument('--config',type=Path)
    p.add_argument('--insertion-index',type=int,help='For a documented partial episode with fewer insertion checkpoints')
    p.add_argument('--grasp-contact-trace',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve all physical probes')
    a.output.mkdir(parents=True);fsm=json.loads((a.episode/'fsm.json').read_text())
    indices=[i for i,x in enumerate(fsm) if x['reason']=='side_book_drawer_insertion'];index=indices[a.rank if a.insertion_index is None else a.insertion_index]
    checkpoint=a.episode/'checkpoints'/f'{index+1:04d}.npz';saved=np.load(checkpoint)
    meta=json.loads((a.episode/'scene.json').read_text());target=next(o for o in meta['objects'] if o['rank']==a.rank)
    settings=json.loads((a.episode/'config.json').read_text());results=[]
    if a.config:settings.update(json.loads(a.config.read_text()))
    model_path=a.episode/'scene.xml'
    if a.book_contact_profile!='source':
        from build_tasks import build
        model_path,_=build('drawer',meta['seed'],a.output/'contact_profile_scene.xml',book_profile=meta['book_profile'],
                           wrist_camera=settings.get('task_wrist_camera',False),book_contact_profile=a.book_contact_profile,
                           layout_profile=meta.get('layout_profile','legacy_v1'))
        old,new=mujoco.MjModel.from_xml_path(str(a.episode/'scene.xml')),mujoco.MjModel.from_xml_path(str(model_path))
        for field in ('qpos0','body_mass','body_inertia','geom_size','geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','actuator_forcerange'):
            np.testing.assert_array_equal(getattr(old,field),getattr(new,field))
    (a.output/'protocol.json').write_text(json.dumps(dict(scope='Paired local insertion, not autonomous complete task',source_episode=str(a.episode),checkpoint=str(checkpoint),clearances=a.clearances,near_front=a.near_front,reposition_x=a.reposition_x,release_contact_trace=a.release_contact_trace,book_contact_profile=meta.get('book_contact_profile','legacy_v1') if a.book_contact_profile=='source' else a.book_contact_profile,config_override=str(a.config) if a.config else None,controller_initialization='Same restored integration state, original gripper control retained, measured-joint reference initialized; normal wheel/arm controls only afterward.'),indent=2)+'\n')
    for clearance in a.clearances:
        folder=a.output/f'clearance{round(clearance*1000):03d}mm';folder.mkdir();sim=Simulation(model_path)
        mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
        r=TaskRuntime(sim,output=folder,sim_limit=sim.data.time+240)
        r.config={**settings,'side_book_release_clearance_m':clearance,'compress_bulk_logs':True,
                  'side_book_near_front_placement':a.near_front,'book_release_contact_trace':a.release_contact_trace}
        if r.config.get('side_book_release_clearance_by_rank') is not None:
            values=list(r.config['side_book_release_clearance_by_rank']);values[a.rank]=clearance
            r.config['side_book_release_clearance_by_rank']=values
        r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
        operations=[];result=dict(success=False,clearance_m=clearance)
        contact_rows=[]
        class GraspContactObserver:
            last=-float('inf')
            def before_step(self):pass
            def after_step(self):
                if sim.data.time-self.last<.02-1e-9:return
                self.last=float(sim.data.time);m,d=sim.model,sim.data;body=m.body(target['body']).id
                tip,rotation=r.arm.pose();external=[]
                for i,c in enumerate(d.contact):
                    names=[m.body(int(m.geom_bodyid[g])).name or '' for g in c.geom]
                    if not(target['body'] in names or (any(n.startswith(('grip_','wrist_')) for n in names)
                                                       and any(n.startswith(('drawer_','cabinet_')) for n in names))):
                        continue
                    force=np.empty(6);mujoco.mj_contactForce(m,d,i,force)
                    external.append({'bodies':names,'geoms':[m.geom(int(g)).name for g in c.geom],
                                     'distance_m':float(c.dist),'normal_force_n':float(force[0]),
                                     'tangential_force_n':float(np.linalg.norm(force[1:3])),
                                     'position':c.pos.tolist()})
                contact_rows.append({'time':float(d.time),'tip':tip.tolist(),'rotation':rotation.tolist(),
                                     'book_position':d.xpos[body].tolist(),'book_rotation':d.xmat[body].reshape(3,3).tolist(),
                                     'relative_position':(rotation.T@(d.xpos[body]-tip)).tolist(),
                                     'bilateral':r.bilateral(target['body']),'contacts':external})
        observer=GraspContactObserver()
        if a.grasp_contact_trace:sim.add_step_controller(observer)
        try:
            if a.reposition_x is not None:
                from robot_exam.tasks.runtime import State
                r.transition(State.NAVIGATE,'local_insertion_physical_reposition',timeout=60)
                r.payload_motion_profile='book';r.begin_payload(target['body'],r.arm.pose()[1],world_stabilize=False)
                try:r.dock([a.reposition_x,.5,0],timeout=45)
                finally:r.end_payload()
            grasp=dict(rotation=side_rotation(),lift_m=0.,scope='Restored real grasp for local insertion test')
            _insert_side_book(r,meta,a.rank,{'area':target['area']},target,grasp,{},operations)
            score=drawer_score(sim,meta,np.asarray(meta['start']))
            item=next(x for x in score['books'] if x['book']==target['body'])
            result.update(success=bool(item['inside_correct_layer'] and abs(item['opening'])<.012 and not r.collision_events and not sim.data.warning.number.any()),book_score=item)
        except Exception as exc:result.update(error=str(exc),error_type=type(exc).__name__)
        finally:
            sim.remove_step_controller(observer)
            if contact_rows:
                import gzip
                with gzip.open(folder/'grasp_contact_trace.json.gz','wt') as f:json.dump(contact_rows,f)
            integration=np.empty(mujoco.mj_stateSize(sim.model,mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(sim.model,sim.data,integration,mujoco.mjtState.mjSTATE_INTEGRATION)
            np.savez_compressed(folder/'final_state.npz',integration_state=integration,state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
                                time=sim.data.time,qgoal=r.qgoal,arm_reference=r.arm.reference,wheel_applied=r.wheel_applied)
            result.update(sim_seconds=float(sim.data.time)-float(saved['time']),warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
            r.save();(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        results.append(result);(a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':main()
