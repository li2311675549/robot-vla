#!/usr/bin/env python3
"""Oracle local push-to-edge then physical side-pinch experiment; not full task."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

from _bootstrap import ROOT
from build_tasks import build
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State,TaskFailure
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.control.joint_path import clear_joint_path
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.evaluation.recording import VideoRecorder


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=811)
    p.add_argument('--rank',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retreat-base',action='store_true');p.add_argument('--collision-aware-pregrasp',action='store_true')
    p.add_argument('--record',action='store_true');p.add_argument('--rrt',action='store_true')
    p.add_argument('--adaptive-edge',action='store_true');p.add_argument('--initial-base-x',type=float,default=0.)
    p.add_argument('--initial-base-y',type=float);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output folder')
    a.output.mkdir(parents=True)
    scene,meta=build('drawer',a.seed,a.output/'scene.xml',book_profile='distinct_v2')
    sim=Simulation(scene);obj=next(o for o in meta['objects'] if o['rank']==a.rank)
    # Independent local fixture initialization only, before any physics step.
    # No task navigation success is claimed and book/slide qpos are untouched.
    sim.data.qpos[:2]=[a.initial_base_x,obj['initial'][1] if a.initial_base_y is None else a.initial_base_y];mujoco.mj_forward(sim.model,sim.data)
    view=mujoco.MjvCamera();view.lookat[:]=[.3,obj['initial'][1],.7]
    view.distance=2.5;view.azimuth=150;view.elevation=-25
    recorder=VideoRecorder(a.output/'video',camera=view,fps=5) if a.record else None
    r=TaskRuntime(sim,output=a.output,callback=recorder,wall_limit=300,sim_limit=180)
    r.config=json.loads((ROOT/'configs/task_defaults.json').read_text())
    body=sim.model.body(obj['body']).id;initial=sim.data.xpos[body].copy()
    result={'success':False,'scope':'local Oracle push/edge-grasp, initialized base; not complete book task',
            'seed':a.seed,'book_size_m':obj['size'],'initial_base':r.base.pose().tolist()}
    try:
        r.transition(State.MANIPULATE,'push_book_to_front_edge',timeout=120)
        r.hold(.6);r.gripper.command(1);r.hold(.5)
        center=sim.data.xpos[body].copy();R=sim.data.xmat[body].reshape(3,3)
        halfx=float(np.sum(np.abs(R[0,:2])*np.asarray(obj['size'][:2])/2))
        far=center[0]+halfx+.035;push_z=.731
        r.move([far,center[1],.84],seconds=3.)
        r.move([far,center[1],push_z],seconds=2.)
        # Match the *near edge*, not the same centre for differently sized
        # books. Closed-pad offset was measured in the preserved max-book run.
        push_x=.230+2*halfx if a.adaptive_edge else .305+halfx+.012
        result['push_tool_target_x']=push_x
        r.move([push_x,center[1],push_z],seconds=4.)
        r.move(r.arm.pose()[0]+[0,0,.12],seconds=2.)
        r.hold(.4)
        center=sim.data.xpos[body].copy();result['after_push_position']=center.tolist()
        result['push_displacement_m']=(center-initial).tolist()
        if center[0]>.34:raise TaskFailure('push_contact','Book did not move into edge-grasp bay')
        if center[2]<.725:raise TaskFailure('book_fell','Book fell off support during push')
        R=sim.data.xmat[body].reshape(3,3)
        halfx=float(np.sum(np.abs(R[0,:2])*np.asarray(obj['size'][:2])/2))
        grasp=center.copy();grasp[0]-=halfx-.015
        if grasp[0]>.27:raise TaskFailure('edge_clearance','No lower-finger clearance outside table')
        rotation=side_rotation();pre=grasp-[.075,0,0]
        if a.retreat_base:
            r.stow();r.backout(-.30)
        r.gripper.command(0)
        r.move([pre[0],pre[1],.90],rotation,seconds=3.)
        if a.rrt:
            from robot_exam.control.joint_rrt import plan_joint_rrt
            failures=[];path=None
            for initial in (r.qgoal,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
                try:
                    q,metrics=r.arm.solve(pre,rotation,seed=initial,max_iterations=500)
                    path,report=plan_joint_rrt(r.arm,q,allowed_bodies=[obj['body']])
                    break
                except RuntimeError as exc:failures.append(str(exc))
            if path is None:raise RuntimeError('All RRT/IK branches failed: '+' | '.join(failures))
            result['pregrasp_rrt']={**report,'ik':metrics,'waypoints':[q.tolist() for q in path]}
            for waypoint in path:r.joint_move(waypoint,seconds=2.)
        elif a.collision_aware_pregrasp:
            q,_=r.arm.solve_multi(pre,rotation,seed=r.qgoal,max_iterations=500,allowed_bodies=[obj['body']])
            for waypoint in clear_joint_path(r.arm,q):r.joint_move(waypoint,seconds=3.)
        else:r.move(pre,rotation,seconds=2.)
        r.move(grasp,rotation,seconds=1.5)
        r.gripper.command(1);r.hold(.8)
        if not r.bilateral(obj['body']):raise TaskFailure('grasp_contact','No bilateral book contact')
        r.move(grasp+[0,0,.10],rotation,seconds=2.)
        held=r.stable_grasp(obj['body'],seconds=.6)
        lift=float(sim.data.xpos[body,2]-center[2])
        result.update(lift_m=lift,bilateral_contact=bool(r.bilateral(obj['body'])),
                      success=bool(held and lift>.055 and not r.collision_events))
    except Exception as exc:
        result['failure']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        r.base.command(0,0);r.save()
        result.update(sim_seconds=float(sim.data.time),final_book_position=sim.data.xpos[body].tolist(),
                      warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
        if recorder:result['video']=recorder.close()
        result['final_contacts']=[{'geoms':[sim.model.geom(int(g)).name for g in contact.geom],
                                   'distance_m':float(contact.dist)} for contact in sim.data.contact]
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(sim.model,spec))
        mujoco.mj_getState(sim.model,sim.data,state,spec)
        np.savez_compressed(a.output/'final_state.npz',integration_state=state,state_spec=int(spec))
        camera=RGBDCamera(sim)
        try:camera.save(a.output/'final_observation',*camera.capture())
        finally:camera.close()
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
