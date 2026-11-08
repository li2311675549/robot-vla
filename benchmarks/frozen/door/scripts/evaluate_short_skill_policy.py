#!/usr/bin/env python3
"""One physical score for an Oracle expert or numeric sensor-only policy backend."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.skills.pick_place import PickPlace
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.vla.interface import ActionChunkExecutor
from robot_exam.vla.client import query_and_execute,connect_openpi_policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--policy',choices=['expert','mock_hold','openpi'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--client-source',type=Path,default=Path('/home/li/vla/openpi/packages/openpi-client/src'))
    a=p.parse_args()
    if not 0<a.seconds<=120:raise ValueError('Bounded short-skill budget must be <=120 seconds')
    if a.output.exists():raise FileExistsError('Use a new evaluation output')
    a.output.mkdir(parents=True);sim=Simulation(a.fixture/'scene.xml');sim.step(sim.data.ctrl,500)
    meta=json.loads((a.fixture/'episode/episode.json').read_text());reference=json.loads((a.fixture/'skill_result.json').read_text())
    body=sim.model.body('object').id;dof=sim.model.jnt_dofadr[sim.model.joint('object_free').id]
    initial=sim.data.xpos[body].copy();goal=np.asarray(reference['drop']);gripper=GripperController(sim)
    start=float(sim.data.time);wall=time.monotonic();samples=[];best_hold=0.;hold=0.;step=sim.step
    def measured_step(ctrl,count=1):
        nonlocal hold,best_hold
        if count!=1:raise RuntimeError('Unified evaluator requires single-step actuator execution')
        if sim.data.time-start>a.seconds or time.monotonic()-wall>300:
            raise TimeoutError('Short-skill evaluation budget exceeded')
        value=step(ctrl,count);lift=float(sim.data.xpos[body,2]-initial[2]);bilateral=gripper.bilateral()
        hold=hold+sim.model.opt.timestep if lift>.05 and bilateral else 0.;best_hold=max(best_hold,hold)
        samples.append(lift)
        return value
    sim.step=measured_step;error=None;camera=None;policy=None
    try:
        if a.policy=='expert':
            class OracleReleaseExpert(PickPlace):
                def move(self,position,seconds,stage):
                    position=np.asarray(position).copy()
                    if stage=='lower':position[:2]-=(self.sim.data.xpos[self.object_body]-self.arm.pose()[0])[:2]
                    return super().move(position,seconds,stage)
            OracleReleaseExpert(sim).run(initial,goal)
        else:
            arm=ArmController(sim);executor=ActionChunkExecutor(sim,arm,gripper);camera=RGBDCamera(sim)
            class HoldPolicy:
                def infer(self,obs):
                    action=np.asarray(obs['state']).copy();action[6]=np.clip(action[6],0,1)
                    return {'actions':np.tile(action,(10,1))}
            if a.policy=='mock_hold':policy=HoldPolicy()
            else:
                sys.path.insert(0,str(a.client_source));policy=connect_openpi_policy(a.host,a.port)
            while sim.data.time-start<a.seconds:
                query_and_execute(sim,arm,gripper,camera,policy,meta['language_instruction'],executor=executor)
        # The expert's own routine does not replace the shared final score.
    except Exception as exc:error={'category':type(exc).__name__,'message':str(exc)}
    finally:
        if camera:camera.close()
        if a.policy=='openpi' and policy is not None:policy._ws.close()
    position=sim.data.xpos[body].copy();placement=float(np.linalg.norm(position[:2]-goal[:2]));speed=float(np.linalg.norm(sim.data.qvel[dof:dof+3]))
    success=bool(error is None and max(samples,default=0)>.055 and best_hold>=1.5
                 and placement<.04 and abs(position[2]-goal[2])<.01 and speed<.01
                 and not gripper.contacts() and not sim.data.warning.number.any())
    result={'success':success,'backend':a.policy,'scope':'fixed-base short skill; mock_hold is not VLA inference',
        'fixture':str(a.fixture.resolve()),'seed':meta['seed'],'skill':meta['skill'],'error':error,
        'sim_seconds':float(sim.data.time-start),'wall_seconds':time.monotonic()-wall,
        'maximum_lift_m':max(samples,default=0),'continuous_bilateral_lift_hold_seconds':best_hold,
        'placement_error_m':placement,'final_speed_m_s':speed,'final_position':position.tolist(),
        'warnings':sim.data.warning.number.tolist(),
        'actual_model_inference':'server identity/weights require separate verification' if a.policy=='openpi' else False}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
