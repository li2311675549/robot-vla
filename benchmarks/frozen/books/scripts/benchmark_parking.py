#!/usr/bin/env python3
"""Paired zero-wheel-speed / encoder-hold trials under actual arm motion."""
import argparse,json,sys,os
from pathlib import Path
import mujoco,numpy as np
from _bootstrap import ROOT
from benchmark_dock_stage7 import fixture
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.control.base import BaseController
from robot_exam.control.parking import WheelParkingHold


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a new experiment directory')
    out.mkdir(parents=True);rows=[]
    empty_scene,empty=fixture(7,'empty',out/'fixture_empty')
    fruit_scene,fruit=fixture(7,'loaded',out/'fixture_fruit')
    source=ROOT/'outputs/stage10/book_third_completed_recorded_v13'
    raw=np.load(source/'checkpoints/0003.npz')
    book={'integration_state':raw['integration_state'],'state_spec':int(raw['state_spec']),'body':'book_2'}
    for kind,scene,snapshot in [('empty',empty_scene,empty),('fruit',fruit_scene,fruit),
                               ('large_book',ROOT/'outputs/stage10/book_full_oracle_seed1011_v2/scene.xml',book)]:
      for profile,scale in enumerate((.5,1.,1.5)):
       for mode in ('zero_velocity','encoder_hold'):
        directory=out/f'{kind}_{profile}_{mode}';directory.mkdir();os.chdir(directory)
        sim=Simulation(scene);mujoco.mj_setState(sim.model,sim.data,snapshot['integration_state'],snapshot['state_spec']);mujoco.mj_forward(sim.model,sim.data)
        arm=ArmController(sim);base=BaseController(sim);grip=GripperController(sim);base.command(0,0)
        q=sim.data.qpos[arm.qadr].copy();arm.reference=q.copy();hold=WheelParkingHold(sim)
        anchor=base.pose().copy();start=float(sim.data.time);samples=[];error=None;bilateral=[]
        if mode=='encoder_hold':hold.begin()
        try:
            for i in range(round(14/sim.model.opt.timestep)):
                t=sim.data.time-start
                # First two seconds settle, then synchronized smooth arm motion.
                phase=max(0.,t-2.)
                delta=scale*np.array([.12,.04,-.04,0,0,.08])*np.sin(np.pi*phase/6)**2
                arm.command(q+delta,max_speed=.7);sim.step(sim.data.ctrl)
                if i%25==0:
                    if mode=='encoder_hold':m=hold.metrics()
                    else:
                        hold.anchor=anchor;m=hold.metrics()
                    samples.append(m)
                    if snapshot['body']:
                        contacts=[]
                        target=sim.model.body(snapshot['body']).id
                        for ci,c in enumerate(sim.data.contact):
                            ids=sim.model.geom_bodyid[c.geom]
                            if target in ids:
                                other=int(ids[1] if ids[0]==target else ids[0]);force=np.empty(6)
                                mujoco.mj_contactForce(sim.model,sim.data,ci,force)
                                contacts.append({'body':sim.model.body(other).name,'normal_force_n':float(force[0])})
                        bilateral.append(grip.bilateral(contacts))
        except Exception as exc:error={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
        finally:
            result={'kind':kind,'profile':profile,'mode':mode,'error':error,'sim_seconds':float(sim.data.time-start),
                    'maximum_position_error_m':max(x['position_error_m'] for x in samples),
                    'maximum_heading_error_rad':max(x['heading_error_rad'] for x in samples),
                    'maximum_linear_speed_m_s':max(x['linear_speed_m_s'] for x in samples),
                    'maximum_yaw_rate_rad_s':max(x['yaw_rate_rad_s'] for x in samples),
                    'bilateral_fraction':float(np.mean(bilateral)) if bilateral else None,
                    'warnings':sim.data.warning.number.tolist(),'samples':samples,
                    'scope':'identical restored physical states; book grip is an earlier real grasp, not a fresh autonomous pickup'}
            result['pass']=bool(error is None and result['maximum_position_error_m']<.010 and
                result['maximum_heading_error_rad']<.020 and (not bilateral or all(bilateral)) and
                (mode!='encoder_hold' or hold.ready))
            (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');rows.append(result)
            (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(kind,profile,mode,result['pass'],flush=True)
            hold.release()


if __name__=='__main__':main()
