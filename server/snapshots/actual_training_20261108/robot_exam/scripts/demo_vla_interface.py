#!/usr/bin/env python3
"""Execute numeric mock-policy actions on real MuJoCo actuators, no VLA model."""
import argparse
import json
from pathlib import Path
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.vla.interface import OriginalUR5Inputs,OriginalUR5Outputs,ActionChunkExecutor


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output folder')
    a.output.mkdir(parents=True)
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml');arm=ArmController(sim);gripper=GripperController(sim)
    camera=RGBDCamera(sim)
    try:
        rgb,depth=camera.capture();camera.save(a.output/'before',rgb,depth)
        driver=[sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint(n).id]]
                for n in ('grip_left_driver_joint','grip_right_driver_joint')]
        state=np.r_[sim.data.qpos[arm.qadr],np.mean(driver)/.8]
        inputs=OriginalUR5Inputs()({'state':state,'front_rgb':rgb,'prompt':'Move the arm slightly and close the gripper.'})
        # Stand-in action generator tests the real control contract only.
        padded=np.zeros((10,32));padded[:,:7]=state;padded[:,0]+=.12;padded[:,6]=.25
        actions=OriginalUR5Outputs()({'actions':padded})['actions']
        executor=ActionChunkExecutor(sim,arm,gripper);executor.execute(actions,sim.data.time)
        rgb,depth=camera.capture();camera.save(a.output/'after',rgb,depth)
        movement=float(sim.data.qpos[arm.qadr[0]]-state[0])
        result={'status':'PASS' if movement>.07 and not sim.data.warning.number.any() else 'FAIL',
                'scope':'mock numeric policy interface, not pi0.5 inference or task success',
                'input_image_mask':{k:bool(v) for k,v in inputs['image_mask'].items()},
                'joint_motion_rad':movement,'sim_seconds':float(sim.data.time),
                'action_semantics':'ur5_absolute_joint_goal_v1','warnings':sim.data.warning.number.tolist(),
                'model_weights_loaded':False,'training_performed':False}
        (a.output/'trace.json').write_text(json.dumps(executor.trace))
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
    finally:camera.close()


if __name__=='__main__':main()
