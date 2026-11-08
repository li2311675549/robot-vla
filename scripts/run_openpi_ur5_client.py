#!/usr/bin/env python3
"""Bounded real policy-client control loop; server/checkpoint are user-provided."""
import argparse
import json
from pathlib import Path
import sys
from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--client-source',type=Path,default=Path('/home/li/vla/openpi/packages/openpi-client/src'))
    p.add_argument('--scene',type=Path,default=ROOT/'scenes/ur5_gripper.xml')
    p.add_argument('--chunks',type=int,default=5);p.add_argument('--execute-actions',type=int,default=1)
    p.add_argument('--prompt',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wrist-camera',action='store_true');p.add_argument('--parking',action='store_true');a=p.parse_args()
    if not 1<=a.chunks<=50:raise ValueError('Use 1..50 bounded queries')
    if a.output.exists():raise FileExistsError('Use a new output folder')
    sys.path.insert(0,str(a.client_source))
    from robot_exam.simulation import Simulation
    from robot_exam.control.arm import ArmController,GripperController
    from robot_exam.perception.rgbd import RGBDCamera
    from robot_exam.vla.interface import ActionChunkExecutor,ACTION_SEMANTICS
    from robot_exam.vla.client import query_and_execute,connect_openpi_policy
    policy=connect_openpi_policy(a.host,a.port)
    a.output.mkdir(parents=True)
    sim=Simulation(a.scene);arm=ArmController(sim);gripper=GripperController(sim)
    hold=None
    if a.parking:
        from robot_exam.control.parking import WheelParkingHold
        hold=WheelParkingHold(sim);hold.begin()
        for _ in range(1100):arm.command(arm.reference);sim.step(sim.data.ctrl)
    executor=ActionChunkExecutor(sim,arm,gripper,base_hold=hold);camera=RGBDCamera(sim);trace=[]
    wrist=RGBDCamera(sim,camera_name='wrist_rgb') if a.wrist_camera else None
    expected=policy.get_server_metadata().get('actual_cameras',['front'])
    if ('wrist' in expected)!=a.wrist_camera:raise ValueError('Server/client camera configuration mismatch')
    try:
        for _ in range(a.chunks):trace.append(query_and_execute(sim,arm,gripper,camera,policy,a.prompt,
            executor=executor,execute_actions=a.execute_actions,wrist_camera=wrist))
    finally:
        camera.close();policy._ws.close()
        if wrist:wrist.close()
        if hold:hold.release()
        (a.output/'queries.json').write_text(json.dumps(trace,indent=2)+'\n')
    print({'queries':len(trace),'sim_seconds':float(sim.data.time),'task_success':'NOT_EVALUATED'})


if __name__=='__main__':main()
