#!/usr/bin/env python3
"""Real OpenPI websocket/msgpack loopback, using a mock policy without weights."""
import argparse
import json
from pathlib import Path
import threading
import time
import subprocess
import sys

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from websockets.sync.server import serve


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--physics',action='store_true')
    p.add_argument('--client-cli-output',type=Path);p.add_argument('--scene',type=Path)
    p.add_argument('--wrist-camera',action='store_true');p.add_argument('--parking',action='store_true');a=p.parse_args()
    received=[]
    def handle(connection):
        connection.send(msgpack_numpy.packb({'robot':'original_ur5','mock':True,'action_semantics':'ur5_absolute_joint_goal_v1',
            'actual_cameras':['front','wrist'] if a.wrist_camera else ['front']}))
        request=msgpack_numpy.unpackb(connection.recv());received.append(request)
        if a.wrist_camera:
            assert np.asarray(request['wrist_rgb']).shape==(480,640,3)
            assert not np.array_equal(request['wrist_rgb'],request['front_rgb'])
        q=np.asarray(request['state']);actions=np.tile(q,(10,1));actions[:,0]+=.12
        actions[:,6]=np.clip(actions[:,6],0,1)
        connection.send(msgpack_numpy.packb({'actions':actions}))
    with serve(handle,'127.0.0.1',0,max_size=4*1024**2) as server:
        port=server.socket.getsockname()[1]
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        if a.client_cli_output:
            from _bootstrap import ROOT
            command=[sys.executable,str(ROOT/'scripts/run_openpi_ur5_client.py'),
                '--host','127.0.0.1','--port',str(port),'--chunks','1','--prompt','Move slightly.',
                '--output',str(a.client_cli_output.resolve())]
            if a.scene:command.extend(['--scene',str(a.scene.resolve())])
            if a.wrist_camera:command.append('--wrist-camera')
            if a.parking:command.append('--parking')
            subprocess.run(command,check=True,timeout=30)
            queries=json.loads((a.client_cli_output/'queries.json').read_text())
            movement=queries[0]['measured_arm_after'][0]-queries[0]['measured_state_before'][0]
            assert len(queries)==1 and movement>.001
            server.shutdown();thread.join(timeout=5)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps({'status':'PASS','scope':'real client CLI -> loopback mock policy -> MuJoCo actuators',
                'queries':len(queries),'joint_motion_rad':movement,'model_weights_loaded':False,'training_performed':False},indent=2)+'\n')
            return
        client=WebsocketClientPolicy('127.0.0.1',port)
        state=np.array([0.,-1.,1.4,-2.,-1.57,0.,.2]);rgb=np.zeros((480,640,3),dtype=np.uint8)
        if a.physics:
            from _bootstrap import ROOT
            from robot_exam.simulation import Simulation
            from robot_exam.control.arm import ArmController,GripperController
            from robot_exam.perception.rgbd import RGBDCamera
            from robot_exam.vla.interface import ActionChunkExecutor
            sim=Simulation(ROOT/'scenes/ur5_gripper.xml');arm=ArmController(sim);gripper=GripperController(sim)
            camera=RGBDCamera(sim)
            try:rgb,_=camera.capture()
            finally:camera.close()
            state=np.r_[sim.data.qpos[arm.qadr],0.]
        start=time.monotonic()
        response=client.infer({'state':state,'front_rgb':rgb,
                               'prompt':'Pick the apple.','timestamp':0.})
        elapsed=time.monotonic()-start
        assert np.asarray(response['actions']).shape==(10,7)
        assert received[0]['front_rgb'].shape==(480,640,3)
        np.testing.assert_allclose(response['actions'][0],state+[.12,0,0,0,0,0,0])
        movement=None
        if a.physics:
            ActionChunkExecutor(sim,arm,gripper).execute(response['actions'],sim.data.time)
            movement=float(sim.data.qpos[arm.qadr[0]]-state[0])
            assert movement>.07 and not sim.data.warning.number.any()
        client._ws.close();server.shutdown();thread.join(timeout=5)
    report={'status':'PASS','protocol':'native OpenPI websocket + msgpack_numpy',
            'scope':'loopback mock policy only; no pi0.5 weights or inference',
            'action_shape':[10,7],'camera_shape':[480,640,3],'wall_round_trip_seconds':elapsed,
            'training_performed':False,'external_service_used':False,
            'physical_actuator_execution':a.physics,'physical_joint_motion_rad':movement}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
