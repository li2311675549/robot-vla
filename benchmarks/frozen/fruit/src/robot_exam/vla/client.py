"""Sensor-only policy queries connected to numeric UR5 actuator commands."""
import time
import numpy as np
from robot_exam.vla.interface import OriginalUR5Outputs,ActionChunkExecutor


def connect_openpi_policy(host='127.0.0.1',port=8000):
    """Reuse native wire format, with bounded connection/response timeouts."""
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import websockets.sync.client
    from robot_exam.vla.interface import ACTION_SEMANTICS
    class BoundedClient(WebsocketClientPolicy):
        def _wait_for_server(self):
            connection=websockets.sync.client.connect(self._uri,compression=None,max_size=None,open_timeout=5)
            return connection,msgpack_numpy.unpackb(connection.recv(timeout=5))
        def infer(self,obs):
            self._ws.send(self._packer.pack(obs));response=self._ws.recv(timeout=2)
            if isinstance(response,str):raise RuntimeError(response)
            return msgpack_numpy.unpackb(response)
    policy=BoundedClient(host,port)
    if policy.get_server_metadata().get('action_semantics')!=ACTION_SEMANTICS:
        policy._ws.close();raise ValueError('Server action semantics do not match original UR5 client')
    return policy


def query_and_execute(sim,arm,gripper,camera,policy,prompt,executor=None,execute_actions=1,max_latency_seconds=.4,wrist_camera=None):
    if not 1<=execute_actions<=50:raise ValueError('Invalid executed action count')
    executor=executor or ActionChunkExecutor(sim,arm,gripper)
    rgb,_depth=camera.capture();timestamp=float(sim.data.time)
    drivers=[sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint(name).id]]
             for name in ('grip_left_driver_joint','grip_right_driver_joint')]
    state=np.r_[sim.data.qpos[arm.qadr],np.mean(drivers)/.8]
    observation={'front_rgb':rgb,'state':state,'prompt':prompt,'timestamp':timestamp}
    if wrist_camera is not None:observation['wrist_rgb']=wrist_camera.capture_rgb()
    start=time.monotonic()
    response=policy.infer(observation)
    latency=time.monotonic()-start
    if latency>max_latency_seconds:raise TimeoutError('Policy response too stale; no actions executed')
    actions=OriginalUR5Outputs()(response)['actions']
    executor.execute(actions[:execute_actions],timestamp)
    return {'observation_time':timestamp,'sim_time_after':float(sim.data.time),
            'query_wall_seconds':latency,'predicted_actions':len(actions),
            'executed_actions':min(execute_actions,len(actions)),
            'measured_state_before':state.tolist(),'measured_arm_after':sim.data.qpos[arm.qadr].tolist()}
