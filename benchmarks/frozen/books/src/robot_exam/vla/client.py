"""Sensor-only policy queries connected to numeric UR5 actuator commands."""
import time
import numpy as np
from robot_exam.vla.interface import OriginalUR5Outputs,ActionChunkExecutor


def connect_openpi_policy(host='127.0.0.1',port=8000,compression=None,noise_seed=None):
    """Reuse native wire format, with bounded connection/response timeouts."""
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import websockets.sync.client
    from robot_exam.vla.interface import ACTION_SEMANTICS
    if noise_seed is not None and (type(noise_seed) is not int or not 0 <= noise_seed < 2**31):
        raise ValueError('Evaluation noise seed must be an integer in[0,2**31)')
    headers = {'X-Robot-Exam-Evaluation-Seed':str(noise_seed)} if noise_seed is not None else None
    class BoundedClient(WebsocketClientPolicy):
        def _wait_for_server(self):
            connection=websockets.sync.client.connect(self._uri,compression=compression,max_size=None,open_timeout=5,
                                                     additional_headers=headers)
            return connection,msgpack_numpy.unpackb(connection.recv(timeout=5))
        def infer(self,obs):
            self._ws.send(self._packer.pack(obs));response=self._ws.recv(timeout=2)
            if isinstance(response,str):raise RuntimeError(response)
            return msgpack_numpy.unpackb(response)
    policy=BoundedClient(host,port)
    if policy.get_server_metadata().get('action_semantics')!=ACTION_SEMANTICS:
        policy._ws.close();raise ValueError('Server action semantics do not match original UR5 client')
    return policy


def query_and_execute(sim,arm,gripper,camera,policy,prompt,executor=None,execute_actions=1,max_latency_seconds=.4,wrist_camera=None,network_image_size=None,observation_callback=None,interpolate_lookahead=False):
    if not 1<=execute_actions<=50:raise ValueError('Invalid executed action count')
    executor=executor or ActionChunkExecutor(sim,arm,gripper)
    rgb,_depth=camera.capture();timestamp=float(sim.data.time)
    drivers=[sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint(name).id]]
             for name in ('grip_left_driver_joint','grip_right_driver_joint')]
    state=np.r_[sim.data.qpos[arm.qadr],np.mean(drivers)/.8]
    observation={'front_rgb':rgb,'state':state,'prompt':prompt,'timestamp':timestamp}
    if wrist_camera is not None:observation['wrist_rgb']=wrist_camera.capture_rgb()
    if network_image_size is not None:
        if network_image_size!=224:raise ValueError('Use the trained224 image transform for network transport')
        from openpi_client.image_tools import resize_with_pad
        for name in ('front_rgb','wrist_rgb'):
            if name in observation:observation[name]=resize_with_pad(observation[name],224,224)
    if observation_callback is not None:observation_callback(observation)
    start=time.monotonic()
    response=policy.infer(observation)
    latency=time.monotonic()-start
    if latency>max_latency_seconds:raise TimeoutError(f'Policy response too stale: {latency:.6f}s > {max_latency_seconds:.3f}s; no actions executed')
    actions=OriginalUR5Outputs()(response)['actions']
    if interpolate_lookahead:executor.execute_prefix(actions,timestamp,min(execute_actions,len(actions)))
    else:executor.execute(actions[:execute_actions],timestamp)
    return {'observation_time':timestamp,'sim_time_after':float(sim.data.time),
            'query_wall_seconds':latency,'predicted_actions':len(actions),
            'executed_actions':min(execute_actions,len(actions)),
            'network_image_size':network_image_size,'actions':actions.tolist(),
            'interpolate_lookahead':bool(interpolate_lookahead),
            'gripper_projection':response.get('gripper_projection'),
            'server_timing':response.get('server_timing'),
            'measured_state_before':state.tolist(),'measured_arm_after':sim.data.qpos[arm.qadr].tolist()}
