import sys,math
from types import SimpleNamespace
import mujoco,numpy as np
from robot_exam.simulation import ROOT,Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.perception.door_handle import locate_handle


def test_handle_detector_uses_rgbd_without_model_access(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    scene,meta=build('door',1091,tmp_path/'door.xml',door_profile='multistart_v2')
    sim=Simulation(scene);base=meta['opening_base']
    sim.data.qpos[:2]=base[:2];sim.data.qpos[3:7]=[math.cos(base[2]/2),0,0,math.sin(base[2]/2)]
    mujoco.mj_forward(sim.model,sim.data);runtime=TaskRuntime(sim);runtime.hold(.6)
    camera=RGBDCamera(sim)
    try:
        rgb,depth=camera.capture()
        sensor_only=SimpleNamespace(unproject=camera.unproject)
        result=locate_handle(sensor_only,rgb,depth)
        truth=sim.data.geom_xpos[sim.model.geom('outer_handle').id]
        assert np.linalg.norm(np.asarray(result['center'])-truth)<.015
        assert result['axis'][2]>.95
    finally:camera.close()
