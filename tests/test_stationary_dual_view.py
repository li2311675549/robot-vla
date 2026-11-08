import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from robot_exam.simulation import ROOT,Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.control.parking import WheelParkingHold
from robot_exam.vla.interface import ActionChunkExecutor,OriginalUR5Inputs
from robot_exam.perception.wrist_mount import add_wrist_camera


def mobile_scene(tmp_path):
    root=ET.parse(ROOT/'scenes/mobile_manipulator.xml').getroot()
    add_wrist_camera(root)
    for e in root.findall('.//mesh'):
        if e.get('file'):e.set('file',str((ROOT/'scenes'/e.get('file')).resolve()))
    path=tmp_path/'mobile.xml';ET.ElementTree(root).write(path)
    return path


def test_physical_parking_is_shared_with_vla_and_releases_for_navigation(tmp_path):
    sim=Simulation(mobile_scene(tmp_path));arm=ArmController(sim);grip=GripperController(sim)
    sim.step(sim.data.ctrl,500);hold=WheelParkingHold(sim)
    executor=ActionChunkExecutor(sim,arm,grip,base_hold=hold)
    actions=np.tile(np.r_[arm.reference,.2],(3,1));actions[:,0]+=.08
    with pytest.raises(RuntimeError,match='parking'):executor.submit(actions,sim.data.time)
    hold.begin()
    for _ in range(1100):arm.command(arm.reference);sim.step(sim.data.ctrl)
    assert hold.ready
    executor.execute(actions,sim.data.time)
    from robot_exam.perception.rgbd import RGBDCamera
    from robot_exam.vla.client import query_and_execute
    class DualMock:
        def infer(self,obs):
            assert obs['front_rgb'].shape==obs['wrist_rgb'].shape==(480,640,3)
            assert not np.array_equal(obs['front_rgb'],obs['wrist_rgb'])
            action=np.asarray(obs['state']).copy();action[6]=.2;action[0]+=.03
            return {'actions':np.tile(action,(10,1))}
    front=RGBDCamera(sim);wrist=RGBDCamera(sim,camera_name='wrist_rgb')
    try:
        query_and_execute(sim,arm,grip,front,DualMock(),'Move slightly.',executor=executor,
                          execute_actions=2,wrist_camera=wrist)
    finally:front.close();wrist.close()
    assert max(x['position_error_m'] for x in hold.samples)<.010
    assert max(x['heading_error_rad'] for x in hold.samples)<.020
    before=hold.base.pose().copy();hold.release();hold.base.command(.12,0)
    for _ in range(500):arm.command(arm.reference);sim.step(sim.data.ctrl)
    assert sim.data.time>4
    assert np.linalg.norm(hold.base.pose()[:2]-before[:2])>.03


def test_parking_fault_discards_policy_chunk_and_brakes(tmp_path):
    from robot_exam.control.parking import ParkingViolation
    sim=Simulation(mobile_scene(tmp_path));arm=ArmController(sim);grip=GripperController(sim)
    sim.step(sim.data.ctrl,500);hold=WheelParkingHold(sim);hold.begin()
    for _ in range(1100):arm.command(arm.reference);sim.step(sim.data.ctrl)
    executor=ActionChunkExecutor(sim,arm,grip,base_hold=hold)
    action=np.tile(np.r_[arm.reference,.2],(10,1));executor.submit(action,sim.data.time)
    # Inject a controller-reference fault, never a robot pose reset or force.
    hold.anchor[0]+=.03
    with pytest.raises(ParkingViolation):
        for _ in range(100):executor.step()
    assert executor.chunk is None and hold.faulted
    np.testing.assert_array_equal(sim.data.ctrl[hold.base.ids],np.zeros(4))
    with pytest.raises(RuntimeError,match='parking'):executor.submit(action,sim.data.time)


def test_real_dual_camera_sync_and_parked_control_replay(tmp_path):
    from robot_exam.data.episodes import EpisodeRecorder,validate_episode
    import imageio.v3 as iio
    sim=Simulation(mobile_scene(tmp_path));arm=ArmController(sim);grip=GripperController(sim)
    sim.step(sim.data.ctrl,500);hold=WheelParkingHold(sim);hold.begin()
    for _ in range(1100):arm.command(arm.reference);sim.step(sim.data.ctrl)
    directory=tmp_path/'episode';logger=EpisodeRecorder(sim,directory,'dual_sensor_test',1200,'unit_fixture',initial_arm_reference=arm.reference)
    goal=arm.reference.copy();goal[0]+=.06
    for _ in range(250):
        arm.command(goal);grip.command(.2)
        logger.capture_pre_step('dual_test',arm_target=goal,arm_reference=arm.reference)
        sim.step(sim.data.ctrl)
    logger.close(False);hold.release()
    result=validate_episode(directory,replay=True)
    assert result['status']=='PASS',result
    assert result['max_replay_qpos_error']<1e-7
    meta=json.loads((directory/'episode.json').read_text())
    assert meta['sensor_schema']=='front_wrist_rgb_v1'
    rows=[json.loads(x) for x in (directory/'frames.jsonl').read_text().splitlines()]
    for row in rows:
        assert row['camera_timestamps']=={'front':row['timestamp'],'wrist':row['timestamp']}
        front=iio.imread(directory/row['rgb']);wrist=iio.imread(directory/row['rgb_wrist'])
        assert front.shape==wrist.shape==(480,640,3)
        assert not np.array_equal(front,wrist)
    transforms=[np.linalg.inv(np.asarray(row['camera_extrinsics']['front']))@
                np.asarray(row['camera_extrinsics']['wrist']) for row in rows]
    assert np.linalg.norm(transforms[-1]-transforms[0])>1e-4
    packed=OriginalUR5Inputs(require_wrist=True)({'state':rows[0]['observation.state'],
        'front_rgb':front,'wrist_rgb':wrist,'prompt':'Move carefully.'})
    assert packed['image_mask']=={'base_0_rgb':True,'left_wrist_0_rgb':True,'right_wrist_0_rgb':False}
    np.testing.assert_array_equal(packed['image']['left_wrist_0_rgb'],wrist)
    with pytest.raises(ValueError,match='wrist'):
        OriginalUR5Inputs(require_wrist=True)({'state':rows[0]['observation.state'],'front_rgb':front,'prompt':'Move.'})
