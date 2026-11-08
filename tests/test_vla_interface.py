import numpy as np
import pytest

from robot_exam.simulation import ROOT,Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.vla.interface import (OriginalUR5Inputs,OriginalUR5Outputs,
    ActionChunkExecutor,StateActionNormalizer,delta_actions)


def test_one_real_camera_and_delta_roundtrip():
    state=np.array([.2,-1.,1.4,-2.,1.,.3,.4])
    actions=np.tile(state,(10,1));actions[:,:6]+=.02;actions[:,6]=.7
    packed=OriginalUR5Inputs()({'state':state,'front_rgb':np.ones((3,32,48),dtype=np.float32),
                               'prompt':'Pick the apple.','actions':actions})
    assert packed['image']['base_0_rgb'].shape==(32,48,3)
    assert packed['image_mask']=={'base_0_rgb':True,'left_wrist_0_rgb':False,'right_wrist_0_rgb':False}
    assert not packed['image']['left_wrist_0_rgb'].any()
    delta=delta_actions(actions,state)
    np.testing.assert_allclose(delta[:,:6],.02,atol=1e-12)
    np.testing.assert_allclose(delta[:,6],.7)
    np.testing.assert_allclose(delta_actions(delta,state,inverse=True),actions)
    norm=StateActionNormalizer(state,np.ones(7)*.2)
    np.testing.assert_allclose(norm.denormalize(norm.normalize(actions)),actions)
    padded=np.pad(actions,((0,0),(0,25)))
    np.testing.assert_allclose(OriginalUR5Outputs()({'actions':padded})['actions'],actions,atol=1e-6)


def test_mock_chunk_moves_real_joints_with_limits_and_no_pose_reset():
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml')
    arm=ArmController(sim);gripper=GripperController(sim)
    executor=ActionChunkExecutor(sim,arm,gripper)
    before=sim.data.qpos[arm.qadr].copy()
    actions=np.tile(np.r_[arm.reference,.25],(10,1));actions[:,0]+=.12
    executor.execute(actions,sim.data.time)
    assert sim.data.time>1.99 and sim.data.qpos[arm.qadr[0]]-before[0]>.07
    velocities=np.array([r['velocity'] for r in executor.trace])
    assert np.max(np.abs(velocities))<=.7+1e-9
    assert np.max(np.abs(np.diff(velocities,axis=0)))/sim.model.opt.timestep<=1.4+1e-9
    assert not sim.data.warning.number.any()
    prior_ctrl=sim.data.ctrl.copy()
    bad=actions.copy();bad[2,0]=100.
    with pytest.raises(ValueError,match='limits'):executor.submit(bad,sim.data.time)
    np.testing.assert_array_equal(sim.data.ctrl,prior_ctrl)
    with pytest.raises(ValueError,match='Stale'):executor.submit(actions,sim.data.time-2.)
    bad=actions.copy();bad[0,6]=1.2
    with pytest.raises(ValueError,match='Gripper'):executor.submit(bad,sim.data.time)


def test_schema3_separates_expert_reference_and_actuator(tmp_path):
    from robot_exam.data.episodes import EpisodeRecorder,validate_episode
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml')
    arm=ArmController(sim);gripper=GripperController(sim)
    recorder=EpisodeRecorder(sim,tmp_path,'interface',99,'mock')
    goal=arm.reference.copy();goal[0]+=.1
    for _ in range(150):
        arm.command(goal);gripper.command(.2)
        recorder.capture_pre_step('mock_interface',arm_target=goal,arm_reference=arm.reference,max_speed=.7)
        sim.step(sim.data.ctrl)
    recorder.close(False)
    raw=np.load(tmp_path/'replay.npz')
    assert np.max(np.abs(raw['command_targets'][0,:6]-raw['arm_references'][0]))>.09
    assert np.max(np.abs(raw['controls'][:,arm.actuators]-raw['arm_references']))>1e-6
    result=validate_episode(tmp_path,replay=True)
    assert result['status']=='PASS' and result['max_replay_qpos_error']<1e-9
    assert not result['success_label']


def test_policy_bridge_uses_only_measured_observations_and_numeric_actions():
    from robot_exam.perception.rgbd import RGBDCamera
    from robot_exam.vla.client import query_and_execute
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml');arm=ArmController(sim);gripper=GripperController(sim)
    class MockPolicy:
        def infer(self,obs):
            assert set(obs)=={'front_rgb','state','prompt','timestamp'}
            assert obs['front_rgb'].shape==(480,640,3)
            goals=np.tile(obs['state'],(10,1));goals[:,0]+=.12
            return {'actions':goals}
    camera=RGBDCamera(sim)
    try:
        record=query_and_execute(sim,arm,gripper,camera,MockPolicy(),'Move slightly.',execute_actions=2)
        assert record['predicted_actions']==10 and record['executed_actions']==2
        assert .39<sim.data.time<.41
        assert record['measured_arm_after'][0]-record['measured_state_before'][0]>.04
    finally:camera.close()
