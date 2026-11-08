"""Runtime-matched schema3 must retain goals and reproduce physical states."""
import json
from pathlib import Path
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.vla.interface import ActionChunkExecutor
from robot_exam.data.episodes import EpisodeRecorder,validate_episode


def test_matched_capture_preserves_goal_semantics_and_runtime_replay(tmp_path):
    sim=Simulation(Path(__file__).resolve().parents[1]/'scenes/mobile_manipulator.xml')
    arm=ArmController(sim);gripper=GripperController(sim);executor=ActionChunkExecutor(sim,arm,gripper)
    episode=tmp_path/'episode';recorder=EpisodeRecorder(sim,episode,'Interface motion only.',1,'oracle',initial_arm_reference=arm.reference)
    recorder.meta.update(action_execution_contract='full_horizon_interpolate_replan5hz_v1',
                         execution_max_speed_rad_s=.7,execution_max_acceleration_rad_s2=1.4)
    original=sim.step
    def record(ctrl,count=1):
        recorder.capture_pre_step('interface_only',arm_target=executor.trace[-1]['goal'][:6],arm_reference=arm.reference)
        return original(ctrl,count)
    sim.step=record
    actions=np.tile(np.r_[arm.reference,0.],(10,1));actions[:,0]+=.03
    for _ in range(3):executor.execute_prefix(actions,sim.data.time)
    recorder.close(False)  # This is NOT a grasp demonstration or VLA success.
    result=validate_episode(episode,replay=True)
    assert result['status']=='PASS' and result['max_replay_qpos_error']<1e-10
    assert result['success_label'] is False
    first=json.loads((episode/'frames.jsonl').open().readline())
    assert not np.allclose(first['action'][:6],first['action.reference'])
    meta=json.loads((episode/'episode.json').read_text());meta['action_execution_contract']='unrecognized_controller'
    (episode/'episode.json').write_text(json.dumps(meta))
    rejected=validate_episode(episode,replay=False)
    assert rejected['status']=='FAIL' and 'unknown action execution contract' in rejected['errors']
