import json
import mujoco
import numpy as np
import pytest
from robot_exam.simulation import ROOT, Simulation
from robot_exam.data.episodes import EpisodeRecorder, validate_episode
from robot_exam.control.arm import ArmController,GripperController


def test_real_rgbd_action_alignment_and_exact_replay(tmp_path):
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml')
    recorder=EpisodeRecorder(sim,tmp_path,'fruit',17,'oracle')
    assert recorder.meta['opengl']['vendor'] and recorder.meta['opengl']['renderer']
    arm=ArmController(sim);gripper=GripperController(sim)
    for _ in range(120):
        arm.command(arm.reference);gripper.command(0)
        recorder.capture_pre_step('stationary_interface_smoke',arm_target=arm.reference)
        sim.step(sim.data.ctrl)
    recorder.close(False)
    result=validate_episode(tmp_path,replay=True)
    assert result['status']=='PASS' and result['max_replay_qpos_error']==0
    assert result['replay_api']=='normal ArmController/GripperController API'
    assert not result['success_label'] # valid data never implies task success
    assert result['frames']==2 and result['physics_steps']==120
    rows=[json.loads(x) for x in (tmp_path/'frames.jsonl').read_text().splitlines()]
    rows[0]['action'][-1]=1.5
    (tmp_path/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    rejected=validate_episode(tmp_path)
    assert rejected['status']=='FAIL' and 'gripper action outside [0,1]' in rejected['errors']
