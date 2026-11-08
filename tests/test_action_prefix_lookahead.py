"""Retaining lookahead must match teacher replay without executing its tail."""
from pathlib import Path
import numpy as np
import pytest
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.vla.interface import ActionChunkExecutor


def setup():
    sim=Simulation(Path(__file__).resolve().parents[1]/'scenes/mobile_manipulator.xml')
    arm=ArmController(sim);gripper=GripperController(sim)
    executor=ActionChunkExecutor(sim,arm,gripper)
    actions=np.tile(np.r_[arm.reference,0.],(10,1))
    actions[:,0]+=np.linspace(0,.06,10)
    return sim,executor,actions


def test_prefix_matches_full_buffer_teacher_interpolation_and_preserves_history():
    first,executor,actions=setup();second,manual,_=setup()
    for _ in range(3):
        executor.execute_prefix(actions,first.data.time,count=1)
        manual.submit(actions,second.data.time)
        for _ in range(100):manual.step()
        np.testing.assert_array_equal(first.data.qpos,second.data.qpos)
        np.testing.assert_array_equal(executor.velocity,manual.velocity)
        np.testing.assert_array_equal(executor.reference,manual.reference)
        assert executor.chunk is None
    assert first.data.time==pytest.approx(.6)


def test_prefix_keeps_lookahead_that_single_row_hold_discards():
    first,executor,actions=setup();second,legacy,_=setup()
    executor.execute_prefix(actions,first.data.time)
    legacy.execute(actions[:1],second.data.time)
    assert executor.reference[0]>legacy.reference[0]+1e-4
    assert first.data.time==pytest.approx(second.data.time)


def test_invalid_future_action_rejected_before_any_actuation():
    sim,executor,actions=setup();ctrl=sim.data.ctrl.copy();qpos=sim.data.qpos.copy()
    actions[-1,0]=executor.limits[0,1]+1
    with pytest.raises(ValueError,match='exceeds limits'):executor.execute_prefix(actions,sim.data.time)
    np.testing.assert_array_equal(ctrl,sim.data.ctrl);np.testing.assert_array_equal(qpos,sim.data.qpos)
    assert sim.data.time==0
