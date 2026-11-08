"""A wall-time diagnostic must not advance physics or bypass strict defaults."""
import numpy as np
import pytest

from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController, GripperController
from robot_exam.vla import client


def test_strict_network_deadline_and_explicit_paused_diagnostic(monkeypatch):
    from pathlib import Path
    sim = Simulation(Path(__file__).resolve().parents[1]/'scenes/mobile_manipulator.xml')
    arm, grip = ArmController(sim), GripperController(sim)
    class Camera:
        def capture(self):
            return np.zeros((224,224,3),dtype=np.uint8),np.ones((224,224))
    class DelayedPolicy:
        def infer(self, obs):
            assert float(sim.data.time) == obs['timestamp']
            return {'actions':np.tile(np.r_[arm.reference,0.],(10,1))}
    start=float(sim.data.time);ctrl=sim.data.ctrl.copy()
    times=iter([0.,.6])
    monkeypatch.setattr(client.time,'monotonic',lambda:next(times))
    with pytest.raises(TimeoutError,match='no actions executed'):
        client.query_and_execute(sim,arm,grip,Camera(),DelayedPolicy(),'Hold.')
    assert sim.data.time == start
    np.testing.assert_array_equal(sim.data.ctrl,ctrl)
    times=iter([0.,.6])
    monkeypatch.setattr(client.time,'monotonic',lambda:next(times))
    row=client.query_and_execute(sim,arm,grip,Camera(),DelayedPolicy(),'Hold.',max_latency_seconds=2.)
    assert row['query_wall_seconds'] == .6
    assert row['sim_time_after']-start == pytest.approx(.2)
    assert np.isfinite(sim.data.qpos).all()
