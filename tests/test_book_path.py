import sys

import mujoco
import numpy as np

from robot_exam.control.arm import ArmController
from robot_exam.control.book_path import held_book_planning_arm
from robot_exam.simulation import ROOT, Simulation


def test_held_book_collision_predictor_keeps_live_integration_state_untouched(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from build_tasks import build
    path, _ = build('drawer', 200011, tmp_path / 'scene.xml', book_profile='distinct_v2')
    sim = Simulation(path)
    arm = ArmController(sim)
    before = np.empty(mujoco.mj_stateSize(sim.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(sim.model, sim.data, before, mujoco.mjtState.mjSTATE_INTEGRATION)
    p, rotation = arm.pose()
    body = sim.model.body('book_2').id
    relative_p = rotation.T @ (sim.data.xpos[body] - p)
    relative_r = rotation.T @ sim.data.xmat[body].reshape(3, 3)
    q = sim.data.qpos[arm.qadr].copy()
    q[0] += .2
    # Only a geometric predictor: no claim that this distant initial book is
    # actually held. Live task checks require independent physical contact.
    first = held_book_planning_arm(arm, q, rotation, 'book_2')
    _, new_rotation = first.pose()
    planning = held_book_planning_arm(arm, q, new_rotation, 'book_2')
    tip, actual_r = planning.pose()
    np.testing.assert_allclose(actual_r.T @ (planning.sim.data.xpos[body] - tip), relative_p, atol=1e-10)
    np.testing.assert_allclose(actual_r.T @ planning.sim.data.xmat[body].reshape(3, 3), relative_r, atol=1e-10)
    after = np.empty_like(before)
    mujoco.mj_getState(sim.model, sim.data, after, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.testing.assert_array_equal(before, after)
