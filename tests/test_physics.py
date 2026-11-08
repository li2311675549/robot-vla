import json
import hashlib

import mujoco
import numpy as np
import pytest

from robot_exam.simulation import ROOT, Simulation, UR5_JOINTS


def test_inline_mjcf_gravity_and_contact():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002"/>
      <worldbody><geom type="plane" size="2 2 0.1"/>
      <body pos="0 0 0.5"><freejoint/><geom type="sphere" size="0.05" mass="1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=50)
    assert 0.44 < data.qpos[2] < 0.47  # 0.1s physical free fall
    mujoco.mj_step(model, data, nstep=950)
    assert data.qpos[2] == pytest.approx(0.05, abs=0.002)
    assert data.ncon > 0
    assert not data.warning.number.any()
    assert data.time == pytest.approx(2)


def test_minimal_target_tracking():
    sim = Simulation()
    sim.step([0.7], 1500)
    assert sim.joint_positions()["hinge"] == pytest.approx(0.7, abs=0.005)
    assert sim.data.time == pytest.approx(3)


@pytest.mark.parametrize("axis,joint", enumerate(UR5_JOINTS))
def test_original_ur5_each_joint_tracks(axis, joint):
    sim = Simulation("ur5")
    assert sim.model.nq == sim.model.nv == sim.model.nu == 6
    assert tuple(sim.joint_positions()) == UR5_JOINTS
    initial = sim.data.qpos.copy()
    target = sim.data.ctrl.copy()
    target[axis] += 0.2
    # Smooth command respects native joint velocity limits for this small test.
    for step in range(1500):
        ctrl = initial + min(1, (step + 1) / 500) * (target - initial)
        sim.step(ctrl)
        assert np.all(sim.data.qpos >= sim.model.jnt_range[:, 0] - 1e-5)
        assert np.all(sim.data.qpos <= sim.model.jnt_range[:, 1] + 1e-5)
        assert np.max(np.abs(sim.data.qvel)) < 3.2
    assert sim.joint_positions()[joint] - initial[axis] > 0.15
    assert np.max(np.abs(sim.data.qpos - target)) < 0.035
    assert np.all(np.abs(sim.data.actuator_force[:3]) <= 150 + 1e-8)
    assert np.all(np.abs(sim.data.actuator_force[3:]) <= 28 + 1e-8)


def test_ur5_geometry_and_inertia_are_original_model():
    sim = Simulation("ur5")
    assert sim.model.body("forearm_link").pos[0] == pytest.approx(-0.425)
    assert sim.model.body("wrist_1_link").pos[0] == pytest.approx(-0.39225)
    assert sim.model.body("shoulder_link").pos[2] == pytest.approx(0.089159)
    assert sim.model.body("upper_arm_link").mass[0] == pytest.approx(8.393)
    assert sim.model.body("wrist_3_link").mass[0] == pytest.approx(0.1879)
    # Load only; metadata checks complement physical tracking and provenance.
    assert sim.model.body("tool0").id >= 0


def test_robotiq_open_close_without_suspended_object():
    sim = Simulation("robotiq")
    assert sim.model.nq == sim.model.nv == 8
    assert sim.model.nu == 1 and sim.model.neq == 3
    sim.step([180], 1500)
    closed = sim.joint_positions()
    assert closed["left_driver_joint"] > 0.4
    assert abs(closed["left_driver_joint"] - closed["right_driver_joint"]) < 0.005
    sim.step([0], 1500)
    opened = sim.joint_positions()
    assert closed["left_driver_joint"] - opened["left_driver_joint"] > 0.35
    assert abs(opened["left_driver_joint"] - opened["right_driver_joint"]) < 0.005


@pytest.mark.parametrize("ctrl", [[float("nan")], [2], [], [0, 0]])
def test_invalid_command_rejected_before_step(ctrl):
    sim = Simulation()
    with pytest.raises(ValueError):
        sim.step(ctrl)
    assert sim.data.time == 0


def test_pinned_model_integrity():
    lock = json.loads((ROOT / "configs/model_sources.lock.json").read_text())
    for source in lock["sources"]:
        for record in source["files"]:
            data = (ROOT / record["path"]).read_bytes()
            digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            assert len(data) == record["bytes"]
            assert digest == record["git_blob"], record["path"]
