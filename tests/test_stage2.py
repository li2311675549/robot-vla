"""Stage-two physical acceptance; no playback/teleport shortcut."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from robot_exam.simulation import ROOT, Simulation
from robot_exam.control.arm import ArmController
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.skills.pick_place import PickPlace

pytestmark = pytest.mark.stage2
sys.path.insert(0, str(ROOT / "scripts"))
from validate_base import run_case
from demo_mobile_manipulator import run as run_mobile


@pytest.mark.parametrize("case", ["straight", "reverse", "turn", "curve"])
def test_husky_wheel_motion_and_stopping(case):
    result = run_case(case)
    assert result["status"] == "PASS", result
    assert not any(result["warnings"])


def test_original_husky_dimensions_mass_and_free_base():
    sim = Simulation(ROOT / "scenes/husky.xml")
    assert sim.model.nq == 11 and sim.model.nv == 10 and sim.model.nu == 4
    assert sim.model.body("husky").mass[0] == pytest.approx(46.034)
    assert sim.model.body("husky").ipos[1] == pytest.approx(-.085)
    for axle, sx in (("front", 1), ("rear", -1)):
        for side, sy in (("left", 1), ("right", -1)):
            wheel = sim.model.body(f"{axle}_{side}_wheel_link")
            assert wheel.mass[0] == pytest.approx(2.637)
            assert wheel.pos == pytest.approx([sx*.256, sy*.2775, .03282])
    sim.step([0, 0, 0, 0], 1500)
    assert np.linalg.norm(sim.data.qpos[:2]) < .001


def original_dh(q):
    # Independent original-UR5 DH table (metres), expressed in ROS base_link.
    transform = np.diag([-1., -1., 1., 1.])
    transform[2, 3] = .395
    for theta, a, d, alpha in zip(q, [0, -.425, -.39225, 0, 0, 0],
                                 [.089159, 0, 0, .10915, .09465, .0823],
                                 [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]):
        c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        transform = transform @ np.array([[c, -s*ca, s*sa, a*c], [s, c*ca, -c*sa, a*s],
                                          [0, sa, ca, d], [0, 0, 0, 1]])
    return transform


@pytest.mark.parametrize("q", [[0, -1.9, 1.57, -1.24, -1.57, 0],
                               [.3, -1.4, 1.0, -1.1, -1.5, .4],
                               [-.2, -1.6, 1.3, -.9, -1.7, -.2]])
def test_ur5_fk_and_gripper_flange_transform(q):
    sim = Simulation(ROOT / "scenes/ur5_gripper.xml")
    scratch = mujoco.MjData(sim.model)
    scratch.qpos[:] = sim.data.qpos
    scratch.qpos[:6] = q
    mujoco.mj_forward(sim.model, scratch)
    expected = original_dh(q)
    tool = sim.model.body("tool0").id
    assert np.allclose(scratch.xpos[tool], expected[:3, 3], atol=1e-6)
    assert np.allclose(scratch.xmat[tool].reshape(3, 3), expected[:3, :3], atol=1e-6)
    pinch = sim.model.site("grip_pinch").id
    expected_pinch = expected[:3, 3] + expected[:3, :3] @ [0, 0, .1558]
    assert np.allclose(scratch.site_xpos[pinch], expected_pinch, atol=1e-6)
    rz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(scratch.site_xmat[pinch].reshape(3, 3), expected[:3, :3] @ rz, atol=1e-6)


@pytest.mark.parametrize("position", [[.55, 0, .75], [.55, 0, .83], [.55, .22, .83]])
def test_ik_solution_without_modifying_live_simulation(position):
    sim = Simulation(ROOT / "scenes/ur5_gripper.xml")
    arm = ArmController(sim)
    original_qpos = sim.data.qpos.copy()
    q, metrics = arm.solve(position, np.diag([1, -1, -1]))
    assert np.array_equal(sim.data.qpos, original_qpos)
    assert sim.data.time == 0
    assert metrics["position_error_m"] < .0005
    assert metrics["orientation_error_rad"] < .003
    assert np.all(q >= sim.model.jnt_range[arm.joints, 0])
    assert np.all(q <= sim.model.jnt_range[arm.joints, 1])


def test_unreachable_ik_rejects_target():
    sim = Simulation(ROOT / "scenes/ur5_gripper.xml")
    with pytest.raises(RuntimeError, match="IK failed"):
        ArmController(sim).solve([5, 0, 2], np.eye(3))


def test_no_object_weld_mocap_or_actuated_free_base():
    sim = Simulation(ROOT / "scenes/mobile_manipulator.xml")
    assert sim.model.nmocap == 0
    assert not np.any(sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    assert sim.model.joint("object_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert sim.model.joint("husky_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert np.all(sim.data.qfrc_applied == 0)
    assert np.all(sim.data.xfrc_applied == 0)
    assert sim.model.body("camera_arch").mass[0] == pytest.approx(.55)
    assert sim.model.body("d435i_virtual").mass[0] == pytest.approx(.072)


@pytest.mark.parametrize("variant,halfheight", [("box", .025), ("small", .020), ("wide", .030), ("cylinder", .025)])
def test_physical_grasp_hold_transfer_release(variant, halfheight):
    suffix = "" if variant == "box" else "_" + variant
    sim = Simulation(ROOT / f"scenes/ur5_gripper{suffix}.xml")
    skill = PickPlace(sim)
    report = skill.run([.55, 0, .725+halfheight], [.55, .22, .725+halfheight])
    assert report["status"] == "PASS", report
    assert report["bilateral_contact_fraction_during_hold"] >= .95
    assert report["min_hold_height_m"] > .05
    assert report["placement_error_m"] < .01
    assert np.all(sim.data.qfrc_applied == 0) and np.all(sim.data.xfrc_applied == 0)


def test_off_axis_metric_backprojection_on_known_plane():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="20 20 .1"/>
      <camera name="d435i_rgbd" pos="-.25 -.25 1.08428" xyaxes="0 -1 0 .385 0 .923" fovy="58"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = RGBDCamera(SimpleNamespace(model=model, data=data))
    try:
        rgb, depth = camera.capture()
        uv = np.array([[u, v] for u in (160, 320, 480) for v in (280, 360, 420)])
        points = camera.unproject(uv, depth[uv[:, 1], uv[:, 0]])
        assert np.max(np.abs(points[:, 2])) < .001
        # The nine rays include both horizontal and vertical offsets.
        assert np.ptp(points[:, 0]) > .3 and np.ptp(points[:, 1]) > .3
    finally:
        camera.close()


def test_pixel_center_depth_preserves_rgb_and_physical_state(monkeypatch):
    sim = Simulation(ROOT / "scenes/ur5_gripper.xml")
    sim.step(sim.data.ctrl, 300)
    camera = RGBDCamera(sim)
    qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
    samples = int(sim.model.vis.quality.offsamples)
    rgb_frames = []
    render = camera.renderer.render
    def remember_actual_rgb(*args, **kwargs):
        frame = render(*args, **kwargs)
        if frame.ndim == 3:
            rgb_frames.append(frame.copy())
        return frame
    monkeypatch.setattr(camera.renderer, "render", remember_actual_rgb)
    try:
        rgb, depth = camera.capture()
        assert rgb.std() > 8 and len(rgb_frames) == 1
        # Compare the SAME sampled RGB buffer, not independently rasterized
        # frames: native GPU MSAA may differ by1 LSB across separate renders.
        np.testing.assert_array_equal(rgb_frames[0], rgb)
        np.testing.assert_array_equal(qpos, sim.data.qpos)
        np.testing.assert_array_equal(qvel, sim.data.qvel)
        assert int(sim.model.vis.quality.offsamples) == samples
        assert np.isfinite(depth).all() and depth.min() > 0
    finally:
        camera.close()


def test_rgbd_fixture_localization_from_pixels():
    sim = Simulation(ROOT / "scenes/ur5_gripper.xml")
    sim.step(sim.data.ctrl, 300)
    camera = RGBDCamera(sim)
    try:
        rgb, depth = camera.capture()
        estimate = camera.detect_red_fixture(rgb, depth)
        truth = sim.data.xpos[sim.model.body("object").id]
        assert np.linalg.norm(np.asarray(estimate["center"])-truth) < .01
        with pytest.raises(RuntimeError, match="not visible"):
            camera.detect_red_fixture(np.zeros_like(rgb), depth)
    finally:
        camera.close()


def test_complete_mobile_manipulator_camera_guided_pick_place(tmp_path):
    result = run_mobile(tmp_path)
    assert result["status"] == "PASS", result
    assert result["parked"][0] - result["drive_start"][0] > .9
    assert result["parking_position_error_m"] < .03
    assert result["rgbd_localization_error_m"] < .01
    assert result["base_drift_during_manipulation_m"] < .01
    assert result["max_base_tilt_rad"] < .03
    assert result["grasp"]["placement_error_m"] < .01
