import imageio.v3 as iio
import mujoco
import numpy as np
import pytest

from robot_exam.simulation import Simulation, camera_for, targets


@pytest.mark.parametrize("scene", ["minimal", "ur5", "robotiq"])
def test_rgb_depth_and_visible_actuated_motion(scene, tmp_path):
    sim = Simulation(scene)
    camera = camera_for(scene)
    with mujoco.Renderer(sim.model, height=240, width=320) as renderer:
        renderer.update_scene(sim.data, camera=camera)
        initial = renderer.render().copy()
        sim.step(targets(sim, scene), 1500)
        renderer.update_scene(sim.data, camera=camera)
        rgb = renderer.render().copy()
        assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
        assert rgb.std() > 8
        assert np.abs(initial.astype(float) - rgb.astype(float)).mean() > 0.1
        output = tmp_path / "frame.png"
        iio.imwrite(output, rgb)
        assert np.array_equal(iio.imread(output), rgb)
        renderer.enable_depth_rendering()
        renderer.update_scene(sim.data, camera=camera)
        depth = renderer.render()
        assert depth.shape == (240, 320)
        assert np.isfinite(depth).all() and depth.min() > 0
        assert np.ptp(depth) > 0.01


def test_metric_depth_and_optical_axis_unprojection():
    # A known plane 2m directly ahead; verifies depth is meters rather than a
    # normalized GL buffer. MuJoCo camera looks along local -Z, image Y is down.
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="5 5 .1"/>
      <camera name="calibration" pos="0 0 2" fovy="60"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="calibration")
        depth = renderer.render()
    assert np.allclose(depth, 2, atol=0.002)
    rotation = data.cam_xmat[0].reshape(3, 3)
    point_world = data.cam_xpos[0] + rotation @ np.array([0, 0, -depth[60, 80]])
    assert np.linalg.norm(point_world) < 0.002
