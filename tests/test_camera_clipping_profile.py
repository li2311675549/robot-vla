import sys

import mujoco
import numpy as np
import pytest

from robot_exam.data.episodes import EpisodeRecorder
from robot_exam.simulation import ROOT, Simulation


def test_metric_near_plane_versions_observations_without_changing_dynamics(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from build_tasks import build
    path, _ = build('fruit', 2100004, tmp_path / 'scene.xml', wrist_camera=True)
    sim = Simulation(path)
    keys = ('body_mass', 'body_inertia', 'geom_size', 'geom_pos', 'geom_quat',
            'geom_friction', 'geom_solref', 'actuator_gainprm', 'cam_pos', 'cam_quat', 'cam_fovy')
    before = {k: getattr(sim.model, k).copy() for k in keys}
    qpos = sim.data.qpos.copy()
    original_near = sim.model.vis.map.znear * sim.model.stat.extent
    recorder = EpisodeRecorder(sim, tmp_path / 'episode', 'camera audit', 2100004, 'oracle', camera_near_m=.005)
    assert recorder.meta['camera_clipping']['near_m'] == pytest.approx(.005, abs=1e-8)
    assert recorder.meta['camera_clipping']['previous_near_m'] == pytest.approx(original_near)
    assert recorder.meta['actual_cameras'] == ['front', 'wrist']
    loaded = mujoco.MjModel.from_binary_path(str(tmp_path / 'episode/model.mjb'))
    assert loaded.vis.map.znear * loaded.stat.extent == pytest.approx(.005, abs=1e-8)
    for key in keys:
        np.testing.assert_array_equal(before[key], getattr(sim.model, key))
        np.testing.assert_array_equal(before[key], getattr(loaded, key))
    np.testing.assert_array_equal(qpos, sim.data.qpos)
    recorder.close(False)
