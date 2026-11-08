#!/usr/bin/env python3
"""Check documented geometry against real functions; optional short model smoke.

This does not evaluate a grasp or complete task, and requires no model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT
import numpy as np

from robot_exam.control.grasp_monitor import relative_grasp_pose
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.skills.book_edge import book_centerline_halfwidth, long_edge_grasp_frame


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-smoke', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; preserve previous checks')
    output.mkdir(parents=True)

    transform = np.eye(4)
    transform[:3, :3] = rotation_z(np.pi / 2)
    transform[:3, 3] = [1., 0., 0.]
    from types import SimpleNamespace
    camera = SimpleNamespace(
        K=np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
        world_from_optical=lambda: transform)
    point = RGBDCamera.unproject(camera, np.array([[420., 240.]]), np.array([.6]))
    assert np.allclose(point, [[1., .12, .6]])

    size = np.array([.24, .16, .025])
    center = np.array([.332, -.65, .7385])
    rotation = rotation_z(np.pi / 2)
    halfwidth = book_centerline_halfwidth(rotation, size)
    approach, _, pinch_rotation = long_edge_grasp_frame(rotation, size)
    grasp = center - approach * (size[1] / 2 - .015)
    assert np.isclose(halfwidth, .08) and np.isclose(center[0] - halfwidth, .252)
    assert np.isclose(grasp[0], .267) and np.isclose(.28 - grasp[0], .013)
    assert np.allclose(pinch_rotation[:, 1], [0., 0., 1.])
    assert np.allclose(pinch_rotation[:, 2], approach)
    diagonal = rotation_z(np.pi / 4)
    assert book_centerline_halfwidth(diagonal, size) < np.abs(diagonal[0, :2]) @ (size[:2] / 2)

    relative = np.array([.1, .02, .01])
    moved_rotation = rotation_z(.5)
    moved_tool = np.array([.5, .15, .9])
    before = relative_grasp_pose(np.zeros(3), np.eye(3), relative, np.eye(3))
    after = relative_grasp_pose(moved_tool, moved_rotation,
                                moved_tool + moved_rotation @ relative, moved_rotation)
    assert np.allclose(before[0], after[0]) and np.allclose(before[1], after[1])

    scenes = []
    if args.model_smoke:
        import mujoco
        from build_tasks import build
        from robot_exam.control.arm import ArmController
        from robot_exam.simulation import Simulation
        for task in ('fruit', 'door', 'drawer'):
            path, _ = build(task, 200009, output / task / 'scene.xml',
                book_profile='distinct_v2' if task == 'drawer' else 'booklet',
                door_profile='aligned_handles_v3', wrist_camera=True,
                layout_profile='bounded_scatter_v3' if task == 'drawer' else 'legacy_v1',
                book_contact_profile='wood_support_v2' if task == 'drawer' else 'legacy_v1')
            sim = Simulation(path)
            arm = ArmController(sim)
            assert sim.model.nu == 11 and sim.model.nmocap == 0
            assert not np.any(sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
            passive = [i for i in range(sim.model.njnt)
                if sim.model.joint(i).name == 'door_hinge'
                or (sim.model.joint(i).name or '').startswith('drawer_slide_')]
            assert not any(j in sim.model.actuator_trnid[:, 0] for j in passive)
            for _ in range(50):
                arm.command(sim.data.qpos[arm.qadr].copy())
                sim.step(sim.data.ctrl)
            scenes.append(dict(task=task, physical_steps=50, sim_seconds=float(sim.data.time),
                               warnings=sim.data.warning.number.tolist()))

    sources = ['perception/rgbd.py', 'skills/book_edge.py', 'control/grasp_monitor.py']
    report = dict(status='PASS_DOCUMENTED_NUMERICAL_EXAMPLES', numerical_checks=5,
        model_smokes=scenes, task_success_measured=False, new_training=False,
        scope='Synthetic arithmetic and initial 50-step physics only, not contact-task success',
        source_sha256={name: hashlib.sha256((ROOT/'src/robot_exam'/name).read_bytes()).hexdigest()
                       for name in sources})
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
