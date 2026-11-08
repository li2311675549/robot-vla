import sys

import mujoco

from robot_exam.data.scene_quality import fruit_scene_preservation
from robot_exam.simulation import ROOT,Simulation


def test_non_target_screen_uses_only_actual_fruit_and_detects_floor_initialization(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    path,metadata=build('fruit',2100005,tmp_path/'scene.xml',fruit_count=3,
                        layout_profile='bounded_scatter_v3')
    sim=Simulation(path)
    report=fruit_scene_preservation(sim,metadata,'fruit_0')
    assert report['success'] and len(report['non_targets'])==2
    assert all(x['body']!='fruit_table' for x in report['non_targets'])
    # Test-fixture initial state only; no task/control rollout or success claim.
    body=sim.model.body('fruit_1').id;adr=int(sim.model.jnt_qposadr[sim.model.body_jntadr[body]])
    sim.data.qpos[adr+2]=.04;mujoco.mj_forward(sim.model,sim.data)
    report=fruit_scene_preservation(sim,metadata,'fruit_0')
    assert not report['success']
    assert not next(x for x in report['non_targets'] if x['body']=='fruit_1')['on_original_table']
