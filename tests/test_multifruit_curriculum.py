import sys
import mujoco
import numpy as np
import pytest
from robot_exam.simulation import ROOT,Simulation
sys.path.insert(0,str(ROOT/'scripts'))
from build_tasks import build


@pytest.mark.parametrize('count',[3,5,10])
def test_curriculum_inventory_is_explicit_and_unchanged_during_physics(tmp_path,count):
    path,meta=build('fruit',2100003,tmp_path/'scene.xml',fruit_count=count,layout_profile='bounded_scatter_v3',wrist_camera=True)
    sim=Simulation(path)
    assert len(meta['objects'])==count and meta['object_count']==count
    assert {o['category'] for o in meta['objects']}=={'apple','banana','orange'}
    bodies=[sim.model.body(i).name for i in range(sim.model.nbody)]
    assert sum(n.startswith('fruit_') and n!='fruit_table' for n in bodies)==count
    if count!=10:assert 'NEVER' in meta['scope']
    sim.step(sim.data.ctrl,20)
    assert not sim.data.warning.number.any()


def test_default_scene_stays_exactly_ten_fruit(tmp_path):
    p,meta=build('fruit',2100001,tmp_path/'default.xml',layout_profile='bounded_scatter_v3',wrist_camera=True,bin_label_profile='upright_v2')
    q,_=build('fruit',2100001,tmp_path/'explicit.xml',fruit_count=10,layout_profile='bounded_scatter_v3',wrist_camera=True,bin_label_profile='upright_v2')
    a,b=mujoco.MjModel.from_xml_path(str(p)),mujoco.MjModel.from_xml_path(str(q))
    assert len(meta['objects'])==10
    for field in ['qpos0','body_mass','body_inertia','geom_size','geom_friction','geom_solref','actuator_gainprm','cam_pos']:
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))


def test_short_curriculum_cannot_be_executed_as_a_formal_fruit_task():
    from robot_exam.tasks.fruit import execute
    from robot_exam.tasks.runtime import TaskFailure
    with pytest.raises(TaskFailure,match='exactly10objects'):
        execute(None,{'objects':[{}, {}, {}]},mode='oracle')
