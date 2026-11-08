"""Physical model and contact-layout invariants; fixtures are not task rates."""
import math
import sys

import mujoco
import numpy as np
import pytest

from robot_exam.simulation import ROOT,Simulation
from robot_exam.tasks.door import handle_position
from robot_exam.tasks.door_latch import closing_station

sys.path.insert(0,str(ROOT/'scripts'))
from build_tasks import build


def test_paired_latch_handles_are_coaxial_on_the_opposite_edge(tmp_path):
    scene,meta=build('door',1901,tmp_path/'door.xml',door_profile='aligned_handles_v3')
    sim=Simulation(scene);model=sim.model
    inside=model.geom('inner_handle').id;outside=model.geom('outer_handle').id
    np.testing.assert_allclose(model.geom_pos[inside,1:],model.geom_pos[outside,1:])
    assert model.geom_pos[inside,0]==-model.geom_pos[outside,0]
    assert meta['inner_handle_radius']==meta['handle_r']==1.40
    assert meta['handle_r']>0.85*meta['door_width']
    assert meta['inner_handle_height']==meta['handle_height']
    assert meta['door_total_mass_kg']==pytest.approx(18.26)
    for side in ('inner','outer'):
        z=model.geom_pos[model.geom(side+'_handle').id,2]
        mounts=[model.geom(side+'_handle_mount_'+s).id for s in ('lower','upper')]
        assert all(abs(model.geom_pos[g,2]-z)>.05 for g in mounts)
    hinge=model.joint('door_hinge').id
    assert not np.any(model.actuator_trnid[:,0]==hinge)
    assert model.nmocap==0 and not np.any(model.eq_type==mujoco.mjtEq.mjEQ_WELD)
    assert math.isclose(meta['start_distance_m'],5.,abs_tol=1e-12)


def test_legacy_door_geometry_remains_a_separate_protocol(tmp_path):
    _,old=build('door',1901,tmp_path/'old.xml',door_profile='multistart_v2')
    _,new=build('door',1901,tmp_path/'new.xml',door_profile='aligned_handles_v3')
    assert old['inner_handle_radius']==.22 and old['inner_handle_height']==.65
    assert new['scene_protocol']!=old['scene_protocol']
    assert new['start']==old['start']


@pytest.mark.parametrize('angle',[0.,.15,.4,.7,1.,1.3,1.56])
def test_closing_chassis_pose_is_clear_of_the_passive_leaf(tmp_path,angle):
    scene,meta=build('door',1901,tmp_path/'door.xml',door_profile='aligned_handles_v3')
    sim=Simulation(scene);pose=closing_station(meta,angle)
    # Isolated time-zero geometry, never a wheel-navigation success assertion.
    sim.data.qpos[:2]=pose[:2]
    sim.data.qpos[3:7]=[math.cos(pose[2]/2),0,0,math.sin(pose[2]/2)]
    sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint('door_hinge').id]]=angle
    mujoco.mj_forward(sim.model,sim.data)
    target=handle_position(meta,angle,True)
    np.testing.assert_allclose(target,sim.data.geom_xpos[sim.model.geom('inner_handle').id],atol=1e-10)
    for contact in sim.data.contact:
        names=[sim.model.body(int(sim.model.geom_bodyid[g])).name for g in contact.geom]
        if 'door_leaf' in names:
            assert 'husky' not in names or contact.dist>=-.001
            assert not any((n or '').endswith('_wheel_link') for n in names) or contact.dist>=-.001
