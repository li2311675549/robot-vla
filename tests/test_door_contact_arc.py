import math
import sys

import mujoco
import numpy as np
import pytest

from robot_exam.simulation import ROOT, Simulation
from robot_exam.tasks.door import handle_position
from robot_exam.tasks.door_contact import arc_station, arc_rotation, unsafe_contact, select_start_arc
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.navigation.local_dock import plan_dock

sys.path.insert(0,str(ROOT/'scripts'))
from build_tasks import build


@pytest.mark.parametrize('inner',[False,True])
def test_contact_arc_is_wheel_executable_and_arm_relative_pose_is_constant(inner):
    metadata={'hinge':[-.09,-.8,0.],'handle_r':1.4,'inner_handle_radius':1.4,
              'handle_height':.95,'inner_handle_height':.95}
    relative=[]
    for angle in np.linspace(0.,1.56,13):
        pose=arc_station(metadata,angle,inner)
        derivative=(arc_station(metadata,angle+1e-6,inner)[:2]-arc_station(metadata,angle-1e-6,inner)[:2])/2e-6
        lateral=np.array([-math.sin(pose[2]),math.cos(pose[2])])
        assert abs(derivative@lateral)<1e-7  # No impossible sideways chassis motion.
        c,s=math.cos(pose[2]),math.sin(pose[2]);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        relative.append((rotation.T@(handle_position(metadata,angle,inner)-np.r_[pose[:2],0.]),
                         rotation.T@arc_rotation(metadata,angle,inner)))
    for position,rotation in relative:
        np.testing.assert_allclose(position,relative[0][0],atol=1e-12)
        np.testing.assert_allclose(rotation,relative[0][1],atol=1e-12)


def test_safety_monitor_detects_nonhand_leaf_contact_in_initial_fixture(tmp_path):
    scene,_=build('door',2101,tmp_path/'door.xml',door_profile='aligned_handles_v3')
    sim=Simulation(scene);r=TaskRuntime(sim)
    assert unsafe_contact(r) is None
    # Independent time-zero negative fixture; no execution or success claim.
    sim.data.qpos[:2]=[-.09,0.]
    mujoco.mj_forward(sim.model,sim.data)
    result=unsafe_contact(r)
    assert result is not None and result['depth_m']>.001
    assert 'husky' in result['bodies'] or 'door_leaf' in result['bodies']


def test_short_recovery_does_not_require_unnecessary_pregoal_space():
    valid=lambda p: -.01<=p[0]<=.10 and abs(p[1])<.03
    with pytest.raises(RuntimeError,match='preapproach'):
        plan_dock([0,0,0],[.08,0,0],strategy='preapproach',valid=valid)
    plan=plan_dock([0,0,0],[.08,0,0],strategy='direct',valid=valid)
    np.testing.assert_allclose(plan['poses'][-1],[.08,0,0])


def test_direct_recovery_still_rejects_a_blocked_connection():
    with pytest.raises(RuntimeError,match='direct'):
        plan_dock([0,0,0],[.08,0,0],strategy='direct',valid=lambda p: not .025<p[0]<.055)


def test_measured_operating_circle_is_validated_without_moving_live_state(tmp_path):
    scene,metadata=build('door',2101,tmp_path/'door.xml',door_profile='aligned_handles_v3')
    sim=Simulation(scene);angle=.965;normal=.685;tangent=.924
    c,s=math.cos(angle),math.sin(angle);yaw=angle+math.atan2(normal,-tangent)+.05
    sim.data.qpos[:2]=np.asarray(metadata['hinge'])[:2]+np.array([[c,-s],[s,c]])@[normal,tangent]
    sim.data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint('door_hinge').id]]=angle
    mujoco.mj_forward(sim.model,sim.data);r=TaskRuntime(sim)
    before=sim.data.qpos.copy();ctrl=sim.data.ctrl.copy()
    result=select_start_arc(r,metadata,True,0.)
    assert result['accepted'] and result['collision_risk'] is None
    np.testing.assert_allclose(metadata['contact_arc_parameters']['close'],[normal,tangent],atol=1e-12)
    np.testing.assert_array_equal(before,sim.data.qpos)
    np.testing.assert_array_equal(ctrl,sim.data.ctrl)
    assert sim.data.time==0.


def test_operating_circle_rejects_an_unsafe_side_instead_of_widening_tolerance(tmp_path):
    scene,metadata=build('door',2101,tmp_path/'door.xml',door_profile='aligned_handles_v3')
    sim=Simulation(scene);r=TaskRuntime(sim)
    result=select_start_arc(r,metadata,True,0.)  # Outside start cannot be an inside closing path.
    assert not result['accepted'] and 'contact_arc_parameters' not in metadata
