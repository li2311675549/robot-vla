"""Actual MJCF inventory and safety checks independent of sparse output logs."""
import sys
import gzip
import json

import mujoco
import numpy as np
import pytest

from robot_exam.simulation import ROOT, Simulation
from robot_exam.tasks.runtime import TaskFailure, TaskRuntime
from robot_exam.control.parking import WheelParkingHold, ParkingViolation

sys.path.insert(0,str(ROOT/'scripts'))
from build_tasks import build


@pytest.mark.parametrize('task',('door','fruit','drawer'))
def test_full_inventory_is_not_a_single_fruit_fixture(task,tmp_path):
    path,meta=build(task,200002,tmp_path/'scene.xml',book_profile='distinct_v2',door_profile='aligned_handles_v3',wrist_camera=True)
    sim=Simulation(path);m=sim.model
    if task=='fruit':
        assert len(meta['objects'])==10
        assert {o['category'] for o in meta['objects']}=={'apple','banana','orange'}
        assert all(m.joint(o['body']+'_free').type==mujoco.mjtJoint.mjJNT_FREE for o in meta['objects'])
        assert all(m.geom('bin_label_'+s).matid>=0 for s in ('apple','banana','orange'))
    elif task=='drawer':
        assert len(meta['objects'])==3 and len({o['area'] for o in meta['objects']})==3
        assert all(m.joint('drawer_slide_'+str(i)).id not in m.actuator_trnid[:,0] for i in range(3))
    else:
        assert np.linalg.norm(np.array(meta['start'][:2])-meta['start_distance_reference'])==pytest.approx(5)
        assert m.joint('door_hinge').id not in m.actuator_trnid[:,0]
        assert abs(sim.data.qpos[m.jnt_qposadr[m.joint('door_hinge').id]])<1e-9


@pytest.mark.parametrize('period',(.02,10.))
def test_contact_loss_monitor_keeps_physics_cadence_when_trace_is_sparse(tmp_path,period):
    path,_=build('fruit',200003,tmp_path/'scene.xml')
    sim=Simulation(path);runtime=TaskRuntime(sim)
    runtime.config={'payload_trace_interval_seconds':period}
    runtime.begin_payload('fruit_0',runtime.arm.pose()[1],world_stabilize=False)
    assert runtime.payload_rotation is None
    with pytest.raises(TaskFailure,match='In-motion grasp witness lost') as caught:
        runtime.hold(.10)
    assert caught.value.category=='gripper_slip'
    assert sim.data.time==pytest.approx(.06)
    assert len(runtime.payload_trace)==(3 if period==.02 else 1)


def test_sparse_trajectory_does_not_reduce_collision_inspection(tmp_path,monkeypatch):
    path,_=build('fruit',200004,tmp_path/'scene.xml')
    sim=Simulation(path);runtime=TaskRuntime(sim)
    runtime.config={'trajectory_log_interval_seconds':10.}
    calls=[];check=runtime.check_collisions
    def counted():calls.append(sim.data.time);check()
    monkeypatch.setattr(runtime,'check_collisions',counted)
    runtime.hold(.10)
    assert len(runtime.rows)==1 and len(calls)==2


@pytest.mark.parametrize('sample_seconds',(.05,10.))
def test_parking_guard_and_two_second_gate_are_independent_of_recording(tmp_path,sample_seconds):
    path,_=build('fruit',200004,tmp_path/'scene.xml')
    sim=Simulation(path);sim.step(sim.data.ctrl,500)
    parking=WheelParkingHold(sim,sample_seconds=sample_seconds);parking.begin()
    sim.step(sim.data.ctrl,1050)
    assert parking.ready and parking.stable_seconds>=2.
    # Reference-only disturbance: no robot/object state assignment.
    parking.anchor[0]+=.04
    with pytest.raises(ParkingViolation):sim.step(sim.data.ctrl,70)
    assert parking.faulted
    assert len(parking.samples)==1 if sample_seconds==10. else len(parking.samples)>20


def test_upright_label_version_preserves_all_task_physics(tmp_path):
    old,_=build('fruit',200005,tmp_path/'old.xml')
    new,meta=build('fruit',200005,tmp_path/'new.xml',bin_label_profile='upright_v2')
    a,b=mujoco.MjModel.from_xml_path(str(old)),mujoco.MjModel.from_xml_path(str(new))
    for field in ('qpos0','body_mass','body_inertia','geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity','actuator_forcerange'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    assert not np.array_equal(a.tex_data,b.tex_data)
    assert meta['scene_protocol']=='fruit_seeded_upright_labels_v2'


def test_new_bulk_logs_compress_losslessly_and_leave_results_small(tmp_path):
    path,_=build('fruit',200006,tmp_path/'scene.xml')
    runtime=TaskRuntime(Simulation(path),output=tmp_path/'run')
    runtime.config={'compress_bulk_logs':True};runtime.hold(.1);runtime.save()
    with gzip.open(tmp_path/'run/trajectory.json.gz','rt') as stream:
        assert json.load(stream)==runtime.rows
    assert not (tmp_path/'run/trajectory.json').exists()
    assert (tmp_path/'run/fsm.json').exists() and (tmp_path/'run/collisions.json').exists()


def test_contact_feedback_does_not_override_release_command(tmp_path):
    from robot_exam.control.contact_grip import update_contact_feedback
    path,_=build('fruit',200007,tmp_path/'scene.xml');runtime=TaskRuntime(Simulation(path))
    runtime.grip_force_feedback={'body':'fruit_0','target_force_n':12.}
    runtime.gripper.command(0);update_contact_feedback(runtime)
    assert runtime.grip_force_feedback is None and runtime.sim.data.ctrl[runtime.gripper.actuator]==0


def test_stopped_grip_check_cannot_accept_a_nonheld_object(tmp_path,monkeypatch):
    from robot_exam.control.grasp_recovery import stop_and_revalidate_grasp
    path,meta=build('fruit',200008,tmp_path/'scene.xml');runtime=TaskRuntime(Simulation(path))
    def no_open(*args):raise AssertionError('Stopped verification must not open/close the gripper')
    monkeypatch.setattr(runtime.gripper,'command',no_open)
    report=stop_and_revalidate_grasp(runtime,'fruit_0',meta['objects'][0]['initial'][2]+.055)
    assert not report['success'] and report['bilateral_fraction']==0
    assert report['max_gripper_distance_m']>.075


def test_touching_round_instances_have_separate_centres_without_lost_pixels():
    from robot_exam.perception.objects import split_round_component
    yy,xx=np.indices((65,100))
    mask=((yy-30)**2+(xx-30)**2<=18**2)|((yy-32)**2+(xx-61)**2<=16**2)
    pixels=np.argwhere(mask);groups=split_round_component(pixels)
    assert len(groups)==2
    assert sorted(map(tuple,np.vstack(groups)))==sorted(map(tuple,pixels))
    centres=sorted([g.mean(axis=0) for g in groups],key=lambda x:x[1])
    assert np.linalg.norm(centres[0]-[30,30])<3
    assert np.linalg.norm(centres[1]-[32,61])<3


def test_single_round_instance_is_not_split():
    from robot_exam.perception.objects import split_round_component
    yy,xx=np.indices((60,60));pixels=np.argwhere((yy-30)**2+(xx-30)**2<=18**2)
    groups=split_round_component(pixels)
    assert len(groups)==1
    np.testing.assert_array_equal(groups[0],pixels)


def test_book_support_profile_changes_only_documented_contact_time_constant(tmp_path):
    old,_=build('drawer',200004,tmp_path/'old.xml',book_profile='distinct_v2')
    new,metadata=build('drawer',200004,tmp_path/'new.xml',book_profile='distinct_v2',book_contact_profile='wood_support_v2')
    a,b=mujoco.MjModel.from_xml_path(str(old)),mujoco.MjModel.from_xml_path(str(new))
    for field in ('qpos0','body_mass','body_inertia','geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity','actuator_gainprm','actuator_forcerange'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    expected={a.geom(f'{prefix}_{rank}_{suffix}').id for rank in range(3) for prefix,suffix in (('book','cover'),('drawer','bottom'))}
    changed=set(np.flatnonzero(np.any(a.geom_solref!=b.geom_solref,axis=1)))
    assert changed==expected
    for geom in changed:np.testing.assert_array_equal(b.geom_solref[geom],[.004,1.])
    assert metadata['book_contact_profile']=='wood_support_v2'


@pytest.mark.parametrize('task',('fruit','drawer'))
@pytest.mark.parametrize('seed',(200006,200007,200008))
def test_scattered_scene_preserves_sizes_and_has_no_initial_target_penetration(tmp_path,task,seed):
    path,metadata=build(task,seed,tmp_path/'scatter.xml',book_profile='distinct_v2',layout_profile='bounded_scatter_v3')
    sim=Simulation(path);sim.step(sim.data.ctrl,350)
    objects=metadata['objects'];names={o['body'] for o in objects}
    assert len(objects)==(10 if task=='fruit' else 3)
    if task=='drawer':
        assert sorted(o['size'] for o in objects)==[[.12,.08,.015],[.18,.12,.02],[.24,.16,.025]]
    else:
        assert {o['category'] for o in objects}=={'apple','banana','orange'}
        assert all(.88<=o['scale']<=1.08 for o in objects)
    table=sim.model.geom('fruit_table_top' if task=='fruit' else 'book_table_top').id
    front=sim.data.geom_xpos[table,0]-sim.model.geom_size[table,0]
    assert front-metadata['start'][0]==pytest.approx(5)
    for contact in sim.data.contact:
        bodies={sim.model.body(int(sim.model.geom_bodyid[g])).name for g in contact.geom}
        if bodies&names:assert contact.dist>-.003
    _,again=build(task,seed,tmp_path/'repeat.xml',book_profile='distinct_v2',layout_profile='bounded_scatter_v3')
    assert objects==again['objects']
    assert not sim.data.warning.number.any()


@pytest.mark.parametrize('rank',(0,1,2))
def test_drawer_support_arrests_a_book_dropped_from_release_height(tmp_path,rank):
    from robot_exam.tasks.scoring import body_bounds
    path,meta=build('drawer',200004,tmp_path/'support.xml',book_profile='distinct_v2',book_contact_profile='wood_support_v2')
    sim=Simulation(path);m,d=sim.model,sim.data;name=f'book_{rank}'
    obj=next(o for o in meta['objects'] if o['rank']==rank)
    adr=int(m.jnt_qposadr[m.body_jntadr[m.body(name).id]])
    # Independent passive impact initialization, never a task-action shortcut.
    d.qpos[m.jnt_qposadr[m.joint(f'drawer_slide_{rank}').id]]=.27
    d.qpos[adr:adr+3]=[.45,.5,meta['drawer_bottoms'][rank]+.008+obj['size'][2]/2+.23]
    d.qpos[adr+3:adr+7]=[np.sqrt(.5),0,0,np.sqrt(.5)]
    mujoco.mj_forward(m,d);maximum_penetration=0.
    for _ in range(700):
        sim.step(d.ctrl)
        for contact in d.contact:
            names={m.geom(int(g)).name for g in contact.geom}
            if {f'drawer_{rank}_bottom',name+'_cover'}<=names:
                maximum_penetration=max(maximum_penetration,-float(contact.dist))
    lower,upper=body_bounds(sim,name)
    assert maximum_penetration<.003
    assert lower[2]>meta['drawer_bottoms'][rank]-.003
    assert upper[2]<meta['drawer_bottoms'][rank]+.14
    assert not d.warning.number.any()


@pytest.mark.parametrize('strict',(False,True))
def test_payload_pivot_does_not_walk_toward_a_lateral_anchor_when_enabled(tmp_path,strict):
    path,_=build('fruit',200009,tmp_path/'scene.xml')
    r=TaskRuntime(Simulation(path));r.config={'payload_strict_pivot_turn':strict}
    # Command-shaping test only: the table fruit is not a physical payload.
    # One physics step precedes the existing60ms contact-loss guard.
    r.payload_motion_profile='orange'
    r.begin_payload('fruit_0',r.arm.pose()[1],world_stabilize=False)
    pose=r.base.pose();r.base.rotate(pose[:2]+[.05,0],pose[2]+.7);r.tick()
    wheels=r.sim.data.ctrl[r.base.ids]
    common_mode=(wheels[0]+wheels[1])/2
    if strict:assert common_mode==pytest.approx(0.,abs=1e-12)
    else:assert common_mode>0


@pytest.mark.parametrize('yaw',(-np.pi/2,np.pi/2,-1.94,2.,0.,np.pi))
def test_rectangular_alignment_does_not_request_an_unnecessary_half_turn(yaw):
    from robot_exam.tasks.drawer import rectangular_axis_alignment
    angle=rectangular_axis_alignment(yaw)
    assert abs(angle)<=np.pi/2+1e-12
    assert abs(np.cos(yaw+angle))<1e-12
    if abs(abs(yaw)-np.pi/2)<1e-12:assert abs(angle)<1e-12
