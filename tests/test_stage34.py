"""Planning, passive scenes, observation, bounded FSM and physical navigation."""
import json
import sys

import mujoco
import numpy as np
import pytest

from robot_exam.simulation import ROOT, Simulation
from robot_exam.navigation.grid import OccupancyGrid, NoPath
from robot_exam.tasks.runtime import TaskRuntime, State, TaskFailure
from robot_exam.tasks.contracts import describe
from robot_exam.perception.objects import observe_objects, grasp_candidates
from robot_exam.perception.rgbd import RGBDCamera

sys.path.insert(0, str(ROOT/'scripts'))
from build_tasks import build


@pytest.mark.parametrize('task', ['fruit', 'door', 'drawer'])
def test_seeded_passive_scene(task, tmp_path):
    path, meta = build(task, 42, tmp_path/(task+'.xml'))
    other, repeated = build(task, 42, tmp_path/(task+'_repeat.xml'))
    assert meta['objects'] == repeated['objects'] if 'objects' in meta else meta['handle_height'] == repeated['handle_height']
    sim = Simulation(path)
    assert sim.model.nu == 11 and sim.model.nmocap == 0
    assert not np.any(sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    actuated = sim.model.actuator_trnid[:, 0]
    for name in (['door_hinge'] if task == 'door' else [f'drawer_slide_{i}' for i in range(3)] if task == 'drawer' else []):
        assert sim.model.joint(name).id not in actuated
    sim.step(sim.data.ctrl, 300)
    assert np.isfinite(sim.data.qpos).all() and not sim.data.warning.number.any()
    assert abs(meta['start'][0] - ({'fruit':.36, 'drawer':.38, 'door':-.09}[task])) == pytest.approx(5)
    if task == 'fruit':
        assert len(meta['objects']) == 10
        assert {o['category'] for o in meta['objects']} == {'apple', 'banana', 'orange'}
        assert len({o['scale'] for o in meta['objects']}) == 10
        assert len(set(sim.model.geom_type)) >= 4


def test_astar_footprint_detour_and_smoothing():
    grid=OccupancyGrid(bounds=(-3,3,-3,3))
    grid.rectangle(-.3,.3,-.4,.4)
    path=grid.plan([-2,0],[2,0])
    assert len(path)>2
    assert all(grid.visible(grid.cell(a),grid.cell(b)) for a,b in zip(path[:-1],path[1:]))
    assert sum(np.linalg.norm(np.array(b)-a) for a,b in zip(path[:-1],path[1:])) > 4
    assert not grid.free(grid.cell([.8,0])) # inflated obstacle, not point robot


def test_astar_blocked_then_replanned():
    blocked=OccupancyGrid(bounds=(-3,3,-3,3));blocked.rectangle(-.2,.2,-3,3)
    with pytest.raises(NoPath):blocked.plan([-2,0],[2,0])
    updated=OccupancyGrid(bounds=(-3,3,-3,3));updated.rectangle(-.2,.2,-1,1)
    assert len(updated.plan([-2,0],[2,0]))>2


def test_door_gap_accounts_for_whole_robot():
    grid=OccupancyGrid();grid.rectangle(-.1,.1,.8,3);grid.rectangle(-.1,.1,-3,-.8)
    assert grid.plan([-2,0],[2,0])
    narrow=OccupancyGrid();narrow.rectangle(-.1,.1,.55,3);narrow.rectangle(-.1,.1,-3,-.55)
    with pytest.raises(NoPath):narrow.plan([-2,0],[2,0])


def test_rgbd_books_sorted_by_metric_area(tmp_path):
    path,meta=build('drawer',0,tmp_path/'drawer.xml')
    sim=Simulation(path)
    # Initial parked fixture for isolated vision validation, before any step.
    sim.data.qpos[:2]=meta['book_dock'][:2];mujoco.mj_forward(sim.model,sim.data)
    runtime=TaskRuntime(sim);runtime.hold(.6)
    camera=RGBDCamera(sim)
    try:
        rgb,depth=camera.capture()
        detected=observe_objects(camera,rgb,depth,workspace=(.34,.64,-1.,-.30),kind='book')
        assert len(detected)==3
        assert [d['category'] for d in sorted(detected,key=lambda o:o['area'])] == ['red','green','blue']
        for d in detected:
            truth=min((sim.data.xpos[sim.model.body(o['body']).id] for o in meta['objects']),key=lambda p:np.linalg.norm(p-d['center']))
            assert np.linalg.norm(truth-d['center']) < .015
    finally:camera.close()


def test_grasp_width_rejection_and_no_live_ik_writes():
    from robot_exam.control.arm import ArmController
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml');arm=ArmController(sim);qpos=sim.data.qpos.copy()
    assert not grasp_candidates(arm,{'center':[.55,0,.75],'length':.12,'width':.10,'pca_ratio':6})
    candidates=grasp_candidates(arm,{'center':[.55,0,.75],'category':'banana','yaw':.2,'length':.095,'width':.026})
    assert candidates and all(c['closing_width']<.078 for c in candidates)
    assert np.array_equal(qpos,sim.data.qpos) and sim.data.time == 0


def test_fsm_contracts_timeout_and_evidence(tmp_path):
    assert set(describe()) == {s.value for s in State}
    sim=Simulation(ROOT/'scenes/mobile_manipulator.xml');r=TaskRuntime(sim,tmp_path)
    r.transition(State.GRASP,'deliberate_timeout',timeout=.01)
    with pytest.raises(TaskFailure,match='exceeded'):r.hold(.02)
    assert json.loads((tmp_path/'fsm.json').read_text())[-1]['reason']=='deliberate_timeout'


@pytest.mark.parametrize('goal',[[0,0,0],[-1,1,np.pi/2]])
def test_physical_navigation_and_stopping(goal,tmp_path):
    path,meta=build('fruit',0,tmp_path/'fruit.xml')
    sim=Simulation(path);r=TaskRuntime(sim,tmp_path);r.hold(.6)
    r.transition(State.NAVIGATE,'physics_validation')
    grid=OccupancyGrid()
    for box in meta['obstacles']:grid.rectangle(*box)
    # The arm operates inside the inflated zone only during precision docking.
    target=[-.4,0,0] if goal==[0,0,0] else goal
    result=r.navigate(target,grid,timeout=180)
    assert result['position_error_m'] < .09
    assert result['heading_error_rad'] < .08
    assert not r.collision_events and not sim.data.warning.number.any()
    before=r.base.pose().copy();r.hold(1)
    assert np.linalg.norm(r.base.pose()[:2]-before[:2]) < .01
    assert np.linalg.norm(sim.data.qpos[7:11]) > 1 # wheel joints actually rotated
