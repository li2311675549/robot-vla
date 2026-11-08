import numpy as np
import pytest
from robot_exam.navigation.local_dock import plan_dock
from robot_exam.control.base import wrap_angle


@pytest.mark.parametrize('strategy',['preapproach','lattice'])
def test_local_dock_path_has_no_lateral_primitives(strategy):
    report=plan_dock([0,0,0],[-.15,.07,.1],strategy)
    poses=np.asarray(report['poses'])
    assert np.allclose(poses[-1],[-.15,.07,.1])
    for a,b in zip(poses[:-1],poses[1:]):
        delta=b[:2]-a[:2]
        if np.linalg.norm(delta)>1e-6:
            assert abs(wrap_angle(b[2]-a[2]))<1e-6
            assert abs(delta@[-np.sin(a[2]),np.cos(a[2])])<1e-7


def test_local_dock_refuses_blocked_goal():
    with pytest.raises(RuntimeError):
        plan_dock([0,0,0],[.2,.2,0],'preapproach',valid=lambda p:p[0]<.1)


def test_settled_entry_keeps_strict_pose_without_unneeded_replan(tmp_path,monkeypatch):
    import sys
    from robot_exam.simulation import ROOT,Simulation
    from robot_exam.tasks.runtime import TaskRuntime,State
    import robot_exam.control.local_dock as local
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    path,_=build('fruit',2100002,tmp_path/'scene.xml')
    sim=Simulation(path);r=TaskRuntime(sim);r.hold(.7)
    r.config={'local_dock_strategy':'direct','local_dock_preserve_settled_entry':True}
    goal=r.base.pose().copy();goal[0]+=.004;goal[1]-=.003
    def unnecessary_plan(*args,**kwargs):
        raise AssertionError('An already-arrived, stopped pose must be checked physically before replanning')
    monkeypatch.setattr(local,'plan_dock',unnecessary_plan)
    start=float(sim.data.time);r.transition(State.NAVIGATE,'test_steady_entry',timeout=5)
    r.dock(goal,timeout=4,tolerance=.023)
    assert sim.data.time-start>=2.-1e-8
    assert np.linalg.norm(r.base.pose()[:2]-goal[:2])<.023
    assert r.dock_traces[-1]['plans']==[]
    assert r.dock_traces[-1]['stable_seconds']>=2.
    assert not sim.data.warning.number.any()


def test_final_pose_capture_brakes_even_when_intermediate_waypoints_remain(tmp_path):
    import sys
    from robot_exam.simulation import ROOT,Simulation
    from robot_exam.tasks.runtime import TaskRuntime,State
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    path,_=build('fruit',2100005,tmp_path/'scene.xml')
    sim=Simulation(path);r=TaskRuntime(sim);r.hold(.7)
    # Entry shortcut deliberately disabled: the local connector would pivot
    # toward the bearing of a5mm residual despite being inside the final pose.
    r.config={'local_dock_strategy':'direct','local_dock_tracking_version':'axis_bounded_v5',
              'local_dock_capture_goal_during_primitives':True}
    goal=r.base.pose().copy();goal[:2]+=[.004,.003]
    start=float(sim.data.time);r.transition(State.NAVIGATE,'capture_test',timeout=5)
    r.dock(goal,timeout=4,tolerance=.023)
    assert sim.data.time-start>=2.-1e-8
    assert np.linalg.norm(r.base.pose()[:2]-goal[:2])<.023
    trace=r.dock_traces[-1]
    assert trace['plans'] and trace['stable_seconds']>=2.
    assert all(s['linear_command_m_s']==0 and s['angular_command_rad_s']==0 for s in trace['samples'])
    assert not sim.data.warning.number.any()


def test_local_preapproach_retains_real_orange_and_stops(tmp_path):
    import sys
    from robot_exam.simulation import ROOT
    sys.path.insert(0,str(ROOT/'scripts'))
    from benchmark_dock_stage7 import fixture,trial
    scene,snapshot=fixture(7,'loaded',tmp_path/'physical_fixture')
    assert snapshot['initial_bilateral_contact']
    result=trial(scene,snapshot,'preapproach',tmp_path/'trial','loaded',timeout=60,
                 config_overrides={'local_dock_strategy':'preapproach'})
    assert result['strict_success_after_2s'],result
    assert result['bilateral_contact']
    assert result['position_error_m']<.023
    assert result['heading_error_rad']<.035
