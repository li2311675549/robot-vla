"""Opt-in fast path commands must remain finite and respect inflated obstacles."""
import numpy as np
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.navigation.pursuit import pursuit_twist,arc_is_free


def test_reverse_target_can_be_reached_without_an_initial_half_turn():
    v,w,pivot=pursuit_twist([0,0,0],[-1,0],.8)
    assert v<0 and abs(w)<1e-9 and not pivot


def test_prediction_rejects_a_forward_arc_entering_an_inflated_obstacle():
    grid=OccupancyGrid(bounds=(-2,2,-2,2))
    grid.rectangle(.7,.9,-.1,.1)
    assert not arc_is_free(grid,[0,0,0],.8,0,horizon=.6)
    assert arc_is_free(grid,[-1.5,0,0],.8,0,horizon=.35)


def test_terminal_approach_decelerates_and_requests_only_finite_wheel_motion():
    far=pursuit_twist([0,0,0],[2,0],.8,final=True)
    near=pursuit_twist([0,0,0],[.06,0],.8,final=True)
    assert 0<near[0]<far[0]<=.8
    assert np.isfinite([*far[:2],*near[:2]]).all()


def test_slew_braking_continues_during_hold_without_reissuing_commands(tmp_path):
    import sys
    from robot_exam.simulation import ROOT
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    from robot_exam.simulation import Simulation
    from robot_exam.tasks.runtime import TaskRuntime
    scene,_=build('door',2101,tmp_path/'braking.xml',door_profile='aligned_handles_v3')
    sim=Simulation(scene);runtime=TaskRuntime(sim);runtime.state_timeout=20
    runtime.config={'empty_twist_slew':True,'empty_acceleration_m_s2':.5,'empty_braking_m_s2':.8}
    runtime.hold(.6)
    runtime.base.command(.6,0);runtime.hold(2.)
    assert np.linalg.norm(sim.data.qvel[runtime.base_velocity_dof:runtime.base_velocity_dof+2])>.2
    runtime.base.command(0,0);runtime.hold(2.)
    assert np.max(np.abs(sim.data.ctrl[runtime.base.ids]))<1e-9
    assert np.linalg.norm(sim.data.qvel[runtime.base_velocity_dof:runtime.base_velocity_dof+2])<.02


def test_measured_twist_turn_physically_stops_without_orbiting(tmp_path):
    import math
    import sys
    from robot_exam.simulation import ROOT,Simulation
    from robot_exam.tasks.runtime import TaskRuntime
    from robot_exam.control.base import wrap_angle
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    scene,_=build('door',2101,tmp_path/'turn.xml',door_profile='aligned_handles_v3')
    runtime=TaskRuntime(Simulation(scene));runtime.state_timeout=30
    runtime.hold(.6);initial=runtime.base.pose().copy();started=runtime.sim.data.time
    while runtime.sim.data.time-started<12.:
        if abs(wrap_angle(initial[2]+math.pi/2-runtime.base.pose()[2]))<.03:break
        runtime.base.rotate_velocity(initial[:2],initial[2]+math.pi/2);runtime.tick()
    runtime.base.command(0,0);runtime.hold(2.)
    assert abs(wrap_angle(initial[2]+math.pi/2-runtime.base.pose()[2]))<.035
    assert np.linalg.norm(runtime.base.pose()[:2]-initial[:2])<.05
    velocity=runtime.sim.data.qvel[runtime.base_velocity_dof:runtime.base_velocity_dof+6]
    assert np.linalg.norm(velocity[:2])<.02 and abs(velocity[5])<.03
    assert not runtime.sim.data.warning.number.any()


def test_trajectory_timing_obeys_actual_controller_speed(tmp_path):
    import sys
    from robot_exam.simulation import ROOT,Simulation
    from robot_exam.tasks.runtime import TaskRuntime
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    scene,_=build('door',2101,tmp_path/'rate.xml',door_profile='aligned_handles_v3')
    r=TaskRuntime(Simulation(scene));r.state_timeout=30
    r.config={'free_arm_speed_rad_s':.3,'trajectory_joint_speed_rad_s':2.8,
              'motion_duration_scale':.2,'joint_settle_seconds':.15}
    r.hold(.6);target=r.sim.data.qpos[r.arm.qadr].copy();target[0]+=.18
    started=float(r.sim.data.time);r.joint_move(target,seconds=.1)
    assert r.sim.data.time-started>=.96
    assert np.max(np.abs(r.sim.data.qpos[r.arm.qadr]-target))<.06
    assert not r.sim.data.warning.number.any()


def test_rejected_camera_forearm_self_collision_preserves_live_state(tmp_path):
    """The real failed fruit placement folded the forearm into the camera.

    Evaluate this independent negative fixture only in IK scratch data. Normal
    STOW remains valid; neither the live pose nor the collision model changes.
    """
    import sys
    from robot_exam.simulation import ROOT, Simulation
    from robot_exam.tasks.runtime import TaskRuntime
    sys.path.insert(0, str(ROOT / 'scripts'))
    from build_tasks import build
    scene, _ = build('fruit', 412, tmp_path / 'self_collision.xml')
    r = TaskRuntime(Simulation(scene))
    bad_q = np.array([-2.3381023211, -1.7217042857, 1.3742735547,
                      -1.2269503987, -1.5715447466, -2.4873576055])
    live_qpos = r.sim.data.qpos.copy()
    live_ctrl = r.sim.data.ctrl.copy()
    r.arm.check_self_collisions = True
    contacts = r.arm._collision_report(bad_q)
    assert any('d435i_virtual' in c['other_body'] and
               'forearm_link' in c['other_body'] for c in contacts)
    assert max(c['maximum_penetration_m'] for c in contacts) > .002
    assert not r.arm._collision_report(r.STOW)
    np.testing.assert_array_equal(r.sim.data.qpos, live_qpos)
    np.testing.assert_array_equal(r.sim.data.ctrl, live_ctrl)


def test_reverse_clearance_brakes_before_losing_working_distance(tmp_path):
    import sys
    from robot_exam.simulation import ROOT, Simulation
    from robot_exam.tasks.runtime import TaskRuntime
    sys.path.insert(0, str(ROOT / 'scripts'))
    from build_tasks import build
    scene, _ = build('door', 2101, tmp_path / 'reverse.xml',
                     door_profile='aligned_handles_v3')
    r = TaskRuntime(Simulation(scene)); r.state_timeout = 20
    r.config = {'backout_linear_speed_m_s': .4,
                'backout_distance_braking': True, 'empty_twist_slew': True,
                'empty_acceleration_m_s2': .5, 'empty_braking_m_s2': 1.2}
    r.hold(.6); target_x = r.base.pose()[0] - .30
    r.backout(target_x)
    assert target_x - .02 < r.base.pose()[0] <= target_x + .012
    velocity = r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof + 6]
    assert np.linalg.norm(velocity[:2]) < .02 and abs(velocity[5]) < .03
    assert not r.sim.data.warning.number.any()


def test_speed_snapshot_requires_explicit_book_geometry(tmp_path):
    """A timing comparison must not silently replace three distinct books."""
    import subprocess
    import sys
    from robot_exam.simulation import ROOT
    output = tmp_path / 'must_not_start'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/run_speed_snapshot.py'),
         '--task', 'drawer', '--output', str(output)],
        text=True, capture_output=True, cwd=ROOT)
    assert result.returncode == 2
    assert 'explicit --book-profile' in result.stderr
    assert not output.exists()
