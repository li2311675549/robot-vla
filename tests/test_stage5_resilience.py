import sys
import numpy as np

from robot_exam.control.arm import ArmController
from robot_exam.perception.objects import grasp_candidates
from robot_exam.simulation import ROOT, Simulation


def test_a6_a5_a4_profile_preserves_full_physical_dimensions(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    path,metadata=build('drawer',602,tmp_path/'a4_scene.xml',book_profile='a6_a5_a4')
    assert metadata['book_profile']=='a6_a5_a4'
    by_rank=sorted(metadata['objects'],key=lambda x:x['rank'])
    assert [o['size'] for o in by_rank]==[[.148,.105,.018],[.210,.148,.022],[.297,.210,.028]]
    model=Simulation(path).model
    assert model.body('book_2').id>0


def test_thin_a4_book_gets_a_physical_edge_grasp_candidate():
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml')
    arm=ArmController(sim)
    detection={'category':'blue','center':[.45,0,.745],'yaw':1.57,'length':.297,
               'width':.210,'thickness':.028,'pca_ratio':1.4}
    candidates=grasp_candidates(arm,detection)
    assert candidates
    assert all(c['grasp_type']=='thin_book_edge' for c in candidates)
    assert all(c['closing_width']<=.078 for c in candidates)
    base=arm.pose()[0][:2]
    radial=np.asarray(detection['center'][:2])-base
    for candidate in candidates:
        delta=np.asarray(candidate['pregrasp'])-candidate['position']
        assert abs(np.linalg.norm(delta)-.075)<1e-9
        assert np.dot(delta[:2],radial)<0


def test_thin_book_candidate_generation_covers_both_pca_edges():
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml');arm=ArmController(sim)
    detection={'category':'blue','center':[.45,0,.745],'yaw':.4,'length':.297,
               'width':.210,'thickness':.028,'pca_ratio':1.4}
    candidates=grasp_candidates(arm,detection,solve_ik=False)
    assert {candidate['approach_axis'] for candidate in candidates}=={'long','short'}
    assert all(np.isfinite(candidate['position']).all() and
               np.isfinite(candidate['pregrasp']).all() for candidate in candidates)


def test_multistart_ik_solves_known_pose_without_mutating_live_state():
    import mujoco
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml');arm=ArmController(sim)
    desired=np.array([.3,-.8,1.2,-1.4,-1.57,.2])
    arm.scratch.qpos[:]=sim.data.qpos;arm.scratch.qpos[arm.qadr]=desired
    mujoco.mj_forward(sim.model,arm.scratch)
    target=arm.scratch.site_xpos[arm.site].copy();rotation=arm.scratch.site_xmat[arm.site].reshape(3,3).copy()
    before=sim.data.qpos.copy();seed=np.array([2.,-1.,1.,-1.,0.,2.])
    q,metrics=arm.solve_multi(target,rotation,seed=seed,max_iterations=500)
    arm.scratch.qpos[:]=sim.data.qpos;arm.scratch.qpos[arm.qadr]=q;mujoco.mj_forward(sim.model,arm.scratch)
    assert metrics['position_error_m']<.0005
    assert metrics['orientation_error_rad']<.003
    limits=sim.model.jnt_range[arm.joints]
    assert np.all(q>=limits[:,0]) and np.all(q<=limits[:,1])
    assert np.array_equal(before,sim.data.qpos)


def test_unreachable_ik_retains_branch_and_joint_limit_diagnostics():
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml');arm=ArmController(sim)
    before=sim.data.qpos.copy()
    with np.testing.assert_raises(RuntimeError):
        arm.solve_multi([3.,0.,2.],np.diag([1.,-1.,-1.]),max_iterations=80)
    report=arm.last_solve_report
    assert report['status']=='failed_all_branches'
    assert report['joint_limits_rad']==sim.model.jnt_range[arm.joints].tolist()
    assert report['elapsed_s']>0 and len(report['trials'])>=4
    assert all('position_error_m' in trial for trial in report['trials'])
    assert np.array_equal(before,sim.data.qpos)
