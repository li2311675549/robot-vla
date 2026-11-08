import numpy as np
import pytest
from robot_exam.skills.book_edge import book_centerline_halfwidth


@pytest.mark.parametrize('yaw',[.3,1.6766792,1.869])
def test_centreline_grasp_lies_inside_the_rotated_book(yaw):
    r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    size=np.array([.24,.16,.025]);half=book_centerline_halfwidth(r,size)
    local=r.T@np.array([-half+.015,0,0])
    assert np.all(np.abs(local[:2])<size[:2]/2)
    boundary=r.T@np.array([-half,0,0])
    assert np.max(np.abs(boundary[:2])/(size[:2]/2))==pytest.approx(1.)


def test_old_aabb_inset_can_miss_actual_rotated_book():
    yaw=1.869;r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    size=np.array([.24,.16,.025]);aabb=np.abs(r[0,:2])@(size[:2]/2)
    old=r.T@np.array([-aabb+.015,0,0])
    assert np.any(np.abs(old[:2])>size[:2]/2)
    assert aabb-book_centerline_halfwidth(r,size)>.025


@pytest.mark.parametrize('drift,contact,distance,expected',[
    (.002,True,.0866,True),   # Actual large-book failure: both pads,2mm drift.
    (.016,True,.0866,False),
    (.002,False,.0866,False),
    (.002,True,.20,False),
])
def test_edge_book_witness_preserves_slip_and_contact_rejection(drift,contact,distance,expected):
    from types import SimpleNamespace
    from robot_exam.skills.book_edge import stable_edge_book
    state=SimpleNamespace(time=0.,xpos=np.array([[distance,0.,0.]]))
    runtime=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(body=lambda _:SimpleNamespace(id=0)),data=state),
                            arm=SimpleNamespace(pose=lambda:(np.zeros(3),np.eye(3))),
                            bilateral=lambda _:contact)
    def hold(dt):
        state.time+=dt;state.xpos[0,1]=drift*min(1.,state.time/.6)
    runtime.hold=hold
    assert stable_edge_book(runtime,{'body':'book_2','size':[.24,.16,.025]},
                            reference_relative=[distance,0.,0.]) is expected


@pytest.mark.parametrize('yaw',[.6,1.,1.869])
def test_long_edge_normal_grasp_keeps_gripper_facing_drawer_front_after_alignment(yaw):
    from robot_exam.skills.book_edge import long_edge_grasp_frame
    r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    axis,support,grasp=long_edge_grasp_frame(r,[.24,.16,.025])
    assert support>=.032 and axis[0]>0
    np.testing.assert_allclose(grasp[:,1],[0,0,1.])
    delta=np.pi/2-yaw;c,s=np.cos(delta),np.sin(delta)
    aligned=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@grasp
    assert aligned[0,2]>.99999


def test_long_edge_normal_refuses_unsupported_overhang():
    from robot_exam.skills.book_edge import long_edge_grasp_frame
    with pytest.raises(RuntimeError,match='centre safely supported'):
        long_edge_grasp_frame(np.eye(3),[.24,.16,.025])
