import numpy as np
import pytest

from robot_exam.control.grasp_monitor import relative_grasp_pose


def yaw(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def test_rigid_grip_does_not_slip_during_common_translation_and_rotation():
    offset=np.array([.071,0,-.003])
    initial,initial_rotation=relative_grasp_pose(np.zeros(3),np.eye(3),offset,np.eye(3))
    for angle in np.linspace(-np.pi,np.pi,50):
        rotation=yaw(angle);tip=np.array([.4,-.5,.8])
        position,orientation=relative_grasp_pose(tip,rotation,tip+rotation@offset,rotation)
        assert np.linalg.norm(position-initial)<1e-12
        assert np.allclose(orientation,initial_rotation)
    # The old world-offset comparison falsely reports >25 mm for this rigid grip.
    assert np.linalg.norm(yaw(.4)@offset-offset)>.025


@pytest.mark.parametrize('angle',[0,.6,2.])
def test_true_26mm_slip_is_not_hidden_by_world_rotation(angle):
    offset=np.array([.071,0,-.003]);rotation=yaw(angle);tip=np.array([.2,.5,.9])
    position,_=relative_grasp_pose(tip,rotation,tip+rotation@(offset+[0,.026,0]),rotation)
    assert np.linalg.norm(position-offset)==pytest.approx(.026)
    assert np.linalg.norm(position-offset)>.025
