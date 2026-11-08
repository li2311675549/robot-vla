import numpy as np
import pytest

from robot_exam.vla.interface import validate_actions
from robot_exam.vla.policy_safety import ProjectGripperOutputs


def test_projection_only_affects_small_gripper_extrapolation_and_reports_it():
    raw=np.zeros((3,7));raw[:,:6]=.25;raw[:,6]=[-.02,.5,1.03]
    original=raw.copy();out=ProjectGripperOutputs()({'actions':raw})
    np.testing.assert_array_equal(raw,original)
    np.testing.assert_allclose(out['actions'][:,:6],raw[:,:6])
    np.testing.assert_allclose(out['actions'][:,6],[0.,.5,1.])
    assert out['gripper_projection']['projected_values']==2
    with pytest.raises(ValueError,match='Gripper'):validate_actions(raw)


@pytest.mark.parametrize('value',[1.5,-.2,np.nan,np.inf])
def test_projection_does_not_admit_severe_or_nonfinite_model_actions(value):
    raw=np.zeros((10,7));raw[0,6]=value
    with pytest.raises(ValueError):ProjectGripperOutputs()({'actions':raw})
