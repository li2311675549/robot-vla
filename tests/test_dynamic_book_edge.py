import numpy as np
import pytest

from robot_exam.skills.book_edge import supported_push_halfwidth,book_centerline_halfwidth
from robot_exam.tasks.runtime import TaskFailure


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def test_push_rotation_changes_edge_eligibility_without_relaxing_support():
    size=[.24,.16,.025];before=rotation(np.pi/4);after=rotation(np.deg2rad(20))
    width,normal=supported_push_halfwidth(before,size,True,True)
    assert normal and width>=.032
    with pytest.raises(TaskFailure,match='centre safely supported'):
        supported_push_halfwidth(after,size,True,False)
    saved=after.copy();width,normal=supported_push_halfwidth(after,size,True,True)
    assert not normal and width==pytest.approx(book_centerline_halfwidth(after,size))
    np.testing.assert_array_equal(after,saved)


def test_dynamic_fallback_rejects_tilted_book_instead_of_fabricating_support():
    r=rotation(.7);a=.4;c,s=np.cos(a),np.sin(a)
    r=r@np.array([[1,0,0],[0,c,-s],[0,s,c]])
    with pytest.raises(TaskFailure,match='not flat enough'):
        supported_push_halfwidth(r,[.24,.16,.025],True,True)
