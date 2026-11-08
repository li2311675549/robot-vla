"""Actual wheel-driven terminal stopping with and without a physical fruit."""
import sys

from robot_exam.simulation import ROOT

sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_dock_stage7 import fixture, trial


def test_empty_husky_dock_stays_inside_original_tolerance(tmp_path):
    scene,snapshot=fixture(0,'empty',tmp_path/'empty')
    result=trial(scene,snapshot,'hysteresis',tmp_path/'empty'/'run','empty')
    assert result['strict_success_after_2s'],result
    assert result['position_error_m']<.023
    assert result['heading_error_rad']<.035
    assert result['linear_speed_m_s']<.02
    assert result['yaw_rate_rad_s']<.03


def test_physically_grasped_orange_dock_stays_stable(tmp_path):
    scene,snapshot=fixture(0,'loaded',tmp_path/'loaded')
    assert snapshot['initial_bilateral_contact']
    assert snapshot['grasp']['lift_m']>.055
    result=trial(scene,snapshot,'hysteresis',tmp_path/'loaded'/'run','loaded')
    assert result['strict_success_after_2s'],result
    assert result['bilateral_contact']
