import sys
from pathlib import Path
import pytest

from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import State,TaskFailure,TaskRuntime

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_banana_closure_stage7 import fixture,trial
from build_tasks import build


def test_regulated_closure_keeps_real_banana_during_reverse_transport(tmp_path):
    scene,info=fixture(420,tmp_path/'fixture')
    assert info['physical_lift_m']>.055
    assert info['initial_bilateral_contact']
    result=trial(scene,tmp_path/'fixture',info,.70,tmp_path/'regulated')
    assert result['transport_success'],result
    assert result['travel_m']>.35
    assert result['terminal_bilateral']
    assert result['max_relative_drift_m']<.025
    assert not any(result['warnings'])


def test_loaded_rotate_command_reaches_heading_without_losing_banana(tmp_path):
    scene,info=fixture(460,tmp_path/'fixture')
    result=trial(scene,tmp_path/'fixture',info,.70,tmp_path/'turn',
                 turn_radians=.3,turn_controller='rotate')
    assert result['transport_success'],result
    assert result['terminal_bilateral']
    assert result['terminal_base_pose'][2]-info['initial_base'][2]>.29
    assert result['elapsed_sim_seconds']<35
    assert not any(result['warnings'])


def test_navigation_preserves_payload_slip_for_bounded_recovery(tmp_path):
    scene,_=build('fruit',0,tmp_path/'scene.xml')
    runtime=TaskRuntime(Simulation(scene),output=tmp_path/'run')
    runtime.transition(State.NAVIGATE,'failure_propagation_probe')
    def slip_on_step():
        raise TaskFailure('gripper_slip','measured contact witness lost')
    runtime.tick=slip_on_step
    with pytest.raises(TaskFailure,match='measured contact witness lost') as error:
        runtime.navigate([.5,0,0],OccupancyGrid())
    assert error.value.category=='gripper_slip'
    assert runtime.retries==0
