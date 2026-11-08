import math
import sys
import numpy as np
import mujoco

from robot_exam.simulation import ROOT,Simulation
from robot_exam.control.arm import ArmController
from robot_exam.perception.objects import grasp_candidates

sys.path.insert(0,str(ROOT/'scripts'))
from build_tasks import build


def test_multistart_door_is_reproducible_and_five_metres(tmp_path):
    _,a=build('door',801,tmp_path/'door_a.xml',door_profile='multistart_v2')
    _,b=build('door',801,tmp_path/'door_b.xml',door_profile='multistart_v2')
    _,c=build('door',802,tmp_path/'door_c.xml',door_profile='multistart_v2')
    assert a['start']==b['start'] and a['start']!=c['start']
    assert math.isclose(a['start_distance_m'],5.,abs_tol=1e-12)
    assert abs(a['start'][1])<=.25 and abs(a['start'][2])<=.18


def test_distinct_books_have_frozen_sizes_passive_slides_and_space(tmp_path):
    scene,meta=build('drawer',811,tmp_path/'books.xml',book_profile='distinct_v2')
    model=mujoco.MjModel.from_xml_path(str(scene))
    sizes=sorted((o['size'] for o in meta['objects']),key=np.prod)
    assert sizes==[[.120,.080,.015],[.180,.120,.020],[.240,.160,.025]]
    assert sizes[-1][0]*sizes[-1][1]/(sizes[0][0]*sizes[0][1])==4.
    assert math.isclose(.28-meta['start'][0],5.,abs_tol=1e-12)
    for layer in range(3):
        joint=model.joint(f'drawer_slide_{layer}').id
        assert not np.any(model.actuator_trnid[:,0]==joint)
    # Fixed geometric footprint and vertical net clearance; manipulation
    # reachability is checked by separate physical task attempts.
    assert sizes[-1][0]<.28 and sizes[-1][1]<.38 and sizes[-1][2]<.12


def test_narrow_booklet_keeps_previously_valid_top_grasp():
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml')
    arm=ArmController(sim)
    candidates=grasp_candidates(arm,{'category':'red','center':[.52,0,.735],
        'yaw':0.,'length':.135,'width':.058,'thickness':.020,'pca_ratio':5.},solve_ik=False)
    assert candidates and all(c['grasp_type']=='topdown' for c in candidates)
    assert all(c['closing_width']<=.078 for c in candidates)
