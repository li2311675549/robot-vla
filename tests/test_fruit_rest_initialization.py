import copy
import sys

import mujoco
import numpy as np
import pytest

from robot_exam.data.rest_initialization import apply_rest_initialization,configure_rest_initialization
from robot_exam.simulation import ROOT,Simulation


def test_passively_calibrated_initial_poses_update_keyframes_without_changing_physics(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_tasks import build
    from calibrate_fruit_rest_pose import calibrate
    source,metadata=build('fruit',2100005,tmp_path/'source.xml',fruit_count=3,
                          layout_profile='bounded_scatter_v3',wrist_camera=True)
    source_bytes=source.read_bytes()
    result=calibrate(source,tmp_path/'calibration',seconds=600.)
    assert result['status']=='PASS_ALL_EXACT_FRUITS_STABLE'
    target=tmp_path/'initialized.xml';target.write_bytes(source_bytes)
    before=Simulation(source)
    apply_rest_initialization(target,copy.deepcopy(metadata),tmp_path/'calibration/calibration.json')
    after=Simulation(target)
    assert source.read_bytes()==source_bytes
    for attr in ('body_mass','body_inertia','geom_size','geom_friction','actuator_gainprm','actuator_forcerange'):
        np.testing.assert_array_equal(getattr(before.model,attr),getattr(after.model,attr))
    first_fruit=int(before.model.jnt_qposadr[before.model.body_jntadr[before.model.body('fruit_0').id]])
    np.testing.assert_array_equal(before.data.qpos[:first_fruit],after.data.qpos[:first_fruit])
    for obj in metadata['objects']:
        body=after.model.body(obj['body']).id;adr=int(after.model.jnt_qposadr[after.model.body_jntadr[body]])
        np.testing.assert_array_equal(before.data.qpos[adr:adr+2],after.data.qpos[adr:adr+2])
        np.testing.assert_allclose(after.data.qpos[adr:adr+7],after.model.qpos0[adr:adr+7],atol=1e-14)
    assert any(abs(np.array(result['rows'][i]['rest_quaternion'])[1:]).max()>.1 for i in range(3))


def test_existing_profiles_are_noop_and_missing_exact_calibration_rejects():
    metadata={'task':'fruit','seed':1}
    assert configure_rest_initialization('nonexistent.xml',metadata,{}) is metadata
    with pytest.raises(ValueError,match='Exact-fruit'):
        configure_rest_initialization('nonexistent.xml',metadata,{'fruit_spawn_profile':'passive_rest_init_v2'})
