import importlib.util
import json
import sys

from robot_exam.simulation import ROOT

sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('parallel_frozen',ROOT/'scripts/evaluate_frozen_parallel.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_interrupted_formal_attempt_becomes_a_persistent_failure(tmp_path):
    folder=tmp_path/'94001';folder.mkdir()
    result=module.previous_attempt(folder,94001)
    assert result['success'] is False and result['failure_category']=='process_interrupted'
    assert json.loads((folder/'result.json').read_text())==result
    assert module.previous_attempt(folder,94001)==result


def test_unstarted_formal_seed_is_distinct_from_existing_failure(tmp_path):
    folder=tmp_path/'94002'
    assert module.previous_attempt(folder,94002) is None and not folder.exists()
    folder.mkdir();original={'seed':94002,'success':False,'failure_category':'handle_grasp','sim_seconds':27.0}
    (folder/'result.json').write_text(json.dumps(original))
    assert module.previous_attempt(folder,94002)==original
