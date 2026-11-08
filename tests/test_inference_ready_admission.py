import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import json

import pytest


def script():
    folder=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('admission_runner',folder/'run_fruit_fifty_evaluation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_ready_admission_retries_transport_but_rejects_wrong_model(monkeypatch):
    module=script();monkeypatch.setattr(module.time,'sleep',lambda _:None)
    calls=[];closed=[]
    metadata=dict(actual_cameras=['front','wrist'],action_semantics='ur5_absolute_joint_goal_v1',checkpoint_step_label='9999')
    def connect(*a,**kw):
        calls.append(True)
        if len(calls)==1:raise ConnectionRefusedError('still warming')
        return SimpleNamespace(get_server_metadata=lambda:metadata,_ws=SimpleNamespace(close=lambda:closed.append(True)))
    result=module.wait_for_ready({'server_checkpoint_label':'9999'},18006,connect=connect)
    assert result['attempts']==2 and result['physical_trials']==0 and len(closed)==1
    metadata['checkpoint_step_label']='999'
    with pytest.raises(ValueError,match='wrong checkpoint'):
        module.wait_for_ready({'server_checkpoint_label':'9999'},18006,connect=connect)


def test_unready_inference_never_spawns_or_counts_a_physics_trial(tmp_path,monkeypatch):
    module=script();config=tmp_path/'protocol.json';out=tmp_path/'results'
    config.write_text(json.dumps({'policies':['openpi'],'skills':['apple'],'scene_seeds':[15001]}))
    def unavailable(*a,**kw):raise TimeoutError('model not warmed')
    def forbidden(*a,**kw):raise AssertionError('Must not launch a physical process before admission')
    monkeypatch.setattr(module,'wait_for_ready',unavailable);monkeypatch.setattr(module.subprocess,'run',forbidden)
    monkeypatch.setattr(sys,'argv',['runner','--protocol',str(config),'--fixtures',str(tmp_path/'fixtures'),
                                 '--output',str(out),'--port','18006'])
    assert module.main()==1
    result=json.loads((out/'run_status.json').read_text())
    assert result['physical_trials']==0 and result['status']=='NOT_STARTED_INFERENCE_NOT_READY'
    assert not (out/'matrix').exists()
