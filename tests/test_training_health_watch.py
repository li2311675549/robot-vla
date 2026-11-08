import importlib.util
from pathlib import Path
import subprocess
import sys


spec=importlib.util.spec_from_file_location('health_watch',Path(__file__).parents[1]/'tools/stage22_training_health_watch.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


def test_health_decisions_preserve_finished_job_and_reject_real_faults():
    normal=dict(guard=None,last_metric={'loss':.02,'grad_norm':.3},idle_seconds=30,files_unchanged=True)
    assert watch.decision(**normal)=='RUNNING'
    assert watch.decision(**{**normal,'guard':{'returncode':0}})=='FINISHED'
    assert watch.decision(**{**normal,'guard':{'returncode':-15}})=='ALERT_GUARD_STOPPED'
    assert watch.decision(**{**normal,'idle_seconds':901}).startswith('STOP_')
    assert watch.decision(**{**normal,'files_unchanged':False})=='STOP_FROZEN_INPUT_CHANGED'
    assert watch.decision(**{**normal,'last_metric':{'loss':float('nan'),'grad_norm':.3}})=='STOP_NONFINITE_METRIC'


def test_signal_targets_exact_training_child_and_never_unrelated_process(tmp_path):
    script=tmp_path/'openpi_ur5_train.py';script.write_text('import time\ntime.sleep(30)\n')
    own=tmp_path/'own';own.mkdir();other=tmp_path/'other';other.mkdir()
    p=subprocess.Popen([sys.executable,str(script),'--output',str(own/'training.json')],start_new_session=True)
    q=subprocess.Popen([sys.executable,str(script),'--output',str(other/'training.json')],start_new_session=True)
    try:
        assert watch.terminate_owned_training(own,script)==[p.pid]
        assert p.wait(timeout=5)==-15
        assert q.poll() is None
    finally:
        for child in (p,q):
            if child.poll() is None:child.terminate();child.wait(timeout=5)


def test_transient_partial_guard_json_is_retried_without_touching_training(tmp_path,monkeypatch):
    path=tmp_path/'guard.json'
    assert watch.read_json(path) is None
    path.write_text('{"returncode":')
    monkeypatch.setattr(watch.time,'sleep',lambda _:path.write_text('{"returncode":0}'))
    assert watch.read_json(path)=={'returncode':0}
