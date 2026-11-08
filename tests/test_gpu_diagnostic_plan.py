"""No-card preparation must not accidentally allocate a model/GPU."""
import json
from pathlib import Path
import subprocess
import sys


def arguments(tmp_path):
    meta=tmp_path/'cache/local/diagnostic_train/meta';meta.mkdir(parents=True)
    (meta/'episodes.jsonl').write_text(''.join(json.dumps({'episode_index':i,'length':102,
        'tasks':[f'Pick up the {skill} and place it at the target.']})+'\n'
        for i,skill in enumerate(('apple','banana','orange'))))
    script=Path(__file__).resolve().parents[1]/'scripts/diagnose_pi05_conditions.py'
    return [str(script),'--openpi-root',str(tmp_path/'no_openpi'),'--assets',str(tmp_path/'no_assets'),
        '--checkpoint',str(tmp_path/'no_weights'),'--tokenizer',str(tmp_path/'no_tokenizer'),
        '--lerobot-cache',str(tmp_path/'cache'),'--repo-id','local/diagnostic_train',
        '--output',str(tmp_path/'plan.json')]


def test_default_diagnostic_is_plan_only_without_model_dependencies(tmp_path):
    run=subprocess.run([sys.executable,*arguments(tmp_path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    plan=json.loads((tmp_path/'plan.json').read_text())
    assert plan['expected_queries']==54
    assert plan['model_loaded'] is False and plan['training_updates']==plan['actuator_commands']==0


def test_explicit_diagnostic_refuses_no_gpu_before_model_imports(tmp_path):
    # Only override the hardware probe in this isolated test subprocess.
    wrapper="import glob,runpy,sys; glob.glob=lambda pattern: []; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    run=subprocess.run([sys.executable,'-c',wrapper,*arguments(tmp_path),'--authorized-gpu-diagnostic'],capture_output=True,text=True)
    assert run.returncode!=0
    assert 'Refuse model loading without GPU' in run.stderr
    assert not (tmp_path/'plan.json').exists()


def test_nonzero_physical_gpu_index_reaches_memory_check_before_model_import(tmp_path):
    wrapper="import glob,pathlib,runpy,sys; glob.glob=lambda pattern: ['/dev/nvidia3']; original=pathlib.Path.read_text; pathlib.Path.read_text=lambda p,*a,**kw: '2147483648' if str(p)=='/sys/fs/cgroup/memory.max' else original(p,*a,**kw); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    run=subprocess.run([sys.executable,'-c',wrapper,*arguments(tmp_path),'--authorized-gpu-diagnostic'],capture_output=True,text=True)
    assert run.returncode!=0
    assert 'requires the prepared server memory budget' in run.stderr
    assert 'Refuse model loading without GPU' not in run.stderr
    assert not (tmp_path/'plan.json').exists()
