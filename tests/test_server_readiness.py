import json
import subprocess
import sys

from robot_exam.simulation import ROOT


def test_subset_prefetch_cannot_be_mistaken_for_a_complete_model(tmp_path):
    manifest=json.loads((ROOT/'configs/pi05_official_assets_20261030.json').read_text())
    manifest['objects']=manifest['objects'][-1:]
    path=tmp_path/'subset.json';path.write_text(json.dumps(manifest))
    output=tmp_path/'result.json'
    process=subprocess.run([sys.executable,str(ROOT/'scripts/verify_official_pi05_assets.py'),
        '--manifest',str(path),'--cache',str(tmp_path),'--output',str(output)],capture_output=True,text=True)
    result=json.loads(output.read_text())
    assert process.returncode==1 and result['status']=='FAIL'
    assert 'subset' in result['failures'][0]['error']
    assert result['network_used'] is False and result['model_loaded'] is False


def test_existing_empty_checkpoint_directory_is_not_ready(tmp_path):
    cache=tmp_path/'cache';(cache/'openpi-assets/checkpoints/pi05_base/params').mkdir(parents=True)
    output=tmp_path/'result.json'
    process=subprocess.run([sys.executable,str(ROOT/'scripts/verify_official_pi05_assets.py'),
        '--manifest',str(ROOT/'configs/pi05_official_assets_20261030.json'),
        '--cache',str(cache),'--output',str(output)],capture_output=True,text=True)
    result=json.loads(output.read_text())
    assert process.returncode==1 and result['status']=='FAIL' and len(result['failures'])==21
    assert result['verified']==[] and result['training_performed'] is False


def test_tomorrow_launcher_requires_an_explicit_mode_before_any_execution():
    script=ROOT/'scripts/autodl_tomorrow.sh'
    process=subprocess.run(['bash',str(script)],capture_output=True,text=True)
    assert process.returncode==2 and 'Usage:' in process.stdout
    help_process=subprocess.run(['bash',str(script),'--help'],capture_output=True,text=True)
    assert help_process.returncode==0 and '30-step' in help_process.stdout
