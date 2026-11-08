#!/usr/bin/env python3
"""Run an immutable archived full-fruit controller, not altered current code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--view-azimuth',type=float,default=135.)
    a=p.parse_args();code=a.code_root.resolve();output=a.output.resolve()
    if output.exists():raise FileExistsError('Preserve recordings')
    output.relative_to(code)  # Old builder requires generated scenes under its own root.
    sys.path[:0]=[str(code/'scripts'),str(code/'src')]
    import run_task
    original=run_task.VideoRecorder
    class ViewRecorder(original):
        def __init__(self,path,camera=None,**kwargs):
            camera.azimuth=a.view_azimuth
            super().__init__(path,camera=camera,**kwargs)
    run_task.VideoRecorder=ViewRecorder
    r=run_task.run('fruit',a.seed,'vision',output,record=True,config=a.config)
    (output/'recording_manifest.json').write_text(json.dumps({'scope':'Actual full physical rerun of a known legacy initial condition, not a new heldout trial',
        'legacy_git_commit':'b4ef295','code_root':str(code),'source_script_sha256':hashlib.sha256((code/'scripts/run_task.py').read_bytes()).hexdigest(),
        'view_changes_only_external_recording_camera':True,'view_azimuth_degrees':a.view_azimuth,
        'robot_policy_camera_changed':False,'success':r.get('success')},indent=2))


if __name__=='__main__':main()
