#!/usr/bin/env python3
"""Execute an unchanged full task with a recording-only external viewpoint.

The two robot policy cameras, MJCF, controls and scoring are untouched. This
wrapper changes only the free camera passed to the existing VideoRecorder.
"""
import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT
import run_task


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['door','fruit','drawer'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['oracle','vision'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view-azimuth',type=float,default=145.)
    p.add_argument('--book-profile',default='distinct_v2');p.add_argument('--door-profile',default='aligned_handles_v3')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Never overwrite a recorded attempt')
    if not -360<=a.view_azimuth<=360:raise ValueError('Finite bounded recording azimuth required')
    original=run_task.VideoRecorder
    class ViewRecorder(original):
        def __init__(self,output,camera=None,**kwargs):
            camera.azimuth=a.view_azimuth
            super().__init__(output,camera=camera,**kwargs)
    run_task.VideoRecorder=ViewRecorder
    result=run_task.run(a.task,a.seed,a.mode,a.output,record=True,config=a.config,
        book_profile=a.book_profile,door_profile=a.door_profile)
    descriptor={'scope':'Full task physical rerun, not independent formal evaluation or model inference',
        'view_azimuth_degrees':a.view_azimuth,'view_changes_only_external_recording_camera':True,
        'robot_policy_camera_changed':False,'source_config':str(a.config.resolve()),
        'result_success':bool(result.get('success')),
        'result_sha256':hashlib.sha256((a.output/'result.json').read_bytes()).hexdigest()}
    (a.output/'recording_manifest.json').write_text(json.dumps(descriptor,indent=2))


if __name__=='__main__':main()
