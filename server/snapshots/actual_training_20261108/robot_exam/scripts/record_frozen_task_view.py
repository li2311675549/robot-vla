#!/usr/bin/env python3
"""Isolated historic full-task code execution with recording-only view changes."""
import argparse
import inspect
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--task',choices=['fruit','door'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view-azimuth',type=float,default=145.);a=p.parse_args()
    code=a.code_root.resolve();output=a.output.resolve();output.relative_to(code)
    if output.exists():raise FileExistsError('Never overwrite physical recordings')
    sys.path[:0]=[str(code/'scripts'),str(code/'src')]
    import run_task
    original=run_task.VideoRecorder;graphics={}
    class ViewRecorder(original):
        def __init__(self,path,camera=None,**kwargs):
            camera.azimuth=a.view_azimuth;super().__init__(path,camera=camera,**kwargs)
        def __call__(self,sim,stage):
            super().__call__(sim,stage)
            if not graphics:
                try:
                    from OpenGL import GL
                    graphics.update(vendor=GL.glGetString(GL.GL_VENDOR).decode(),renderer=GL.glGetString(GL.GL_RENDERER).decode())
                except Exception as e:graphics['query_error']=str(e)
    run_task.VideoRecorder=ViewRecorder
    kwargs=dict(record=True,config=a.config)
    if 'door_profile' in inspect.signature(run_task.run).parameters:kwargs['door_profile']='aligned_handles_v3'
    r=run_task.run(a.task,a.seed,'vision' if a.task=='fruit' else 'oracle',output,**kwargs)
    (output/'recording_manifest.json').write_text(json.dumps({'scope':'Fixed historical code physical recording replay; not new formal independent evaluation',
        'code_root':str(code),'frozen_origin':json.loads((code/'frozen_origin.json').read_text()),
        'view_changes_only_external_recording_camera':True,'view_azimuth_degrees':a.view_azimuth,
        'robot_policy_camera_changed':False,'opengl':graphics,'success':r.get('success')},indent=2))


if __name__=='__main__':main()
