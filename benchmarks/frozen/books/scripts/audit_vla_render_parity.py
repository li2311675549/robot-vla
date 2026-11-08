#!/usr/bin/env python3
"""Offline render reconstruction from recorded states, never a policy rollout.

Object IDs are used only to quantify visibility in QA masks, never as VLA input.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image
import _bootstrap
from robot_exam.perception.rgbd import RGBDCamera


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',default=[0,35])
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior render checks')
    a.output.mkdir(parents=True)
    sys.path.insert(0,'/home/li/vla/openpi/packages/openpi-client/src')
    from openpi_client.image_tools import resize_with_pad
    manifest=json.loads(a.manifest.read_text());rows=[]
    for skill in ('apple','banana','orange'):
        entry=next(e for e in manifest['accepted'] if e['split']=='train' and e['skill']==skill)
        folder=Path(entry['path']);frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
        data=mujoco.MjData(model);replay=np.load(folder/'replay.npz')
        mujoco.mj_setState(model,data,replay['initial_state'],int(replay['state_spec']))
        recorded_qpos=replay['qpos']
        # Independent scratch-state reconstruction of already recorded images.
        # No actions, dynamics-based success claims or existing records change.
        sim=SimpleNamespace(model=model,data=data)
        geoms=np.flatnonzero(model.geom_bodyid==model.body('object').id)
        for name,cam in [('front',RGBDCamera(sim)),('wrist',RGBDCamera(sim,camera_name='wrist_rgb'))]:
            try:
                from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER
                for index in a.frames:
                    frame=frames[index];data.qpos[:]=recorded_qpos[frame['physics_index']]
                    mujoco.mj_forward(model,data)
                    rendered=cam.capture_rgb()
                    gl={'vendor':glGetString(GL_VENDOR).decode(),'renderer':glGetString(GL_RENDERER).decode()}
                    original=np.asarray(Image.open(folder/frame['rgb' if name=='front' else 'rgb_wrist']))
                    cam.renderer.enable_segmentation_rendering();cam.renderer.update_scene(data,camera=cam.id)
                    segmentation=cam.renderer.render().copy();cam.renderer.disable_segmentation_rendering()
                    mask=np.isin(segmentation[:,:,0],geoms)&(segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                    small=resize_with_pad(rendered,224,224);reference=resize_with_pad(original,224,224)
                    small_mask=resize_with_pad(np.repeat((mask*255).astype(np.uint8)[:,:,None],3,axis=2),224,224)[:,:,0]
                    stem=f'{skill}_frame{index:03d}_{name}'
                    Image.fromarray(small).save(a.output/f'{stem}_rendered.png')
                    Image.fromarray(reference).save(a.output/f'{stem}_training.png')
                    Image.fromarray(small_mask).save(a.output/f'{stem}_qa_object_mask.png')
                    difference=np.abs(small.astype(float)-reference.astype(float))
                    rows.append({'skill':skill,'source_seed':entry['seed'],'frame':index,'camera':name,
                        'opengl':gl,'raw_pixel_mean_absolute_error':float(np.abs(rendered.astype(float)-original.astype(float)).mean()),
                        'model224_pixel_mean_absolute_error':float(difference.mean()),'model224_pixel_max_error':float(difference.max()),
                        'object_pixels_at224_core':int(np.count_nonzero(small_mask>127)),
                        'object_pixels_at224_any_support':int(np.count_nonzero(small_mask)),
                        'note':'Segmentation uses truth only in offline visibility QA, not model observations.'})
            finally:cam.close()
        replay.close()
    (a.output/'result.json').write_text(json.dumps({'scope':'offline_recorded_state_render_QA_not_new_physical_rollout',
        'frames_per_camera':a.frames,'samples':rows,'training_updates':0,'model_loaded':False},indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
