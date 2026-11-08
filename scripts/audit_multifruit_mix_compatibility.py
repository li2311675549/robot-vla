#!/usr/bin/env python3
"""Offline old/new action and camera-contract audit; never moves a live robot."""
import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT


def minimum_depth(model,data,geom,axis,origin):
    local=data.geom_xmat[geom].reshape(3,3).T@axis
    size=model.geom_size[geom];kind=model.geom_type[geom]
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:radius=size[0]
    elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:radius=np.linalg.norm(local*size)
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:radius=np.abs(local)@size
    elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:radius=size[0]+size[1]*abs(local[2])
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:radius=size[0]*np.linalg.norm(local[:2])+size[1]*abs(local[2])
    else:radius=model.geom_rbound[geom]
    return float((data.geom_xpos[geom]-origin)@axis-radius)


def inspect(rows,old):
    summary=[];fields={k:set() for k in ['semantics','sensor','fps','dt','mount']}
    for row in rows:
        folder=Path(row['path']);meta=json.loads((folder/'episode.json').read_text())
        frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
        model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));data=mujoco.MjData(model)
        replay=np.load(folder/'replay.npz');qpos=replay['qpos']
        body=model.body('object' if old else meta['target_body']).id
        geoms=np.flatnonzero(model.geom_bodyid==body)
        cam=model.camera('wrist_rgb').id;K=np.asarray(meta['wrist_camera_K'])
        near=float(model.stat.extent*model.vis.map.znear);clipped=[];visible=0
        for frame in frames:
            # Scratch inverse-kinematics analysis only, no task execution.
            data.qpos[:]=qpos[int(frame['physics_index'])];mujoco.mj_forward(model,data)
            rotation=data.cam_xmat[cam].reshape(3,3);position=data.cam_xpos[cam]
            local=rotation.T@(data.xpos[body]-position);depth=-float(local[2])
            if depth<=0:continue
            u=K[0,0]*local[0]/depth+K[0,2];v=-K[1,1]*local[1]/depth+K[1,2]
            if not(0<=u<640 and 0<=v<480):continue
            visible+=1
            lower=min(minimum_depth(model,data,int(g),-rotation[:,2],position) for g in geoms)
            if lower<near:clipped.append({'frame':frame['frame_index'],'depth_m':depth,'nearest_surface_m':lower})
        fields['semantics'].add(meta['action_semantics']);fields['sensor'].add(meta['sensor_schema'])
        fields['fps'].add(meta['frame_rate']);fields['dt'].add(meta['physics_timestep'])
        fields['mount'].add(tuple(meta['wrist_mount_position_tool_m']))
        states=np.asarray([x['observation.state'] for x in frames]);actions=np.asarray([x['action'] for x in frames])
        if states.shape!=(len(frames),7) or actions.shape!=(len(frames),7) or not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError('Invalid state/action contract')
        if (actions[:,6]<0).any() or (actions[:,6]>1).any():raise ValueError('Illegal gripper command')
        summary.append({'path':str(folder),'seed':row['seed'],'split':row['split'],'category':row['skill'],
                        'frames':len(frames),'camera_near_m':near,'visible_centre_frames':visible,
                        'near_clipped_surface_frames':len(clipped),'first_clipped':clipped[:1]})
    return {'fields':{k:[list(v) if isinstance(v,tuple) else v for v in sorted(x)] for k,x in fields.items()},
            'episodes':len(rows),'frames':sum(x['frames'] for x in summary),
            'near_clipped_surface_frames':sum(x['near_clipped_surface_frames'] for x in summary),
            'episodes_with_clipped_surface':sum(bool(x['near_clipped_surface_frames']) for x in summary),'rows':summary}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-manifest',type=Path,default=ROOT/'outputs/stage11/dual_expansion_release_v1/manifest/manifest.json')
    p.add_argument('--new-manifest',type=Path,default=ROOT/'outputs/stage20/multifruit_pilot_manifest_v4/manifest.json')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior camera/data audits')
    old=json.loads(a.old_manifest.read_text())['accepted'];new=json.loads(a.new_manifest.read_text())['accepted']
    report={'scope':'offline recorded-pose geometry and action contract, not physical replay or learned-policy evaluation',
            'old':inspect(old,True),'new':inspect(new,False),'training_started':False,'mixing_released':False}
    report['common_non_render_contract']=report['old']['fields']==report['new']['fields']
    report['clipping_note']='Exact support for primitive fruit collisions, conservative for meshes; centre-in-FOV gating. This is not pixel segmentation or a proof of VLA failure causality.'
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('old','new')}),flush=True)
    print('old',report['old']['episodes'],report['old']['episodes_with_clipped_surface'],report['old']['near_clipped_surface_frames'],flush=True)
    print('new',report['new']['episodes'],report['new']['episodes_with_clipped_surface'],report['new']['near_clipped_surface_frames'],flush=True)


if __name__=='__main__':main()
