#!/usr/bin/env python3
"""Truth segmentation for offline camera QA only; never a policy observation."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new visibility audit')
    rows=[]
    for entry in json.load(open(a.manifest))['accepted']:
        folder=Path(entry['path']);meta=json.load(open(folder/'episode.json'));raw=np.load(folder/'replay.npz')
        frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
        model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));data=mujoco.MjData(model)
        target=model.body('object').id;geoms=np.flatnonzero(model.geom_bodyid==target)
        counts=[]
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            renderer.enable_segmentation_rendering()
            for index in (24,36,48,60,78):
                if index>=len(frames):continue
                row=frames[index];data.qpos[:]=raw['qpos'][row['physics_index']]
                data.time=row['timestamp'];mujoco.mj_forward(model,data)
                result={'frame':index,'timestamp':row['timestamp']}
                for label,camera in [('front','d435i_rgbd'),('wrist','wrist_rgb')]:
                    renderer.update_scene(data,camera=camera);mask=renderer.render()
                    result[label+'_object_pixels']=int(np.count_nonzero(np.isin(mask[:,:,0],geoms)&
                        (mask[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))))
                counts.append(result)
        rows.append({'path':str(folder),'skill':meta['skill'],'seed':meta['seed'],'views':counts})
    report={'scope':'offline truth-ID visibility instrument only; segmentation is not exported to model inputs',
        'episodes':len(rows),'sampled_frames_per_episode':5,'results':rows,
        'episodes_with_visible_wrist_target':sum(any(x['wrist_object_pixels']>=20 for x in r['views']) for r in rows),
        'training_performed':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':main()
