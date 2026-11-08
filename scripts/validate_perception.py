#!/usr/bin/env python3
"""RGB-D error audit on development seeds; truth is used only for scoring."""
import json
from pathlib import Path
from _bootstrap import ROOT
from build_tasks import build
import mujoco,numpy as np
from robot_exam.simulation import Simulation
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.perception.objects import observe_objects


def main():
    out=ROOT/'outputs/stage34/perception';out.mkdir(parents=True,exist_ok=True);records=[]
    for task in ['fruit','drawer']:
        for seed in range(5):
            folder=out/f'{task}_{seed}';folder.mkdir(exist_ok=True);path,meta=build(task,seed,folder/'scene.xml');s=Simulation(path)
            s.data.qpos[:2]=(meta['dock'] if task=='fruit' else meta['book_dock'])[:2];mujoco.mj_forward(s.model,s.data);s.step(s.data.ctrl,300)
            camera=RGBDCamera(s)
            try:
                rgb,depth=camera.capture()
                detected=observe_objects(camera,rgb,depth,**({'kind':'book','workspace':(.34,.64,-1.,-.30)} if task=='drawer' else {}))
                if seed==0:camera.save(folder/'camera',rgb,depth)
            finally:camera.close()
            remaining=list(meta['objects']);matches=[]
            for d in detected:
                if not remaining:break
                nearest=min(remaining,key=lambda o:np.linalg.norm(s.data.xpos[s.model.body(o['body']).id]-d['center']));remaining.remove(nearest)
                truth=s.data.xpos[s.model.body(nearest['body']).id];label=nearest.get('category') or ['red','green','blue'][nearest['rank']]
                matches.append({'body':nearest['body'],'predicted':d['category'],'correct_class':label==d['category'],
                                'center_error_m':float(np.linalg.norm(truth-d['center'])),'area':d['area'],'true_area':nearest.get('area')})
            sorted_correct=(len(matches)==3 and [m['body'] for m in sorted(matches,key=lambda m:m['area'])]==['book_0','book_1','book_2']) if task=='drawer' else None
            records.append({'task':task,'seed':seed,'detected':len(detected),'total_objects':len(meta['objects']),'area_order_correct':sorted_correct,'matches':matches})
    errors=[m['center_error_m'] for r in records for m in r['matches']]
    result={'scope':'isolated parked RGB-D observation on development seeds 0-4; not a complete-task success metric',
            'mean_center_error_m':float(np.mean(errors)),'max_center_error_m':max(errors),'records':records}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
