#!/usr/bin/env python3
"""Save actual onboard views of all three category labels."""
import json
from _bootstrap import ROOT
from build_tasks import build
import mujoco
from robot_exam.simulation import Simulation
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.perception.objects import observe_bins
out=ROOT/'outputs/stage34/labels';out.mkdir(parents=True,exist_ok=True)
path,meta=build('fruit',0,out/'scene.xml');results=[]
for expected,station in zip(['apple','banana','orange'],meta['bin_stations']):
 s=Simulation(path);s.data.qpos[:2]=station[:2];mujoco.mj_forward(s.model,s.data);s.step(s.data.ctrl,300)
 camera=RGBDCamera(s)
 try:
  rgb,depth=camera.capture();camera.save(out/expected,rgb,depth);observed=observe_bins(camera,rgb,depth)
 finally:camera.close()
 success=expected in observed and abs(observed[expected]['center'][1]-station[1])<.2
 results.append({'station':station,'expected':expected,'success':success,'observed':observed})
result={'status':'PASS' if all(r['success'] for r in results) else 'FAIL','scope':'onboard color label visibility, not OCR','results':results}
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if result['status']!='PASS':raise SystemExit(1)
