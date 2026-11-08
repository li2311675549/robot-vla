#!/usr/bin/env python3
"""Read-only paired camera rendering of a saved real bin-survey checkpoint."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v3 as iio
import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.perception.objects import observe_bins
from robot_exam.perception.bin_labels import upright_label_textures


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--checkpoint',default='0003.npz');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior previews')
    a.output.mkdir(parents=True)
    scene=a.episode/'scene.xml';tree=ET.parse(scene);root=tree.getroot()
    for e in root.findall('.//mesh')+root.findall('.//texture'):
        if e.get('file'):e.set('file',str((scene.parent/e.get('file')).resolve()))
    changes=upright_label_textures(root);new=a.output/'scene.xml';tree.write(new,encoding='unicode')
    saved=np.load(a.episode/'checkpoints'/a.checkpoint);sims=[Simulation(scene),Simulation(new)]
    measurements={}
    for label,sim in zip(('old','upright_v2'),sims):
        mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
        camera=RGBDCamera(sim)
        try:
            rgb,depth=camera.capture();iio.imwrite(a.output/(label+'.png'),rgb)
            measurements[label]=observe_bins(camera,rgb,depth)
        finally:camera.close()
    fields=('qpos0','body_mass','body_inertia','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','geom_friction','actuator_gainprm','actuator_forcerange')
    for field in fields:np.testing.assert_array_equal(getattr(sims[0].model,field),getattr(sims[1].model,field))
    np.testing.assert_array_equal(sims[0].data.qpos,sims[1].data.qpos)
    result=dict(scope='Paired render of actual saved state; texture-only fix, not a new physical successful task',changes=changes,physical_arrays_identical=list(fields),live_process_modified=False,bin_observations=measurements)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
