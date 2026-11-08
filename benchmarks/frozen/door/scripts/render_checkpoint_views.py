#!/usr/bin/env python3
"""Render an actual saved checkpoint from several views; no simulation actions."""
import argparse,hashlib,json
from pathlib import Path
from _bootstrap import ROOT
import mujoco,numpy as np
from PIL import Image,ImageDraw
from robot_exam.simulation import Simulation


def main():
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--reason',default='release');p.add_argument('--minimum-z',type=float,default=.73);p.add_argument('--output',type=Path,default=ROOT/'outputs/stage34/figures/book_checkpoint_views.png');a=p.parse_args()
    fsm=json.loads((a.episode/'fsm.json').read_text())
    index=next(i for i,t in enumerate(fsm) if t['reason']==a.reason and isinstance(t['target'],list) and t['target'][2]>a.minimum_z)
    checkpoint=a.episode/'checkpoints'/f'{index+1:04d}.npz';scene=a.episode/'scene.xml';s=Simulation(scene);data=np.load(checkpoint)
    mujoco.mj_setState(s.model,s.data,data['integration_state'],int(data['state_spec']));mujoco.mj_forward(s.model,s.data)
    canvas=Image.new('RGB',(1280,960));draw=ImageDraw.Draw(canvas)
    with mujoco.Renderer(s.model,height=480,width=640) as renderer:
        for j,azimuth in enumerate([45,135,225,315]):
            camera=mujoco.MjvCamera();camera.lookat[:]=[.25,.25,.65];camera.distance=2.4;camera.azimuth=azimuth;camera.elevation=-25
            renderer.update_scene(s.data,camera=camera);canvas.paste(Image.fromarray(renderer.render().copy()),((j%2)*640,(j//2)*480))
            draw.text(((j%2)*640+10,(j//2)*480+10),f'Actual saved placement checkpoint / azimuth={azimuth}',fill='white')
    a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output)
    a.output.with_suffix('.json').write_text(json.dumps({'scope':'rendered actual saved state; not a new successful task',
        'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'sim_time':s.data.time,'transition':fsm[index]},indent=2))
    print(a.output)


if __name__=='__main__':main()
