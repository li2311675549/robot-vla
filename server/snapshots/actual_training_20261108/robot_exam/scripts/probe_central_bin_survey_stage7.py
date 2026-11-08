#!/usr/bin/env python3
"""Physical one-station RGB-D label survey from a saved task checkpoint."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.perception.objects import observe_bins
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import State,TaskRuntime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',default='0002.npz')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((source/'scene.json').read_text())
    sim=Simulation(source/'scene.xml')
    checkpoint=np.load(source/'checkpoints'/args.checkpoint)
    mujoco.mj_setState(sim.model,sim.data,checkpoint['integration_state'],int(checkpoint['state_spec']))
    row=min(json.loads((source/'trajectory.json').read_text()),
            key=lambda x:abs(x['time']-float(checkpoint['time'])))
    sim.data.ctrl[:]=row['ctrl'];mujoco.mj_forward(sim.model,sim.data)
    runtime=TaskRuntime(sim,output=output,wall_limit=900,sim_limit=3000)
    runtime.config=json.loads((source/'config.json').read_text())
    runtime.qgoal=sim.data.qpos[runtime.arm.qadr].copy()
    runtime.arm.reference=runtime.qgoal.copy()
    grid=OccupancyGrid();runtime.grid=grid
    for box in metadata['obstacles']:grid.rectangle(*box)
    result={'source':str(source),'checkpoint':args.checkpoint,'success':False}
    start=float(sim.data.time)
    try:
        runtime.transition(State.NAVIGATE,'central_bin_label_survey',timeout=240)
        runtime.stow();runtime.backout(-.40)
        runtime.navigate([-.40,1.35,.25],grid)
        runtime.dock([0.,1.35,.25])
        camera=RGBDCamera(sim)
        try:
            rgb,depth=camera.capture();seen=observe_bins(camera,rgb,depth)
            camera.save(output/'observation',rgb,depth)
        finally:camera.close()
        result['observed_bins']=seen
        result['success']=len(seen)==3 and all(x['pixels']>=200 for x in seen.values())
    except Exception as exc:
        result['failure']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        result['elapsed_sim_seconds']=float(sim.data.time-start)
        result['base_pose']=runtime.base.pose().tolist()
        result['warnings']=sim.data.warning.number.tolist()
        result['collision_events']=len(runtime.collision_events)
        runtime.save()
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
