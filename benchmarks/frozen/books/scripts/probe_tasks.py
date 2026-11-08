#!/usr/bin/env python3
import json
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.navigation.grid import OccupancyGrid


def main():
    output=ROOT/'outputs/stage34/probes/navigation'
    s=Simulation(ROOT/'scenes/fruit.xml');r=TaskRuntime(s,output=output)
    meta=json.loads((ROOT/'scenes/fruit.json').read_text())
    grid=OccupancyGrid()
    for obstacle in meta['obstacles']:grid.rectangle(*obstacle)
    r.hold(.6);r.transition(State.NAVIGATE,'five_metre_approach')
    try:
        print(r.navigate(meta['staging'],grid))
        r.dock(meta['dock']);print('dock',r.base.pose(),'collisions',r.collision_events)
    finally:r.save()


if __name__=='__main__':main()
