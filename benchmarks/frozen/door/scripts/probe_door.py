#!/usr/bin/env python3
import json
import xml.etree.ElementTree as E
from pathlib import Path
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.door import open_segment


def main():
    tree=E.parse(ROOT/'scenes/door.xml');key=tree.find('keyframe/key');q=key.get('qpos').split()
    q[0]='-.905';q[1]='.42'
    import math
    q[3]=str(math.cos(.59/2));q[6]=str(math.sin(.59/2));key.set('qpos',' '.join(q))
    for e in tree.findall('.//mesh'):
        if e.get('file'):e.set('file',str((ROOT/'scenes'/e.get('file')).resolve()))
    out=ROOT/'outputs/stage34/probes/door';out.mkdir(parents=True,exist_ok=True);tree.write(out/'park.xml')
    s=Simulation(out/'park.xml');r=TaskRuntime(s,output=out);meta=json.loads((ROOT/'scenes/door.json').read_text())
    try:
        r.hold(.6);print('open',open_segment(r,meta,.10))
    except Exception as e:print(type(e).__name__,str(e),'time',s.data.time,'tip',r.arm.pose()[0])
    finally:r.save()


if __name__=='__main__':main()
