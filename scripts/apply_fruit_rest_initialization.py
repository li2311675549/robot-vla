#!/usr/bin/env python3
"""Create a new scene version; preserve the original failed scene unchanged."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from _bootstrap import ROOT
from robot_exam.data.rest_initialization import apply_rest_initialization


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve scene versions')
    a.output.mkdir(parents=True);root=ET.parse(a.source).getroot()
    for node in root.iter():
        if 'file' in node.attrib:
            path=Path(node.attrib['file'])
            if not path.is_absolute():node.set('file',str((a.source.parent/path).resolve()))
    scene=a.output/'scene.xml';ET.ElementTree(root).write(scene)
    metadata=json.loads(a.source.with_suffix('.json').read_text())
    apply_rest_initialization(scene,metadata,a.calibration)
    print(json.dumps({'status':'STATIC_SCENE_INITIALIZED_NOT_YET_PASSIVELY_VALIDATED','scene':str(scene),
                      'protocol':metadata['scene_protocol'],'runtime_pose_writes':False}),flush=True)
