#!/usr/bin/env python3
"""Record success and failure from one previously saved physical grasp."""
import argparse
import json
from pathlib import Path

from benchmark_banana_closure_stage7 import trial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('fixture',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=args.fixture.resolve();root=args.output.resolve()
    info=json.loads((folder/'fixture.json').read_text())
    scene=folder/'scene.xml'
    for label,closure in (('full_failure',1.),('regulated_success',.7)):
        result=trial(scene,folder,info,closure,root/label,record=True)
        print(label,result['transport_success'],result.get('video_error'),flush=True)


if __name__=='__main__':main()
