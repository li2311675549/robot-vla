#!/usr/bin/env python3
import argparse,json
from _bootstrap import ROOT
from robot_exam.data.episodes import validate_episode
p=argparse.ArgumentParser();p.add_argument('episode');p.add_argument('--replay',action='store_true');a=p.parse_args()
r=validate_episode(a.episode,a.replay);print(json.dumps(r,indent=2))
if r['status']!='PASS':raise SystemExit(1)
