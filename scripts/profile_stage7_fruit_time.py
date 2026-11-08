#!/usr/bin/env python3
"""Summarize measured FSM intervals from a saved fruit episode."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('episode',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    events=json.loads((args.episode/'fsm.json').read_text())
    result=json.loads((args.episode/'result.json').read_text())
    durations=defaultdict(float)
    terminal=float(result['sim_seconds'])
    for current,next_event in zip(events,events[1:]+[{'time':terminal}]):
        durations[current['reason']]+=max(0.,float(next_event['time'])-float(current['time']))
    summary={'episode':str(args.episode.resolve()),'status':result['status'],
             'sim_seconds':terminal,'wall_seconds':result['wall_seconds'],
             'reason_seconds':dict(sorted(durations.items(),key=lambda kv:-kv[1])),
             'last_transition':events[-1] if events else None}
    encoded=json.dumps(summary,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded+'\n')
    print(encoded)


if __name__=='__main__':main()
