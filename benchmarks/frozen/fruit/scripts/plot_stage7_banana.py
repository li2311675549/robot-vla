#!/usr/bin/env python3
"""Plot paired contact evidence from the same saved physical banana grasp."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('case',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(4,1,figsize=(11,12),sharex=True)
    for label,folder,color in [('full',args.case/'full','tab:red'),
                               ('regulated',args.case/'regulated','tab:blue')]:
        samples=json.loads((folder/'payload_traces.json').read_text())[0]['samples']
        t=np.array([s['time']-samples[0]['time'] for s in samples])
        drift=np.array([s['relative_error_m'] for s in samples])
        axes[0].plot(t,drift*1000,label=label,color=color)
        axes[1].plot(t,[s['left_normal_force_n'] for s in samples],label=label+' left',color=color)
        axes[1].plot(t,[s['right_normal_force_n'] for s in samples],label=label+' right',color=color,ls='--')
        axes[2].plot(t,[max((c['friction_utilization'] for c in s['finger_contacts']),default=0.)
                        for s in samples],label=label,color=color)
        axes[3].plot(t,[s['base_pose'][0] for s in samples],label=label,color=color)
        axes[3].plot(t,[s['gripper_closure'] for s in samples],color=color,ls=':',alpha=.65)
    axes[0].axhline(25,color='gray',ls=':',label='online drift stop')
    for ax,title in zip(axes,['Fruit/tool relative displacement (mm)','Finger normal force (N)',
                              'Maximum tangential/normal force ratio','Base x (m); dotted = closure command']):
        ax.set_ylabel(title);ax.legend(loc='best');ax.grid(alpha=.25)
    axes[-1].set_xlabel('Simulated seconds after payload start')
    fig.suptitle('Seed 420: identical physical grasp state, different hold commands')
    fig.tight_layout();fig.savefig(args.output,dpi=140)
    print(args.output)


if __name__=='__main__':main()
