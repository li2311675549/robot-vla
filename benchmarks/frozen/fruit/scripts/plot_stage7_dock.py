#!/usr/bin/env python3
"""Plot paired dock traces. Run with the existing openpi Python/matplotlib."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def samples(path):
    episodes=json.loads(path.read_text())
    trace=episodes[-1]
    if trace.get('controller')=='hysteresis_v1':
        rows=trace['samples'];kind='hysteresis';target='wheel_target_rad_s';wheel='wheel_velocity_rad_s'
        base_t=rows[0]['time']
    else:
        rows=trace.get('command_trace',[]);kind='legacy';target='wheel_target';wheel='wheel_velocity_rad_s'
        # Stage 6 stores only the final 100 coarse samples on a timeout.
        base_t=rows[-1]['time']-trace['elapsed_seconds'] if rows else 0.
    return kind,np.asarray([x['time']-base_t for x in rows]),rows,target,wheel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('pair',type=Path,help='Case folder containing legacy/ and hysteresis/')
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(4,1,figsize=(11,11),sharex=True)
    for folder,color in [('legacy','#b54b4b'),('hysteresis','#177d8d')]:
        kind,t,rows,target,wheel=samples(args.pair/folder/'dock.json')
        if not len(rows):continue
        p=np.asarray([x['pose'] for x in rows]);q=np.asarray([x[target] for x in rows]);v=np.asarray([x[wheel] for x in rows])
        axes[0].plot(t,[x['position_error_m'] for x in rows],label=kind,color=color)
        axes[1].plot(t,[x['heading_error_rad'] for x in rows],label=kind,color=color)
        axes[2].plot(t,np.mean(q[:,[0,2]],axis=1),label=kind+' left target',color=color)
        axes[2].plot(t,np.mean(q[:,[1,3]],axis=1),label=kind+' right target',color=color,linestyle=':')
        axes[3].plot(t,np.mean(v[:,[0,2]],axis=1),label=kind+' left actual',color=color)
        axes[3].plot(t,np.mean(v[:,[1,3]],axis=1),label=kind+' right actual',color=color,linestyle=':')
        if kind=='hysteresis':
            for switch in json.loads((args.pair/folder/'dock.json').read_text())[-1]['mode_switches']:
                if switch['to'] in ('heading','settle'):
                    axes[1].axvline(switch['time']-rows[0]['time'],color=color,alpha=.18,linewidth=.7)
    axes[0].axhline(.023,color='black',linestyle='--',linewidth=.8)
    axes[1].axhline(.035,color='black',linestyle='--',linewidth=.8)
    for ax,label in zip(axes,['position error (m)','heading error (rad)',
                              'commanded wheel speed (rad/s)','measured wheel speed (rad/s)']):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel('simulated seconds since dock start')
    fig.suptitle(args.pair.as_posix()+' — same saved integration/controller state',fontsize=11)
    fig.tight_layout();fig.savefig(args.output,dpi=150);plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
