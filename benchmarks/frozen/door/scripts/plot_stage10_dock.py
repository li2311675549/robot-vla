#!/usr/bin/env python3
"""Export measured dock errors, motion, wheel commands and trajectory to PNG."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use new plot filename')
    fig,axes=plt.subplots(4,2,figsize=(14,13));axes=axes.ravel();mode_names=[]
    for strategy,color in [('hysteresis','#b84a4a'),('preapproach','#247f8c')]:
        episode=json.loads((a.case/strategy/'dock.json').read_text())[-1];rows=episode['samples']
        t=np.asarray([x['time']-rows[0]['time'] for x in rows]);xy=np.asarray([x['pose'][:2] for x in rows])
        for ax,key in zip(axes[:4],('position_error_m','heading_error_rad','linear_speed_m_s','yaw_rate_rad_s')):
            ax.plot(t,[abs(x[key]) for x in rows],label=strategy,color=color)
        for ax,key in zip(axes[4:6],('wheel_target_rad_s','wheel_velocity_rad_s')):
            wheels=np.asarray([x[key] for x in rows])
            ax.plot(t,wheels[:,[0,2]].mean(axis=1),color=color,label=strategy+' left')
            ax.plot(t,wheels[:,[1,3]].mean(axis=1),color=color,ls=':',label=strategy+' right')
        axes[6].plot(xy[:,0],xy[:,1],color=color,label=strategy)
        for plan in episode.get('plans',[]):
            path=np.asarray(plan['poses']);axes[6].plot(path[:,0],path[:,1],color=color,ls='--',alpha=.5)
        modes=[]
        for row in rows:
            if row['mode'] not in mode_names:mode_names.append(row['mode'])
            modes.append(mode_names.index(row['mode']))
        axes[7].step(t,modes,color=color,label=strategy)
    for ax,value in zip(axes[:4],(.023,.035,.02,.03)):ax.axhline(value,color='black',ls='--',lw=.8)
    labels=['position error (m)','heading error (rad)','linear speed (m/s)','yaw speed (rad/s)',
            'wheel target (rad/s)','actual wheel speed (rad/s)','world y (m)','control mode']
    for i,(ax,label) in enumerate(zip(axes,labels)):
        ax.set_ylabel(label);ax.set_xlabel('world x (m)' if i==6 else 'simulated seconds')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[6].axis('equal');axes[7].set_yticks(range(len(mode_names)),mode_names)
    fig.suptitle('Same physical fixture: '+str(a.case));fig.tight_layout()
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=140);plt.close(fig)


if __name__=='__main__':main()
