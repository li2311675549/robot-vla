#!/usr/bin/env python3
"""Scientific plots from saved evidence, using the existing openpi environment."""
import json,os
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
OUT=ROOT/'outputs/stage34';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)


def main():
    summary=json.loads((OUT/'acceptance/navigation/summary.json').read_text())
    fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
    for ax,case in zip(axes.ravel(),summary['results']):
        folder=OUT/'acceptance/navigation'/case['case'];rows=json.loads((folder/'trajectory.json').read_text());xy=np.array([r['base'][:2] for r in rows])
        for geom in ET.parse(folder/'scene.xml').findall('worldbody/geom'):
            if not (geom.get('name') or '').startswith('obstacle'):continue
            x,y,_=map(float,geom.get('pos').split());sx,sy,_=map(float,geom.get('size').split());ax.add_patch(Rectangle((x-sx,y-sy),2*sx,2*sy,fc='gray',alpha=.5))
        for leg in case['legs']:
            path=np.asarray(leg['path']);ax.plot(path[:,0],path[:,1],'--',c='#d55e00',lw=1.5,label='A* path')
        ax.plot(xy[:,0],xy[:,1],c='#0072b2',lw=1,label='wheel-driven');ax.scatter(*xy[0],c='black',s=15);ax.scatter(*xy[-1],c='#009e73',marker='*',s=60)
        ax.set_title(case['case']);ax.set_box_aspect(1);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    fig.suptitle('Physical navigation: eight independent cases (robot radius 0.64 m)')
    fig.savefig(FIG/'navigation.png',dpi=160);fig.savefig(FIG/'navigation.svg');plt.close(fig)
    perception=json.loads((OUT/'perception/summary.json').read_text());groups=[]
    for task in ['fruit','drawer']:groups.append([1000*m['center_error_m'] for r in perception['records'] if r['task']==task for m in r['matches']])
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True);ax.boxplot(groups,tick_labels=['Fruit: 48 visible / 50','Books: 15 / 15'],showmeans=True);ax.set_ylabel('RGB-D centre error [mm]');ax.set_title('Parked development seeds 0-4; truth used only for scoring');ax.grid(axis='y',alpha=.2);fig.savefig(FIG/'perception_error.png',dpi=160);plt.close(fig)
    roots={'fruit':'formal_fruit_v1','door':'formal_door_v1','drawer':'formal_v1'}
    if all((OUT/folder/'summary.json').exists() for folder in roots.values()):
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        rates=[]
        for index,(task,folder) in enumerate(roots.items()):
            info=json.loads((OUT/folder/'summary.json').read_text())[task];rates.append(info['complete_success_rate']*100)
            records=json.loads((OUT/folder/'episodes.json').read_text())
            for r in records:axes[1].scatter(index+(r['seed']%100-5.5)*.015,r['sim_seconds'],marker='o' if r['success'] else 'x',c='#009e73' if r['success'] else '#d55e00',s=30)
        axes[0].bar(list(roots),rates,color=['#d55e00','#d55e00','#0072b2']);axes[0].set_ylim(0,110);axes[0].set_ylabel('Complete-task success [%]');axes[0].set_title('10 held-out attempts per task')
        for i,rate in enumerate(rates):axes[0].text(i,rate+2,f'{rate:.0f}% ({round(rate/10)}/10)',ha='center')
        axes[1].set_xticks(range(3),list(roots));axes[1].set_ylabel('Simulated termination time [s]');axes[1].set_title('Each attempt: green complete / orange failed');axes[1].grid(axis='y',alpha=.2)
        fig.savefig(FIG/'formal_evaluation.png',dpi=160);fig.savefig(FIG/'formal_evaluation.svg');plt.close(fig)
    print('Saved evidence plots to',FIG)


if __name__=='__main__':main()
