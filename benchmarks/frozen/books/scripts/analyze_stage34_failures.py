#!/usr/bin/env python3
"""Derived failure taxonomy; original per-episode results remain untouched."""
import bisect
from collections import Counter
import json
from pathlib import Path
from _bootstrap import ROOT
import numpy as np
OUT=ROOT/'outputs/stage34'


def classify(r):
    cat=r.get('failure_category');message=r.get('error','')
    if r.get('success'):return None
    if cat=='perception':return 'PERCEPTION'
    if cat=='ik_unreachable' or 'IK failed' in message:return 'IK_UNREACHABLE'
    if cat=='gripper_slip':return 'GRIPPER_SLIP'
    if cat in ('path_planning','navigation_timeout','navigation_accuracy'):return 'PATH_PLANNING_OR_TRACKING'
    if cat=='state_timeout':return 'STATE_TIMEOUT'
    if cat in ('task_timeout','process_timeout_or_crash'):return 'TASK_TIMEOUT_OR_PROCESS'
    if r.get('collision_events'):return 'MOTION_COLLISION'
    if cat=='arm_tracking':return 'MOTION_TRACKING_OR_CONTACT_BLOCK'
    if cat in ('handle_grasp','grasp_contact','door_stuck','drawer_stuck'):return 'CONTACT_OPERATION'
    if r['task']=='door' and r.get('failure_qpos') and abs(r['failure_qpos'][-1])>=.035:return 'CONTACT_OPERATION_FINAL_DOOR_OPEN'
    return 'TASK_PREDICATE_OR_EXCEPTION'


def main():
    records=[];errors=[]
    roots={'fruit':'formal_fruit_v1','door':'formal_door_v1','drawer':'formal_v1'}
    for task,folder in roots.items():
        for path in sorted((OUT/folder/task).glob('*/result.json')):
            r=json.loads(path.read_text());fsm=json.loads((path.parent/'fsm.json').read_text());trajectory=json.loads((path.parent/'trajectory.json').read_text())
            record={'task':task,'seed':r['seed'],'success':r['success'],'category':classify(r),'original_category':r.get('failure_category'),
                    'time':r['sim_seconds'],'state':r.get('failure_stage'),'recoveries':r.get('num_retries',0),
                    'error':r.get('error'),'last_state_reasons':[t['reason'] for t in fsm[-3:]],
                    'robot_before_and_after_terminal_event':[row['base'] for row in trajectory[-2:]],
                    'collision_events':len(r.get('collision_events',[])),'remaining_target_states':r.get('failure_objects'),
                    'strict_score':r.get('score'),'log':str(path.relative_to(ROOT))}
            if task=='door':record['final_door_angle_rad']=r.get('final_door_angle',r.get('failure_qpos',[None])[-1])
            records.append(record)
            if task=='fruit':
                # Independent diagnostic association, never fed back to policy.
                times=[row['time'] for row in trajectory]
                for transition in fsm:
                    if transition['reason']!='select_visible_fruit':continue
                    d=transition['target'];k=max(0,bisect.bisect_right(times,transition['time'])-1)
                    objects=trajectory[k].get('dynamic_objects',{})
                    if not objects:continue
                    point=np.asarray(d['center']);distance=min(np.linalg.norm(point-p) for p in objects.values())
                    errors.append({'seed':r['seed'],'time':transition['time'],'category':d['category'],'nearest_body_origin_error_m':float(distance)})
    by_task={}
    for task in roots:
        group=[r for r in records if r['task']==task]
        by_task[task]={'episodes':len(group),'successes':sum(r['success'] for r in group),
                       'failures_by_category':dict(Counter(r['category'] for r in group if not r['success'])),'collision_events':sum(r['collision_events'] for r in group)}
    result={'scope':'post-freeze descriptive analysis; no parameters changed from held-out outcomes',
            'by_task':by_task,'episodes':records,'fruit_selected_target_errors':errors,
            'fruit_selected_target_error_mean_m':float(np.mean([e['nearest_body_origin_error_m'] for e in errors])) if errors else None,
            'causal_ablation_performed':False}
    (OUT/'failure_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(by_task,indent=2))


if __name__=='__main__':main()
