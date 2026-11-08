"""Optional compact carry experiment, not an enabled production fix.

The recorded same-state negative control did not fix spontaneous fruit drift;
passive scene tests instead identified the initial resting-pose issue.
"""
import json

import numpy as np

from robot_exam.control.book_path import plan_book_lowering
from robot_exam.tasks.runtime import State, TaskFailure


def prepare_compact_carry(runtime,body,rotation,minimum_payload_z):
    """Normal joint/position actuators only; this motion is not VLA supervision.

    Reuse the existing constant-orientation held-object path planner. The
    geometric attachment is used exclusively in scratch collision queries;
    every live step still depends on real pad contact and the original guard.
    """
    r=runtime;pose=r.base.pose();c,s=np.cos(pose[2]),np.sin(pose[2])
    rotation=np.asarray(rotation);plans=[];chosen=None
    r.transition(State.MANIPULATE,'unrecorded_parked_compact_carry_preparation',target=body,timeout=90.)
    r._ensure_operation_hold()
    for x,height in ((.22,.96),(.26,.96),(.22,1.02)):
        position=np.array([pose[0]+c*x,pose[1]+s*x,height])
        row={'target':position.tolist(),'feasible':False}
        try:
            path,report=plan_book_lowering(r.arm,position,rotation,body)
            row.update(feasible=True,planning=report);chosen=(path,position);plans.append(row);break
        except RuntimeError as exc:row['error']=str(exc)
        plans.append(row)
    result={'body':body,'scope':'traditional parked carry preparation; excluded from7D VLA phase supervision',
            'candidates':plans,'success':False}
    try:
        if chosen is None:raise TaskFailure('compact_carry_path','No orientation/collision-safe compact fruit carry path')
        r.begin_payload(body,rotation,world_stabilize=False)
        try:
            for q in chosen[0]:r.joint_move(q,seconds=.5)
            error=float(np.linalg.norm(r.arm.pose()[0]-chosen[1]))
            held=r.sim.data.xpos[r.sim.model.body(body).id,2]>minimum_payload_z
            stable=r.stable_grasp(body,seconds=.5)
            if error>.018 or not held or not stable:
                raise TaskFailure('gripper_slip','Compact carry endpoint/contact/height did not physically validate')
            result.update(success=True,actual_tool=r.arm.pose()[0].tolist(),endpoint_error_m=error)
        finally:r.end_payload()
        return result
    finally:
        if r.output:
            with (r.output/'compact_carry.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
