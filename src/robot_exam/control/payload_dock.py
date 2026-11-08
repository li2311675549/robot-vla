"""Bounded physical revalidation after a carried-object dock guard trips."""
import json
from robot_exam.control.grasp_recovery import stop_and_revalidate_grasp
from robot_exam.tasks.runtime import State, TaskFailure


def dock_held_object(runtime, body, goal, rotation, minimum_height_m,
                     strategy, timeout=45., maximum_recoveries=1,
                     evidence_name='guarded_dock_attempts.json'):
    """Original dock tolerances and slip guard stay active on every attempt.

    After an actual guard event, stop normally and independently establish a
    new stable lifted grip before reacquiring its measured frame. Never opens
    the gripper, resets an object, or retries a falling/lost object blindly.
    """
    if type(maximum_recoveries) is not int or not 0 <= maximum_recoveries <= 2:
        raise ValueError('At most two bounded grasp recoveries')
    from pathlib import Path
    if Path(evidence_name).name != evidence_name or not evidence_name.endswith('.json'):
        raise ValueError('Evidence must be one unique JSON filename')
    r = runtime
    old = {k:r.config.get(k) for k in ('local_dock_strategy','local_dock_tracking_version','local_dock_preserve_settled_entry')}
    previous_docking = getattr(r, 'payload_docking', False)
    attempts = []
    if strategy:
        r.config.update(local_dock_strategy=strategy, local_dock_tracking_version='axis_bounded_v5')
    r.config['local_dock_preserve_settled_entry']=True
    try:
        for attempt in range(maximum_recoveries+1):
            r.transition(State.NAVIGATE, 'guarded_precision_dock', target=list(goal), timeout=timeout+1)
            r.payload_docking = True
            try:
                r.dock(goal, timeout=timeout, tolerance=.03)
                if not r.stable_grasp(body):
                    raise TaskFailure('gripper_slip', 'Grip lost after dock')
                attempts.append({'attempt':attempt,'dock_success':True})
                return attempts
            except TaskFailure as exc:
                row = {'attempt':attempt,'dock_success':False,'category':exc.category,'error':str(exc)}
                attempts.append(row)
                if exc.category != 'gripper_slip' or attempt == maximum_recoveries:
                    raise
                r.end_payload()
                r.transition(State.RECOVER, 'stopped_grasp_revalidation_without_opening', target=body, timeout=5)
                witness = stop_and_revalidate_grasp(r, body, minimum_height_m)
                row['stopped_grip_witness'] = witness
                if not witness['success']:
                    raise TaskFailure('gripper_slip', 'Stopped grip revalidation rejected the carried object') from exc
                r.retries += 1
                r.begin_payload(body, rotation, world_stabilize=True)
    finally:
        r.payload_docking = previous_docking
        for key, value in old.items():
            if value is None:
                r.config.pop(key, None)
            else:
                r.config[key] = value
        if r.output:
            path = r.output/evidence_name
            # A single collector parent calls this once; reject accidental
            # evidence replacement rather than erase an earlier recovery.
            with path.open('x') as stream:
                json.dump(attempts, stream, indent=2)
