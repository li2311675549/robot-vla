"""Orientation-constrained book lowering, planned entirely in scratch MjData."""
from types import SimpleNamespace

import mujoco
import numpy as np

from robot_exam.control.arm import ArmController
from robot_exam.control.joint_rrt import plan_joint_rrt


def rotation_distance(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def held_book_planning_arm(arm, q, rotation, body):
    """Predict a rigid held offset for collision queries, never constrain physics."""
    model, live = arm.sim.model, arm.sim.data
    data = mujoco.MjData(model)
    data.qpos[:] = live.qpos
    data.qvel[:] = live.qvel
    mujoco.mj_forward(model, data)
    planning = ArmController(SimpleNamespace(model=model, data=data))
    body_id = model.body(body).id
    adr = int(model.jnt_qposadr[model.body_jntadr[body_id]])
    tip, old_rotation = arm.pose()
    offset = old_rotation.T @ (live.xpos[body_id] - tip)
    object_rotation = old_rotation.T @ live.xmat[body_id].reshape(3, 3)
    data.qpos[planning.qadr] = q
    mujoco.mj_forward(model, data)
    data.qpos[adr:adr + 3] = planning.pose()[0] + rotation @ offset
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (rotation @ object_rotation).ravel())
    data.qpos[adr + 3:adr + 7] = quat
    mujoco.mj_forward(model, data)
    return planning


def check_release_forearm_clearance(arm, minimum_m=.003):
    """The passive opening chain must clear the forearm, not just closed pads.

    All configurations are scratch predictions. A small positive planning
    margin accounts for tracking error without relaxing physical tolerances.
    """
    model = arm.sim.model
    data = mujoco.MjData(model)
    original = arm.sim.data.qpos.copy()
    addresses = [int(model.jnt_qposadr[i]) for i in range(model.njnt)
                 if (model.joint(i).name or '').startswith('grip_')]
    forearm = model.body('forearm_link').id
    forearm_geoms = np.flatnonzero(model.geom_bodyid == forearm)
    grip_geoms = [i for i in range(model.ngeom)
                  if (model.body(int(model.geom_bodyid[i])).name or '').startswith('grip_')
                  and (model.geom_contype[i] or model.geom_conaffinity[i])]
    minimum = float('inf')
    for alpha in np.linspace(0, 1, 11):
        data.qpos[:] = original
        data.qpos[addresses] *= 1 - alpha
        mujoco.mj_forward(model, data)
        for g1 in forearm_geoms:
            for g2 in grip_geoms:
                distance = mujoco.mj_geomDistance(model, data, int(g1), g2, .02, None)
                minimum = min(minimum, float(distance))
    if minimum < minimum_m:
        raise RuntimeError(f'Open-gripper/forearm release clearance {minimum:.6f}m < planning margin {minimum_m:.6f}m')
    return {'minimum_forearm_clearance_m': minimum, 'planning_margin_m': minimum_m,
            'opening_configurations_checked': 11, 'scratch_only': True}


def plan_book_lowering(arm, position, rotation, body, spacing=.025, maximum_rotation=.12):
    """Keep the held frame near its requested orientation along the whole path.

    Endpoint-only RRT can rotate a book by pi on its way to an equal-orientation
    goal. Reject those paths before execution. The held object's geometric
    offset is propagated only inside the scratch collision predictor; live
    contact and slip must still be independently monitored during execution.
    """
    planning = held_book_planning_arm(arm, arm.sim.data.qpos[arm.qadr], arm.pose()[1], body)
    model, data = planning.sim.model, planning.sim.data
    start, _ = planning.pose()
    position = np.asarray(position)
    count = max(2, int(np.ceil(np.linalg.norm(position - start) / spacing)))
    body_id = model.body(body).id
    adr = int(model.jnt_qposadr[model.body_jntadr[body_id]])
    tip, initial_r = planning.pose()
    relative_p = initial_r.T @ (data.xpos[body_id] - tip)
    relative_r = initial_r.T @ data.xmat[body_id].reshape(3, 3)
    links = {'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link',
             'wrist_2_link', 'wrist_3_link', 'tool0', 'wrist_rgb_mount', body}
    waypoints, reports = [], []
    peak = 0.
    for step in range(1, count + 1):
        q, metrics = planning.solve(start + (position - start) * step / count,
                                    rotation, seed=data.qpos[planning.qadr], max_iterations=500)
        path, report = plan_joint_rrt(planning, q, allowed_bodies=[body], wall_seconds=3.)
        previous_q = data.qpos[planning.qadr].copy()
        for waypoint in path:
            samples = max(9, int(np.max(np.abs(waypoint - previous_q)) / .025) + 2)
            for alpha in np.linspace(0, 1, samples):
                data.qpos[planning.qadr] = previous_q + alpha * (waypoint - previous_q)
                mujoco.mj_forward(model, data)
                sample_tip, sample_r = planning.pose()
                deviation = rotation_distance(rotation, sample_r)
                peak = max(peak, deviation)
                if deviation > maximum_rotation:
                    raise RuntimeError(f'Held-book path would rotate {deviation:.4f}rad away from its target')
                data.qpos[adr:adr + 3] = sample_tip + sample_r @ relative_p
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, (sample_r @ relative_r).ravel())
                data.qpos[adr + 3:adr + 7] = quat
                mujoco.mj_forward(model, data)
                for contact in data.contact:
                    if contact.dist >= -.0003:
                        continue
                    names = [model.body(int(model.geom_bodyid[g])).name or '' for g in contact.geom]
                    if not any(n in links or n.startswith('grip_') for n in names):
                        continue
                    if all(n.startswith('grip_') for n in names):
                        continue
                    if body in names and any(n.startswith('grip_') for n in names):
                        continue
                    raise RuntimeError('Held-book scratch path collision: ' + '/'.join(names)
                                       + f' penetration={-float(contact.dist):.6f}m')
            previous_q = waypoint
            waypoints.append(waypoint.copy())
        reports.append({'ik': metrics, 'planning': report})
    return waypoints, {'cartesian_steps': count, 'peak_orientation_deviation_rad': peak,
                       'held_object_pose_used_only_in_scratch': True, 'segments': reports}
