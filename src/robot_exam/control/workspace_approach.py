"""Real-wheel corridor travel and explicitly reselected feasible workstations.

The nominal operating pose is a planning variable. Every accepted actual pose
is certified against the complete arm targets and then passes the unchanged
strict stopped dock. No live state or passive object coordinate is assigned.
"""
import json
import math
import numpy as np
from robot_exam.control.base import wrap_angle
from robot_exam.control.base_pose import evaluate_operational_base_poses


def corridor_to_y(r,grid,y_goal,x_corridor=-.85,linear_limit=.35):
    from robot_exam.tasks.runtime import TaskFailure
    if r.base.pose()[0]>x_corridor+.012:r.backout(x_corridor,timeout=15)
    start=float(r.sim.data.time);origin=r.base.pose().copy();direction=1. if y_goal>origin[1] else -1.
    yaw=direction*math.pi/2;anchor=origin[:2].copy()
    while abs(wrap_angle(yaw-r.base.pose()[2]))>.035:
        if r.sim.data.time-start>18:raise TaskFailure('navigation_timeout','Corridor entry turn timed out')
        r.base.rotate_velocity(anchor,yaw,body_yaw_limit=float(r.config.get('workspace_loaded_turn_rate_rad_s',.25)) if r.payload else .8);r.tick()
        if r.collision_events:raise TaskFailure('base_collision','Collision while orienting into free corridor')
    r.base.command(0,0);r.hold(.5)
    # Certify the prospective straight corridor in the original inflated map.
    if not grid.visible(grid.cell(r.base.pose()),grid.cell([x_corridor,y_goal])):
        raise TaskFailure('path_planning','No free inflated-map corridor')
    while direction*(y_goal-r.base.pose()[1])>.055:
        if r.sim.data.time-start>50:raise TaskFailure('navigation_timeout','Corridor translation timed out')
        pose=r.base.pose();distance=abs(y_goal-pose[1])
        target_yaw=math.atan2(direction*.45,x_corridor-pose[0])
        error=wrap_angle(target_yaw-pose[2]);velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
        v=min(linear_limit,1.3*distance) if abs(error)<.6 else 0.
        w=float(np.clip(4.*error-.7*velocity[5]+(.35*np.sign(error) if abs(error)>.03 else 0.),-2.5,2.5))
        r.base.command(v,w);r.tick()
        if r.collision_events:raise TaskFailure('base_collision','Collision during free corridor travel')
    r.base.command(0,0);r.hold(.6)
    anchor=r.base.pose()[:2].copy();turned=float(r.sim.data.time)
    while abs(wrap_angle(r.base.pose()[2]))>.035:
        if r.sim.data.time-turned>18:raise TaskFailure('navigation_timeout','Free corridor operating-heading turn timed out')
        r.base.rotate_velocity(anchor,0.,body_yaw_limit=float(r.config.get('workspace_loaded_turn_rate_rad_s',.25)) if r.payload else .8);r.tick()
        if r.collision_events:raise TaskFailure('base_collision','Collision while aligning work heading')
    r.base.command(0,0);r.hold(.6)
    if not grid.free(grid.cell(r.base.pose())):raise TaskFailure('path_planning','Corridor endpoint left the original free map')
    return {'origin':origin.tolist(),'actual':r.base.pose().tolist(),'elapsed_sim_seconds':float(r.sim.data.time)-start}


def approach_feasible_workspace(r,targets,maximum_x=.08,linear_limit=.25,name='workspace',minimum_x=-float('inf'),tolerance=.03):
    from robot_exam.tasks.runtime import TaskFailure
    r._release_operation_hold()
    started=float(r.sim.data.time);trials=[]
    try:
        for _ in range(8):
            pose=r.base.pose()
            if pose[0]>maximum_x:break
            next_x=min(maximum_x,max(pose[0]+.20,minimum_x+.01))
            while r.base.pose()[0]<next_x-.015:
                if r.sim.data.time-started>35:raise TaskFailure('navigation_timeout','Workspace approach budget exceeded')
                pose=r.base.pose();velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
                error=wrap_angle(-pose[2]);v=min(linear_limit,1.3*max(0,next_x-pose[0]))
                w=float(np.clip(4*error-.7*velocity[5]+(.35*np.sign(error)if abs(error)>.03 else 0.),-.85,.85))
                r.base.command(v,w);r.tick()
                if r.collision_events:raise TaskFailure('base_collision','Collision during workspace approach')
            r.base.command(0,0);r.hold(.5);actual=r.base.pose().copy()
            if actual[0]<minimum_x:continue
            previous_grid=getattr(r,'grid',None);r.grid=None
            try:report=evaluate_operational_base_poses(r,actual,targets,offsets=[(0,0,0)])
            finally:r.grid=previous_grid
            trials.append({'actual':actual.tolist(),'feasibility':report})
            if report['selected'] is not None:
                # A fixed, feasible new goal; never recenter a failing dock.
                r.dock(actual,timeout=10.,tolerance=tolerance)
                return actual
            if actual[0]>=maximum_x-.02:break
        raise TaskFailure('placement_unreachable','No actual stopped pose certifies the complete operation path')
    finally:
        if r.output:(r.output/(name+'.json')).write_text(json.dumps(trials,indent=2)+'\n')
