"""Aligned latch-edge door closing using wheel repositioning and real contacts."""
import math

import mujoco
import numpy as np

from robot_exam.control.joint_path import clear_joint_path
from robot_exam.control.joint_rrt import plan_joint_rrt
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.tasks.door import handle_position, inner_rotation, move_with_ik_seeds
from robot_exam.tasks.runtime import State, TaskFailure


def door_angle(runtime):
    joint=runtime.sim.model.joint('door_hinge').id
    return float(runtime.sim.data.qpos[runtime.sim.model.jnt_qposadr[joint]])


def closing_station(metadata, angle, margin=.06):
    """Keep the full chassis on the inside face, passing down the doorway axis.

    The long chassis axis follows the leaf, reducing its normal projection.
    Before full closure reserve space for a later collision-free chassis turn.
    """
    yaw=angle+math.pi/2+float(metadata.get('closing_yaw_bias_rad',-.30))
    normal=.60+.05*(1-min(1.,angle/.35))
    tangent=-.50-.24*math.sin(2*angle)**2
    hinge=np.asarray(metadata['hinge'])
    c,s=math.cos(angle),math.sin(angle)
    normal+=margin-.06
    local=np.array([.065+normal,metadata['handle_r']+tangent])
    xy=hinge[:2]+np.array([[c,-s],[s,c]])@local
    return np.r_[xy,yaw]


def handle_contacts(runtime, inner):
    model,data=runtime.sim.model,runtime.sim.data
    target=model.geom('inner_handle' if inner else 'outer_handle').id
    rows=[]
    for index,contact in enumerate(data.contact):
        if target not in contact.geom:continue
        other=int(contact.geom[0] if contact.geom[1]==target else contact.geom[1])
        body=model.body(int(model.geom_bodyid[other])).name or ''
        if not body.startswith('grip_'):continue
        force=np.empty(6);mujoco.mj_contactForce(model,data,index,force)
        rows.append({'body':body,'handle_geom':model.geom(target).name,
                     'normal_force_n':float(force[0]),'position':contact.pos.tolist()})
    return rows


def bilateral_handle(runtime,inner=True):
    return runtime.gripper.bilateral(handle_contacts(runtime,inner))


def safe_joint_move(runtime,target,seconds=3.):
    try:path=clear_joint_path(runtime.arm,target)
    except RuntimeError:
        path,report=plan_joint_rrt(runtime.arm,target,wall_seconds=12.,max_nodes=3500)
        runtime.ik_diagnostics.append({'context':runtime.stage,'joint_path':report})
    for waypoint in path:runtime.joint_move(waypoint,seconds,_collision_checked=True)


def grasp_inner(runtime,metadata,at_handle=False):
    r=runtime;angle=door_angle(r);point=handle_position(metadata,angle,True)
    rotation=inner_rotation(r,point)
    r.transition(State.APPROACH,'aligned_inner_latch_pregrasp',target=point.tolist())
    r.gripper.command(0)
    if not at_handle:
        pre=point-.09*rotation[:,2]
        q,_=r.solve_ik(pre,rotation,max_iterations=500,context='aligned_latch_pregrasp')
        safe_joint_move(r,q)
    move_with_ik_seeds(r,point,rotation,seconds=.6 if at_handle else 1.4,tolerance=.012)
    r.gripper.command(1);r.hold(.7)
    contacts=handle_contacts(r,True)
    if not r.gripper.bilateral(contacts):
        raise TaskFailure('handle_grasp','Aligned inner latch lacks bilateral contacts on the actual handle capsule')
    return contacts


class WorldTipReference:
    """Released hand follows a safe handle prepose during wheel motion via IK."""
    def __init__(self,runtime,metadata,distance):
        self.r=runtime
        self.metadata=metadata;self.distance=distance
        self.steps=0

    def before_step(self):pass

    def after_step(self):
        self.steps+=1
        model=self.r.sim.model
        leaf=model.body('door_leaf').id
        for contact in self.r.sim.data.contact:
            a,b=[int(model.geom_bodyid[g]) for g in contact.geom]
            if leaf not in (a,b):continue
            other=b if a==leaf else a
            name=model.body(other).name or ''
            if other in self.r._robot and not name.startswith('grip_') and contact.dist<-.001:
                raise TaskFailure('base_collision',f'Non-hand robot contact with door during coordinated close: {name}, {-contact.dist:.4f}m')
        if self.steps%10:return
        try:
            point=handle_position(self.metadata,door_angle(self.r),True)
            rotation=inner_rotation(self.r,point)
            self.r.qgoal,_=self.r.solve_ik(point-self.distance*rotation[:,2],rotation,seed=self.r.qgoal,
                                         max_iterations=200,context='aligned_door_world_tip_during_reposition')
        except RuntimeError as error:
            raise TaskFailure('ik_unreachable','Released hand world reference became unreachable during wheel reposition') from error


def retreat(runtime,final=False):
    r=runtime;position,rotation=r.arm.pose();failures=[]
    for distance,dz in ((.08,0.),(.06,0.),(.10,0.),(.06,.02),(.12,-.02)):
        target=position-distance*rotation[:,2]+[0,0,dz]
        try:q,metrics=r.arm.solve(target,rotation,seed=r.qgoal,max_iterations=250)
        except RuntimeError as error:
            failures.append(str(error));continue
        r.ik_diagnostics.append({'context':'aligned_handle_retreat_candidate','distance_m':distance,'dz_m':dz,**metrics})
        safe_joint_move(r,q,seconds=1.2 if final else .9)
        error=float(np.linalg.norm(r.arm.pose()[0]-target))
        if error>.018:raise TaskFailure('arm_tracking',f'Actual released-hand retreat error {error:.4f}m')
        return distance
    raise TaskFailure('ik_unreachable','No reachable released-hand retreat: '+' | '.join(failures[-2:]))


def dock_station(runtime,goal,timeout=50,tolerance=.025):
    """Re-enter position correction after a measured, stopped lateral residual."""
    r=runtime;previous=r.config.get('dock_residual_reentry_seconds')
    previous_strategy=r.config.get('local_dock_strategy')
    r.config['dock_residual_reentry_seconds']=float(r.config.get('aligned_door_residual_reentry_seconds',.8))
    if r.config.get('aligned_door_stow_between_segments',True):
        r.config['local_dock_strategy']=r.config.get('aligned_door_dock_strategy','preapproach')
    try:
        try:return r.dock(goal,timeout=timeout,tolerance=tolerance)
        except RuntimeError as error:
            if str(error)!='No collision-free preapproach connection':raise
            r.config['local_dock_strategy']='lattice'
            return r.dock(goal,timeout=timeout,tolerance=tolerance)
    finally:
        if previous is None:r.config.pop('dock_residual_reentry_seconds',None)
        else:r.config['dock_residual_reentry_seconds']=previous
        if previous_strategy is None:r.config.pop('local_dock_strategy',None)
        else:r.config['local_dock_strategy']=previous_strategy


def close_latch(runtime,metadata,already_at_station=False):
    r=runtime;trace=[];retries=0
    if not already_at_station:
        r.transition(State.NAVIGATE,'return_to_open_latch_work_position',timeout=180)
        grid=OccupancyGrid(bounds=(-6,4,-3,3))
        for box in metadata['obstacles']:grid.rectangle(*box)
        # Reapproach the inner face through the open doorway. This may bring
        # the chassis back to the threshold; final entry is independently checked.
        goal=closing_station(metadata,door_angle(r))
        r.navigate(goal,grid,timeout=150)
        dock_station(r,goal,timeout=45,tolerance=.03)
    retained=False;at_handle=False
    while door_angle(r)>.025:
        before=door_angle(r)
        try:contacts=handle_contacts(r,True) if retained else grasp_inner(r,metadata,at_handle=at_handle)
        except TaskFailure as error:
            if error.category!='handle_grasp' or retries>=2:raise
            retries+=1;r.retries+=1
            r.transition(State.RECOVER,'aligned_latch_contact_retry',timeout=60)
            r.gripper.command(0);r.hold(.3)
            continue
        r.transition(State.MANIPULATE,'aligned_latch_close_arc',target=before,timeout=120)
        target=max(0.,before-float(r.config.get('aligned_door_close_step_rad',.08)))
        point=handle_position(metadata,target,True);rotation=inner_rotation(r,point)
        move_with_ik_seeds(r,point,rotation,seconds=.7,tolerance=.018)
        after=door_angle(r)
        trace.append({'time':float(r.sim.data.time),'before':before,'target':target,'actual':after,
                      'base':r.base.pose().tolist(),'contacts_at_grasp':contacts,
                      'bilateral_at_end':bilateral_handle(r,True),
                      'tip_error_m':float(np.linalg.norm(r.arm.pose()[0]-point))})
        if r.output:
            import json
            (r.output/'aligned_close_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
        if before-after<.008:
            raise TaskFailure('door_stuck',f'Normal latch arc made insufficient physical progress: {before:.3f}->{after:.3f}')
        if after<=.025:
            r.hold(1.5);r.gripper.command(0);r.hold(.7)
            break
        if r.config.get('aligned_door_stow_between_segments',True):
            r.gripper.command(0);r.hold(.3)
            r.transition(State.APPROACH,'aligned_handle_release_collision_checked_stow',timeout=90)
            safe_joint_move(r,r.STOW,seconds=2.5)
            goal=closing_station(metadata,door_angle(r))
            r.transition(State.NAVIGATE,'aligned_stowed_chassis_local_path',target=goal.tolist(),timeout=120)
            dock_station(r,goal,timeout=90)
            retained=False;at_handle=False
            continue
        if r.config.get('aligned_door_keep_contact',False):
            goal=closing_station(metadata,door_angle(r))
            r.transition(State.MANIPULATE,'aligned_latch_base_follow_with_contact',target=goal.tolist(),timeout=90)
            hold=WorldTipReference(r,metadata,0.);r.sim.add_step_controller(hold)
            try:dock_station(r,goal)
            finally:r.sim.remove_step_controller(hold)
            retained=bilateral_handle(r,True)
            if not retained:
                retries+=1;r.retries+=1
                if retries>3:raise TaskFailure('handle_grasp','Aligned latch contact lost during repeated chassis follow')
                r.gripper.command(0);r.hold(.3)
            continue
        r.gripper.command(0);r.hold(.2)
        if r.config.get('aligned_door_regrasp_at_handle',True):
            goal=closing_station(metadata,door_angle(r))
            r.transition(State.MANIPULATE,'aligned_open_hand_wheel_reposition',target=goal.tolist(),timeout=90)
            hold=WorldTipReference(r,metadata,0.);r.sim.add_step_controller(hold)
            try:dock_station(r,goal)
            finally:r.sim.remove_step_controller(hold)
            retained=False;at_handle=True
            continue
        distance=retreat(r)
        goal=closing_station(metadata,door_angle(r))
        r.transition(State.NAVIGATE,'aligned_latch_wheel_reposition',target=goal.tolist(),timeout=90)
        hold=WorldTipReference(r,metadata,distance);r.sim.add_step_controller(hold)
        try:dock_station(r,goal)
        finally:r.sim.remove_step_controller(hold)
    # Release and withdraw before folding, then clear the entire robot inside.
    r.transition(State.APPROACH,'aligned_latch_release_and_retreat',timeout=90)
    r.gripper.command(0)
    if r.config.get('aligned_door_stow_between_segments',True):
        safe_joint_move(r,r.STOW,seconds=3.5)
    else:
        goal=closing_station(metadata,door_angle(r))
        hold=WorldTipReference(r,metadata,0.);r.sim.add_step_controller(hold)
        try:dock_station(r,goal)
        finally:r.sim.remove_step_controller(hold)
        retreat(r,final=True)
        safe_joint_move(r,r.STOW,seconds=3.5)
    r.transition(State.NAVIGATE,'aligned_closed_door_inside_clearance',timeout=120)
    # The polar dock controller can initially reverse towards the closed leaf
    # while correcting heading. Plan wheel-executable clearance primitives
    # against the actual closed-door geometry before making that turn.
    previous_tracking=r.config.get('local_dock_tracking_version')
    r.config['local_dock_tracking_version']='anchored_v4'
    try:dock_station(r,[1.1,0.,0.],timeout=100,tolerance=.04)
    finally:
        if previous_tracking is None:r.config.pop('local_dock_tracking_version',None)
        else:r.config['local_dock_tracking_version']=previous_tracking
    r.base.command(0,0);r.hold(2.)
    final=door_angle(r)
    model,data=r.sim.model,r.sim.data
    minimum=math.inf
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in r._robot:continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):continue
        rotation=data.geom_xmat[geom].reshape(3,3)
        bounds=model.geom_aabb[geom]
        center=data.geom_xpos[geom]+rotation@bounds[:3]
        minimum=min(minimum,float(center[0]-np.abs(rotation[0])@bounds[3:]))
    success=abs(final)<.035 and r.base.pose()[0]>.7 and minimum>.07 and not r.collision_events
    return {'success':bool(success),'final_door_angle':final,'final_base':r.base.pose().tolist(),
            'whole_robot_min_x_m':minimum,'close_trace':trace,'close_regrasp_count':retries,
            'mode':'Oracle paired latch geometry, segmented physical closing with wheel repositioning',
            'scope':'aligned latch-edge close; entry witness supplied by complete task runner',
            'handle_layout':metadata['handle_layout']}
