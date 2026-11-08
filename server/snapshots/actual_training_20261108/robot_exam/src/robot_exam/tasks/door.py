"""Passive hinged-door manipulation expert; joint truth explicitly Oracle-labelled."""
import math
import numpy as np
from robot_exam.tasks.runtime import State,TaskFailure
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.control.joint_path import clear_joint_path


def rotation_z(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def handle_position(metadata,angle,inner=False):
    local=np.array([.065 if inner else -.065,metadata.get('inner_handle_radius',.06) if inner else metadata.get('handle_r',1.22),metadata.get('inner_handle_height',.65) if inner else metadata['handle_height']])
    return np.asarray(metadata['hinge'])+rotation_z(angle)@local


def move_with_ik_seeds(runtime,position,rotation,seconds,tolerance=.018):
    """Retry a failed geometric IK solve with bounded UR5 branches."""
    r=runtime
    try:return r.move(position,rotation,seconds=seconds,tolerance=tolerance)
    except TaskFailure as exc:
        if exc.category!='ik_unreachable':raise
    failures=[]
    for seed in ([0,-1.5,2,-2,1.57,0],[0,-1.1,1.9,-2.35,1.57,0],[0,-1.9,2.2,-1.9,1.57,0]):
        try:
            q,_=r.solve_ik(position,rotation,seed=seed,max_iterations=500,context='door_handle_branch_search')
            r.qgoal=q
            return r.move(position,rotation,seconds=seconds,tolerance=tolerance)
        except (RuntimeError,TaskFailure) as exc:
            failures.append(str(exc))
    raise TaskFailure('ik_unreachable','All bounded door IK branches failed: '+' | '.join(failures[-2:]))


def inner_rotation(runtime,position):
    forward=np.asarray(position)-np.r_[runtime.base.pose()[:2],position[2]]
    forward[2]=0
    norm=np.linalg.norm(forward[:2])
    if norm<1e-8:raise TaskFailure('ik_unreachable','Degenerate inner handle approach direction')
    forward/=norm
    return np.column_stack([[0,0,1],[forward[1],-forward[0],0],forward])


def door_grasp(runtime,metadata,inner=False):
    r=runtime;angle=float(r.sim.data.qpos[r.sim.model.jnt_qposadr[r.sim.model.joint('door_hinge').id]])
    R0=np.array([[0,0,1],[0,-1,0],[1,0,0.]])
    if inner:R0=R0@np.diag([-1,1,-1])
    R=rotation_z(angle)@R0;p=handle_position(metadata,angle,inner)
    if inner:
        forward=p-r.arm.pose()[0];forward[2]=0;forward/=np.linalg.norm(forward)
        R=np.column_stack([[0,0,1],[forward[1],-forward[0],0],forward]);R0=rotation_z(-angle)@R
    r.transition(State.APPROACH,'door_inner_handle' if inner else 'door_outer_handle',target=p.tolist())
    r.gripper.command(0)
    if inner:
        q,_=r.solve_ik(p-R[:,2]*.09,R,seed=[0,-1.5,2,-2,1.57,0],max_iterations=500,context='door_inner_pregrasp')
        for waypoint in clear_joint_path(r.arm,q):r.joint_move(waypoint,4.)
    else:move_with_ik_seeds(r,p-R[:,2]*.09,R,seconds=2.2)
    move_with_ik_seeds(r,p,R,seconds=1.6)
    r.gripper.command(1);r.hold(.9)
    if not r.bilateral('door_leaf'):raise TaskFailure('handle_grasp','Door handle lacks bilateral physical contact')
    if metadata.get('handle_layout')=='paired_latch_edge':
        from robot_exam.tasks.door_latch import bilateral_handle
        if not bilateral_handle(r,inner):raise TaskFailure('handle_grasp','Contact is not bilateral on the declared latch handle')
    return angle,R0


def _open_segment_once(runtime,metadata,target):
    r=runtime;angle,R0=door_grasp(r,metadata)
    r.transition(State.MANIPULATE,'pull_hinge_arc',target=target)
    for theta in np.linspace(angle,target,max(3,round(abs(target-angle)/.012)))[1:]:
        p=handle_position(metadata,theta)
        move_with_ik_seeds(r,p,rotation_z(theta)@R0,seconds=.20)
    actual=float(r.sim.data.qpos[r.sim.model.jnt_qposadr[r.sim.model.joint('door_hinge').id]])
    if abs(actual-target)>.07:raise TaskFailure('door_stuck',f'Opening arc target {target}, actual {actual}')
    return actual


def open_segment(runtime,metadata,target):
    """Versioned bounded physical regrasp; zero retries preserves the baseline."""
    r=runtime;limit=int(r.config.get('door_open_regrasp_attempts',0))
    for attempt in range(limit+1):
        try:return _open_segment_once(r,metadata,target)
        except TaskFailure as exc:
            if exc.category not in ('door_stuck','handle_grasp') or attempt==limit:raise
            actual=float(r.sim.data.qpos[r.sim.model.jnt_qposadr[r.sim.model.joint('door_hinge').id]])
            r.retries+=1
            r.transition(State.RECOVER,'open_handle_contact_regrasp',target={'target':target,'actual':actual,
                         'attempt':attempt+1,'contacts':r.contacts('door_leaf')},timeout=60)
            r.base.command(0,0);r.gripper.command(0);r.hold(.3)
            retreat=r.arm.pose()[0]-rotation_z(actual)[:,0]*.035+np.array([0,0,.04])
            move_with_ik_seeds(r,retreat,r.arm.pose()[1],seconds=1)
    raise AssertionError('Unreachable bounded regrasp exit')


def open_and_enter(runtime,metadata):
    r=runtime;grid=OccupancyGrid(bounds=(-6,4,-3,3))
    for box in metadata['obstacles']:grid.rectangle(*box)
    r.hold(.6);r.transition(State.NAVIGATE,'five_metre_door_approach')
    r.navigate([-1.2,.42,.59],grid)
    r.dock(metadata['opening_base'],timeout=float(r.config.get('door_initial_dock_timeout_seconds',20)),tolerance=.045)
    # Segment the arc and move the wheeled base backwards along its tangent.
    # Only wheel actuators and UR5 ctrl drive either rigid body.
    angles=[]
    for target in np.arange(.10,1.561,.10):
        actual=open_segment(r,metadata,float(target));angles.append(actual)
        r.gripper.command(0);r.hold(.3)
        r.move(r.arm.pose()[0]-rotation_z(actual)[:,0]*.05+np.array([0,0,.05]),rotation_z(actual)@np.array([[0,0,1],[0,-1,0],[1,0,0.]]),seconds=1)
        r.stow()
        # The passive hinge drifts during release and chassis repositioning.
        # Plan the next handle approach from the measured angle after settling,
        # never from the previous commanded angle.
        hinge_adr=r.sim.model.jnt_qposadr[r.sim.model.joint('door_hinge').id]
        measured_angle=float(r.sim.data.qpos[hinge_adr])
        radius=metadata.get('handle_r',1.22)
        ideal=np.asarray(metadata['hinge'])[:2]+(rotation_z(measured_angle)@np.array([-.815,radius,0]))[:2]
        desired_yaw=measured_angle+math.atan2(.815,radius)
        r.transition(State.NAVIGATE,'door_arc_base_reposition',target=[*ideal,desired_yaw])
        # The former local reverse controller recomputed a backward heading
        # from the instantaneous target vector. Near the target this vector
        # changed sides and caused near-full chassis rotations. Use the shared
        # A* tracker to reach a staging pose, then the tested physical dock
        # controller for the final short approach and heading correction.
        staging=np.array([ideal[0]-.34*np.cos(desired_yaw),
                          ideal[1]-.34*np.sin(desired_yaw),desired_yaw])
        try:
            r.navigate(staging,grid,timeout=45)
            r.dock([*ideal.tolist(),desired_yaw],timeout=25,tolerance=.035)
        except TaskFailure as exc:
            raise TaskFailure('navigation_timeout',
                              f'Door arc physical reposition failed: {exc.category}: {exc}') from exc
        r.base.command(0,0);r.hold(.4)
    # Complete the final six degrees if needed, then release and stow.
    actual=open_segment(r,metadata,1.56);r.gripper.command(0);r.hold(.3)
    r.move(r.arm.pose()[0]-rotation_z(actual)[:,0]*.05+np.array([0,0,.05]),rotation_z(actual)@np.array([[0,0,1],[0,-1,0],[1,0,0.]]),seconds=1)
    r.stow()
    r.base.command(-.08,0);r.hold(2.0);r.base.command(0,0);r.hold(.4)
    r.transition(State.NAVIGATE,'enter_room_through_open_door')
    passage=OccupancyGrid(bounds=(-6,4,-3,3))
    for box in metadata['obstacles']:passage.rectangle(*box)
    passage.rectangle(-1.6,0,-.85,-.73)
    r.navigate([1.0,0,0],passage,timeout=180)
    # Entire robot entered before any closing action.
    if r.base.pose()[0]<.7:raise TaskFailure('not_entered','Base has not cleared door plane')
    return angles


def execute(runtime,metadata,mode='oracle'):
    if runtime.config.get('door_contact_controller')=='tangent_arc_v4':
        if metadata.get('handle_layout')!='paired_latch_edge':
            raise TaskFailure('model_contract','Continuous contact requires the normal paired-latch model')
        from robot_exam.tasks.door_contact import execute_contact_task
        return execute_contact_task(runtime,metadata)
    r=runtime;angles=open_and_enter(r,metadata)
    if metadata.get('handle_layout')=='paired_latch_edge':
        from robot_exam.tasks.door_latch import close_latch
        result=close_latch(r,metadata)
        result.update(opened_angles=angles,entered_before_close=True,
                      scope='complete open-enter-reapproach-normal-latch-close attempt')
        return result
    passage=OccupancyGrid(bounds=(-6,4,-3,3))
    for box in metadata['obstacles']:passage.rectangle(*box)
    passage.rectangle(-1.6,0,-.85,-.73)
    r.navigate(metadata['closing_base'],passage,timeout=100)
    r.dock(metadata['closing_base'],timeout=30,tolerance=.03)
    r.base_hold_anchor=r.base.pose().copy()
    angle,R0=door_grasp(r,metadata,inner=True)
    r.transition(State.MANIPULATE,'close_from_inside',timeout=180)
    close_trace=[];regrasp_count=0;stalled=0
    if not r.config.get('door_close_feedback',True):
        for theta in np.linspace(angle,0,100)[1:]:
            p=handle_position(metadata,theta,inner=True);R=inner_rotation(r,p)
            move_with_ik_seeds(r,p,R,seconds=.20)
        r.gripper.command(0);r.hold(.5)
    else:
    # Incremental closed-loop arc while retaining the same physical grasp.
    # Regrasp only after measured hinge progress stalls, since unnecessary
    # release/recontact was itself causing missed inner-handle contacts.
        while angle>.025:
            start_angle=angle;target=max(-.12,start_angle-.12)
            p=handle_position(metadata,target,inner=True);R=inner_rotation(r,p)
            move_with_ik_seeds(r,p,R,seconds=.34,tolerance=.035)
            angle=float(r.sim.data.qpos[r.sim.model.jnt_qposadr[r.sim.model.joint('door_hinge').id]])
            progress=start_angle-angle
            close_trace.append({'time':float(r.sim.data.time),'target_angle':target,'actual_angle':angle,'progress_rad':progress})
            if angle<=.025:break
            stalled=stalled+1 if progress<.006 else 0
            if stalled>=3:
                regrasp_count+=1
                r.gripper.command(0);r.hold(.2)
                retreat=r.arm.pose()[0]-R[:,2]*.06
                move_with_ik_seeds(r,retreat,R,seconds=.7)
                if regrasp_count>4:
                    raise TaskFailure('door_stuck',f'No measured hinge progress after {regrasp_count} physical regrasp attempts; angle={angle:.3f}')
                angle,_=door_grasp(r,metadata,inner=True)
                r.transition(State.MANIPULATE,'close_from_inside_regrasp',target=angle,timeout=180)
                stalled=0
    # Keep contact at the closed stop long enough for the passive hinge to
    # dissipate closing velocity before letting go; release motion is then
    # followed by an independent final-angle check.
    r.hold(1.5)
    r.gripper.command(0);r.hold(1.0)
    r.move(r.arm.pose()[0]-R[:,2]*.06,R,seconds=1.2)
    post_release_recoveries=0
    hinge_adr=r.sim.model.jnt_qposadr[r.sim.model.joint('door_hinge').id]
    final=float(r.sim.data.qpos[hinge_adr])
    while r.config.get('door_close_feedback',True) and final>.035 and post_release_recoveries<2:
        post_release_recoveries+=1
        angle,_=door_grasp(r,metadata,inner=True)
        target=max(-.12,angle-.20);p=handle_position(metadata,target,inner=True);R=inner_rotation(r,p)
        move_with_ik_seeds(r,p,R,seconds=.55,tolerance=.035)
        actual=float(r.sim.data.qpos[hinge_adr])
        close_trace.append({'time':float(r.sim.data.time),'target_angle':target,'actual_angle':actual,
                            'progress_rad':angle-actual,'phase':'post_release_recovery'})
        if angle-actual<.025:
            r.gripper.command(0)
            raise TaskFailure('door_stuck',f'Post-release physical close recovery made insufficient progress: {angle:.3f}->{actual:.3f}')
        r.hold(1.5);r.gripper.command(0);r.hold(1.0)
        r.move(r.arm.pose()[0]-R[:,2]*.06,R,seconds=1.2)
        final=float(r.sim.data.qpos[hinge_adr])
    r.stow()
    r.base_hold_anchor=None
    # Orient in the room before a straight clearance drive. The generic dock
    # controller cannot reliably resolve a large lateral target from a sideways
    # heading in this confined work bay.
    r.transition(State.NAVIGATE,'clear_closed_door_inside_room')
    anchor=r.base.pose().copy()
    for _ in range(round(15/r.sim.model.opt.timestep)):
        if abs((r.base.pose()[2]+np.pi)%(2*np.pi)-np.pi)<.03:break
        r.base.rotate(anchor[:2],0);r.tick()
    else:raise TaskFailure('navigation_timeout','Inside clearance orientation failed')
    r.base.command(0,0);r.hold(.4)
    r.dock([1.,float(r.base.pose()[1]),0],timeout=20,tolerance=.07)
    final=float(r.sim.data.qpos[hinge_adr])
    success=abs(final)<.035 and r.base.pose()[0]>.7 and not r.collision_events
    return {'success':bool(success),'opened_angles':angles,'close_trace':close_trace,'close_regrasp_count':regrasp_count,'post_release_recoveries':post_release_recoveries,'final_door_angle':final,'final_base':r.base.pose().tolist(),
            'mode':'oracle hinge/handle geometry','scope':'complete open-enter-close attempt'}
