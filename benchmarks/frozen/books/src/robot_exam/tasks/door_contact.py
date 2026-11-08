"""Wheel/arm closed-chain door arcs, with bounded physical safety release.

Oracle hinge geometry is explicit. The live base and passive hinge are never
assigned a pose. The base path has tangent heading (no lateral primitive).
"""
import json
import math
from types import SimpleNamespace

import mujoco
import numpy as np

from robot_exam.control.arm import ArmController
from robot_exam.control.base import wrap_angle
from robot_exam.tasks.door import handle_position, rotation_z
from robot_exam.tasks.door_latch import door_angle, handle_contacts
from robot_exam.tasks.runtime import State, TaskFailure


def arc_parameters(inner,metadata=None):
    normal,tangent=(.65,1.00) if inner else (-.70,1.30)
    selected=(metadata or {}).get('contact_arc_parameters',{}).get('close' if inner else 'open')
    if selected is not None:normal,tangent=selected
    offset=math.atan2(normal,-tangent)+(0. if inner else math.pi)
    return normal,tangent,offset


def arc_station(metadata,angle,inner):
    normal,tangent,offset=arc_parameters(inner,metadata)
    xy=np.asarray(metadata['hinge'])[:2]+(rotation_z(angle)@np.array([normal,tangent,0.]))[:2]
    return np.r_[xy,angle+offset]


def arc_rotation(metadata,angle,inner):
    # Oblique pinch puts most hinge-arc load across the jaws rather than
    # pulling axially along the finite silicone pad. Keep the original bar,
    # friction and actuator force limits unchanged.
    skew=math.pi/3
    forward=rotation_z(angle)@np.array([math.cos(skew)*(-1. if inner else 1.),math.sin(skew),0.])
    return np.column_stack([[0,0,1],[forward[1],-forward[0],0],forward])


def select_start_arc(runtime,metadata,inner,target):
    """Validate a tangent circle through the measured base, in scratch only.

    A different feasible operating pose is selected, rather than declaring an
    inaccurate dock successful. Original task/dock tolerances are untouched.
    """
    r=runtime;m=r.sim.model;angle=door_angle(r);pose=r.base.pose()
    normal,tangent=(rotation_z(-angle)@np.r_[pose[:2]-np.asarray(metadata['hinge'])[:2],0.])[:2]
    offset=math.atan2(normal,-tangent)+(0. if inner else math.pi)
    report={'time':float(r.sim.data.time),'phase':'close' if inner else 'open',
            'normal':float(normal),'tangent':float(tangent),'angle':angle,'target':target,'accepted':False}
    safe_side=(.55<normal<.80 and .80<tangent<1.10) if inner else (-.90<normal<-.55 and 1.1<tangent<1.55)
    if safe_side and abs(wrap_angle(angle+offset-pose[2]))<.15:
        scratch=mujoco.MjData(m);scratch.qpos[:]=r.sim.data.qpos;scratch.ctrl[:]=r.sim.data.ctrl
        scratch.qpos[3:7]=[math.cos((angle+offset)/2),0,0,math.sin((angle+offset)/2)]
        mujoco.mj_forward(m,scratch);virtual=SimpleNamespace(model=m,data=scratch)
        arm=ArmController(virtual)
        try:
            q,metrics=arm.solve(handle_position(metadata,angle,inner),arc_rotation(metadata,angle,inner),
                                seed=r.qgoal,max_iterations=350)
            scratch.qpos[arm.qadr]=q;ha=int(m.jnt_qposadr[m.joint('door_hinge').id]);risk=None
            for a in np.linspace(angle,target,max(3,int(abs(target-angle)/.025)+2)):
                xy=np.asarray(metadata['hinge'])[:2]+(rotation_z(a)@np.array([normal,tangent,0.]))[:2]
                scratch.qpos[:2]=xy;scratch.qpos[3:7]=[math.cos((a+offset)/2),0,0,math.sin((a+offset)/2)];scratch.qpos[ha]=a
                mujoco.mj_forward(m,scratch)
                for c in scratch.contact:
                    if c.dist>=-.0005:continue
                    g1,g2=map(int,c.geom);b1,b2=map(int,m.geom_bodyid[[g1,g2]])
                    if b1 not in r._robot and b2 not in r._robot:continue
                    names=[m.body(b).name or '' for b in (b1,b2)]
                    if 'floor' in (m.geom(g1).name,m.geom(g2).name):continue
                    if all(n.startswith('grip_') for n in names):continue
                    if 'door_leaf' in names and any(n.startswith('grip_') for n in names):continue
                    risk={'angle':float(a),'bodies':names,'depth_m':-float(c.dist)};break
                if risk:break
            report.update(accepted=risk is None,collision_risk=risk,ik=metrics)
        except RuntimeError as error:report['error']=str(error)
    if report['accepted']:
        metadata.setdefault('contact_arc_parameters',{})[report['phase']]=[float(normal),float(tangent)]
    history=getattr(r,'contact_path_choices',[]);history.append(report);r.contact_path_choices=history
    if r.output:(r.output/'contact_path_choices.json').write_text(json.dumps(history,indent=2)+'\n')
    return report


def grasp_contact(runtime,metadata,inner,retry=False,target=None):
    from robot_exam.tasks.door_latch import safe_joint_move
    from robot_exam.tasks.runtime import TaskFailure
    r=runtime
    r.transition(State.APPROACH,'continuous_inner_pregrasp' if inner else 'continuous_outer_pregrasp',timeout=100)
    r.gripper.command(0.)
    if retry:
        r.hold(.25)
        position,rotation=r.arm.pose()
        r.move(position-.05*rotation[:,2],rotation,seconds=.7,tolerance=.012)
        dof=int(r.sim.model.jnt_dofadr[r.sim.model.joint('door_hinge').id]);stable=0.
        for _ in range(120):
            r.hold(.05)
            stable=stable+.05 if abs(r.sim.data.qvel[dof])<.004 else 0.
            if stable>=.25:break
        else:raise TaskFailure('door_motion','Released leaf did not settle before bounded regrasp')
    select_start_arc(r,metadata,inner,(0. if inner else 1.56) if target is None else target)
    angle=door_angle(r);point=handle_position(metadata,angle,inner);rotation=arc_rotation(metadata,angle,inner)
    q,_=r.solve_ik(point-.09*rotation[:,2],rotation,max_iterations=450,context='continuous_oblique_pregrasp')
    safe_joint_move(r,q)
    angle=door_angle(r);point=handle_position(metadata,angle,inner);rotation=arc_rotation(metadata,angle,inner)
    r.move(point,rotation,seconds=1.2,tolerance=.012)
    if retry:
        angle=door_angle(r);point=handle_position(metadata,angle,inner);rotation=arc_rotation(metadata,angle,inner)
        r.move(point,rotation,seconds=.4,tolerance=.012)
    r.gripper.command(1.);r.hold(.7)
    if not r.gripper.bilateral(handle_contacts(r,inner)):
        raise TaskFailure('handle_grasp','Oblique grasp lacks bilateral capsule contact')


def unsafe_contact(runtime):
    """Intended finger/leaf and internal gripper linkage contact is allowed."""
    m,d=runtime.sim.model,runtime.sim.data
    for c in d.contact:
        if c.dist>=-.001:continue
        g1,g2=map(int,c.geom);b1,b2=map(int,m.geom_bodyid[[g1,g2]])
        if b1 not in runtime._robot and b2 not in runtime._robot:continue
        names=[m.body(b).name or '' for b in (b1,b2)]
        if 'floor' in [m.geom(g1).name,m.geom(g2).name]:continue
        if all(n.startswith('grip_') for n in names):continue
        if 'door_leaf' in names:
            bar=(m.geom(g1).name in ('inner_handle','outer_handle') or m.geom(g2).name in ('inner_handle','outer_handle'))
            if bar and any(n in ('grip_left_pad','grip_right_pad') for n in names):continue
        return {'bodies':names,'depth_m':-float(c.dist)}
    return None


def follow_contact_arc(runtime,metadata,target,inner):
    """One continuous closed-gripper arc; release only at an actual safety stop.

    Returns progress and safety evidence. Caller must not relabel an incomplete
    arc as a full task success, even if a subsequent fallback is attempted.
    """
    r=runtime;m,d=r.sim.model,r.sim.data;dt=float(m.opt.timestep)
    start=float(d.time);initial=door_angle(r);direction=1. if target>initial else -1.
    theta=initial;normal,tangent,_=arc_parameters(inner,metadata);radius=math.hypot(normal,tangent)
    hinge_dof=int(m.jnt_dofadr[m.joint('door_hinge').id])
    tangent_sign=1. if inner else -1.
    speed=float(r.config.get('door_contact_angle_speed_rad_s',.020))
    maximum_seconds=float(r.config.get('door_contact_arc_timeout_seconds',180.))
    r.transition(State.MANIPULATE,'continuous_inner_close' if inner else 'continuous_outer_open',
                 target=target,timeout=maximum_seconds+5.)
    r.gripper.command(1.)
    trace=[];reason=None;unsafe=None;lost=0.;tracking_bad=0.;completed=False
    pad_margin_bad=0.;last_advancing=start
    wheels=d.ctrl[r.base.ids].copy();q=r.qgoal.copy();linear=angular=0.;rate=0.;stable=0.
    joint_limits=m.jnt_range[r.arm.joints]
    fixed_arm=bool(r.config.get('door_contact_base_fixed_arm',False))
    reference_arc=fixed_arm or bool(r.config.get('door_contact_world_reference',True))
    max_error=0.;max_base_error=0.
    initial_position,initial_rotation=r.arm.pose()
    initial_local=initial_rotation.T@(handle_position(metadata,initial,inner)-initial_position)
    try:
        for step in range(round(maximum_seconds/dt)):
            actual=door_angle(r);position,rotation=r.arm.pose()
            # Contact is constrained by the measured passive hinge, not by an
            # obsolete open-loop reference. A few mm axial slip along a finger
            # pad must be corrected before the bar reaches the fingertip.
            point=handle_position(metadata,actual,inner)
            error=float(np.linalg.norm(position-point))
            contacts=handle_contacts(r,inner);bilateral=r.gripper.bilateral(contacts)
            forces={side:sum(x['normal_force_n'] for x in contacts if x['body'].startswith('grip_'+side)) for side in ('left','right')}
            lost=0. if bilateral else lost+dt
            tracking_bad=tracking_bad+dt if error>.025 else 0.
            unsafe=unsafe_contact(r)
            if unsafe:reason='collision';break
            if max(forces.values())>150.:reason='excessive_contact_force';break
            if lost>.080:reason='bilateral_contact_lost';break
            if error>.045 or tracking_bad>.10:reason='tool_tracking';break
            if r.collision_events:reason='static_collision';break
            if step%10==0:
                pose=r.base.pose();goal=arc_station(metadata,theta if reference_arc else actual,inner)
                delta=goal[:2]-pose[:2];base_error=float(np.linalg.norm(delta))
                max_error=max(max_error,error);max_base_error=max(max_base_error,base_error)
                if base_error>.12:reason='base_tracking';break
                # Slow/freeze the common reference before arm reach is consumed.
                reference_error=float(np.linalg.norm(position-handle_position(metadata,theta,inner)))
                local=rotation.T@(handle_position(metadata,actual,inner)-position)
                axial_slip=float(abs(local[2]-initial_local[2]))
                advance=(reference_error<.012 and axial_slip<.012 and base_error<.080 and bilateral)
                if advance:last_advancing=float(d.time)
                pad_margin_bad=pad_margin_bad+.02 if axial_slip>.012 else 0.
                if pad_margin_bad>.20:reason='pad_edge_margin';break
                if float(d.time)-last_advancing>3.:reason='reference_stalled';break
                gap=target-theta if reference_arc else target-actual
                wanted_rate=direction*min(speed,1.5*abs(gap)) if advance else 0.
                wanted_rate*=float(np.clip((.080-base_error)/.050,.25,1.))
                wanted_rate*=min(1.,(float(d.time)-start)/2.)
                measured_rate=float(d.qvel[hinge_dof])
                # A bounded Cartesian spring offset provides physical hinge
                # torque while velocity feedback prevents the leaf running
                # ahead and sliding axially out of the pad.
                lead=float(np.clip(.20*(wanted_rate-measured_rate),-.006,.006)) if advance else 0.
                theta=(theta+wanted_rate*.02) if reference_arc else float(np.clip(actual+lead,0.,1.56))
                theta=min(theta,target) if direction>0 else max(theta,target)
                rate=wanted_rate if reference_arc else float(np.clip(measured_rate,-.05,.05))
                point=handle_position(metadata,theta,inner);wanted_rotation=arc_rotation(metadata,theta,inner)
                if not fixed_arm:
                    try:q,metrics=r.arm.solve(point,wanted_rotation,seed=q,max_iterations=100)
                    except RuntimeError:reason='ik_unreachable';break
                if np.min(np.minimum(q-joint_limits[:,0],joint_limits[:,1]-q))<.025:
                    reason='joint_limit';break
                if np.max(np.abs(q-r.qgoal))>.15:reason='ik_branch_jump';break
                r.qgoal=q
                goal=arc_station(metadata,theta if reference_arc else actual,inner);delta=goal[:2]-pose[:2]
                heading=np.array([math.cos(pose[2]),math.sin(pose[2])]);lateral=float(delta@[-heading[1],heading[0]])
                velocity=d.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
                linear=float(np.clip(tangent_sign*radius*rate+1.2*(delta@heading),-.075,.075))
                heading_error=wrap_angle(goal[2]-pose[2]);reverse=-1. if tangent_sign*direction<0 else 1.
                steering_error=heading_error+reverse*2.*lateral
                angular=rate+4.*steering_error-.65*velocity[5]
                if abs(steering_error)>.008 or abs(rate)>.001:
                    angular+=math.copysign(.45,steering_error if abs(steering_error)>.008 else rate)
                angular=float(np.clip(angular,-.85,.85))
                r.base.command(linear,angular)
                trace.append({'time':float(d.time),'door_angle':actual,'reference_angle':theta,
                    'base':pose.tolist(),'base_goal':goal.tolist(),'base_error_m':base_error,
                    'tip_error_m':error,'command_tracking_error_m':reference_error,'axial_slip_m':axial_slip,
                    'handle_position_tool_m':local.tolist(),'left_force_n':forces['left'],'right_force_n':forces['right'],
                    'bilateral':bilateral,'reference_advancing':advance,'linear_command':linear,
                    'angular_command':angular,'linear_speed':float(np.linalg.norm(velocity[:2])),
                    'yaw_rate':float(velocity[5]),'q':d.qpos[r.arm.qadr].tolist(),
                    'arm_reference':r.arm.reference.tolist(),'joint_target':q.tolist()})
                done=(abs(target-actual)<(.010 if inner else .012) and abs(measured_rate)<.015 and error<.012)
                stable=stable+.02 if done else 0.
                if stable>=.6:completed=True;break
            desired=r.base.wheel_target.copy()
            wheels+=np.clip(desired-wheels,-1.2*dt,1.2*dt)
            d.ctrl[r.base.ids]=wheels
            r.tick()
        if not completed and reason is None:reason='arc_timeout'
    finally:
        r.base.command(0,0)
        # Do not chase a sliding bar here: that can drive the leaf further.
        # Brake with the last arm reference retained, then release if needed.
        r.hold(.5)
        if not completed:
            # Remove stored arm servo error under gravity compensation while
            # the fingers remain closed. Releasing a taut arm immediately
            # recoiled the otherwise passive leaf in the prior experiment.
            for i in range(round(.8/dt)):
                if i%10==0:r.qgoal=d.qpos[r.arm.qadr].copy()
                r.tick()
        if not completed:r.gripper.command(0.);r.hold(.3)
        result={'completed':completed,'initial_angle':initial,'target_angle':target,'actual_angle':door_angle(r),
                'seconds':float(d.time)-start,'safety_release_reason':reason,'unsafe_contact':unsafe,
                'max_tip_error_m':max_error,'max_base_error_m':max_base_error,
                'grasp_count':1,'safety_releases':0 if completed else 1,'samples':trace}
        if r.output:
            name='close' if inner else 'open'
            attempts=getattr(r,'door_contact_attempt_numbers',{})
            attempts[name]=attempts.get(name,0)+1;r.door_contact_attempt_numbers=attempts
            path=r.output/f'continuous_{name}_{attempts[name]:02d}.json'
            if path.exists():raise FileExistsError('Keep each contact attempt as separate evidence')
            result['evidence']=str(path)
            path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def run_contact_phase(runtime,metadata,target,inner):
    """At most four genuine grasps; regrasp only after a measured safety stop."""
    r=runtime;attempts=[];start=float(r.sim.data.time)
    limit=int(r.config.get('door_contact_max_grasps_per_phase',4))
    for index in range(limit):
        if index:
            r.retries+=1
            r.transition(State.RECOVER,'regrasp_after_contact_safety_stop',target=attempts[-1]['safety_release_reason'],timeout=100)
        try:grasp_contact(r,metadata,inner,retry=bool(index),target=target)
        except TaskFailure as error:
            error.partial={'contact_attempts':attempts,'contact_phase':'close' if inner else 'open',
                           'physical_grasp_attempts':index+1}
            raise
        result=follow_contact_arc(r,metadata,target,inner)
        attempts.append({k:v for k,v in result.items() if k!='samples'})
        if result['completed']:
            return {'completed':True,'actual_angle':door_angle(r),'seconds':float(r.sim.data.time)-start,
                    'grasp_count':len(attempts),'safety_releases':len(attempts)-1,'attempts':attempts}
        if result['safety_release_reason'] not in ('pad_edge_margin','reference_stalled','bilateral_contact_lost','tool_tracking'):
            break
        if abs(result['actual_angle']-result['initial_angle'])<.04:break
    error=TaskFailure('door_contact','Continuous-contact phase stopped: '+str(attempts[-1]['safety_release_reason']))
    error.partial={'contact_attempts':attempts,'contact_phase':'close' if inner else 'open'}
    raise error


def release_and_stow_legacy(runtime):
    """A necessary phase-boundary release, not a scheduled regrasp."""
    from robot_exam.tasks.door_latch import safe_joint_move
    r=runtime;r.base.command(0,0);r.hold(1.)
    for closure in np.linspace(1.,0.,21)[1:]:
        r.gripper.command(float(closure));r.hold(.04)
    r.hold(.3)
    position,rotation=r.arm.pose()
    r.move(position-.06*rotation[:,2],rotation,seconds=1.,tolerance=.018)
    safe_joint_move(r,r.STOW,seconds=3.5)


def release_and_stow(runtime,metadata,inner):
    if runtime.config.get('door_contact_release','legacy')!='normal_withdraw':
        return release_and_stow_legacy(runtime)
    from robot_exam.tasks.door_latch import safe_joint_move
    r=runtime;r.base.command(0,0);r.hold(1.)
    position,rotation=r.arm.pose();angle=door_angle(r)
    away=rotation_z(angle)[:,0]*(1. if inner else -1.)
    # Open while withdrawing away from the leaf plane. Opening the oblique
    # jaws at a stationary pinch pose could drag the passive leaf back open.
    seconds=1.;steps=round(seconds/r.sim.model.opt.timestep)
    release_trace=[]
    for i in range(steps):
        f=(i+1)/steps
        if i%10==0:
            target=position+.045*f*away
            r.qgoal,_=r.arm.solve(target,rotation,seed=r.qgoal,max_iterations=120)
        r.gripper.command(float(1.-f));r.tick()
        contacts=handle_contacts(r,inner)
        forces={side:sum(x['normal_force_n'] for x in contacts if x['body'].startswith('grip_'+side)) for side in ('left','right')}
        if max(forces.values())>150.:
            r.gripper.command(0.);r.qgoal=r.sim.data.qpos[r.arm.qadr].copy()
            raise TaskFailure('door_release_force','Release exceeded the same 150N per-side safety limit')
        collision=unsafe_contact(r)
        if collision:
            r.gripper.command(0.);r.qgoal=r.sim.data.qpos[r.arm.qadr].copy()
            raise TaskFailure('door_release_collision',str(collision))
        if i%10==0:release_trace.append({'time':float(r.sim.data.time),'closure':float(1.-f),
            'door_angle':door_angle(r),'left_force_n':forces['left'],'right_force_n':forces['right'],
            'tip_position':r.arm.pose()[0].tolist()})
    r.gripper.command(0.);r.hold(.3)
    target=position+.045*away
    if np.linalg.norm(r.arm.pose()[0]-target)>.018:
        raise TaskFailure('arm_tracking','Normal release withdrawal did not reach its checked endpoint')
    safe_joint_move(r,r.STOW,seconds=3.5)
    if r.output:
        (r.output/('contact_release_close.json' if inner else 'contact_release_open.json')).write_text(
            json.dumps({'strategy':'normal_withdraw','final_angle':door_angle(r),'samples':release_trace},indent=2)+'\n')


def whole_robot_min_x(runtime):
    m,d=runtime.sim.model,runtime.sim.data;minimum=math.inf
    for geom in range(m.ngeom):
        if int(m.geom_bodyid[geom]) not in runtime._robot:continue
        if not (m.geom_contype[geom] or m.geom_conaffinity[geom]):continue
        rotation=d.geom_xmat[geom].reshape(3,3);bounds=m.geom_aabb[geom]
        center=d.geom_xpos[geom]+rotation@bounds[:3]
        minimum=min(minimum,float(center[0]-np.abs(rotation[0])@bounds[3:]))
    return minimum


def execute_contact_task(runtime,metadata):
    """Full five-metre task with the same door/contact/whole-entry conditions."""
    from robot_exam.navigation.grid import OccupancyGrid
    from robot_exam.tasks.door_latch import dock_station
    r=runtime;grid=OccupancyGrid(bounds=(-6,4,-3,3))
    for box in metadata['obstacles']:grid.rectangle(*box)
    r.hold(.6);r.transition(State.NAVIGATE,'contact_five_metre_approach')
    r.navigate([-1.2,.42,.59],grid)
    r.dock(arc_station(metadata,door_angle(r),False),timeout=45,tolerance=.030)
    opening=run_contact_phase(r,metadata,1.56,False)
    r.transition(State.APPROACH,'contact_open_release_stow',timeout=100)
    release_and_stow(r,metadata,False)
    r.base.command(-.08,0);r.hold(2.);r.base.command(0,0);r.hold(.4)
    r.transition(State.NAVIGATE,'contact_enter_room')
    passage=OccupancyGrid(bounds=(-6,4,-3,3))
    for box in metadata['obstacles']:passage.rectangle(*box)
    passage.rectangle(-1.6,0,-.85,-.73)
    r.navigate([1.,0.,0.],passage,timeout=180)
    entered_minimum=whole_robot_min_x(r)
    if entered_minimum<=.07:raise TaskFailure('not_entered','Whole robot has not cleared the door before closing')
    r.transition(State.NAVIGATE,'contact_return_to_inner_latch',timeout=200)
    goal=arc_station(metadata,door_angle(r),True)
    r.navigate(goal,grid,timeout=150)
    dock_station(r,goal,timeout=60,tolerance=.025)
    closing=run_contact_phase(r,metadata,0.,True)
    r.transition(State.APPROACH,'contact_close_release_stow',timeout=100)
    release_and_stow(r,metadata,True)
    # At the tangent closing pose, reverse motion goes directly into the room.
    # Avoid the old short reverse-towards-leaf connector and near-leaf turn.
    r.transition(State.NAVIGATE,'contact_straight_inside_clearance',timeout=50)
    heading=float(r.base.pose()[2])
    if math.cos(heading)>-.25:
        raise TaskFailure('clearance_geometry','Reverse clearance does not point away from the leaf')
    start=float(r.sim.data.time)
    while r.base.pose()[0]<1.10:
        if r.sim.data.time-start>35.:raise TaskFailure('navigation_timeout','Straight room clearance timed out')
        error=wrap_angle(heading-r.base.pose()[2]);velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
        angular=float(np.clip(4.*error-.55*velocity[5]+(.45*math.copysign(1.,error) if abs(error)>.012 else 0.),-.65,.65))
        r.base.command(-.06,angular);r.tick()
        if r.collision_events:raise TaskFailure('base_collision','Unexpected collision during straight clearance')
    r.base.command(0,0);r.hold(2.)
    velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
    stopped=np.linalg.norm(velocity[:2])<.02 and abs(velocity[5])<.03
    final=door_angle(r);minimum=whole_robot_min_x(r)
    success=abs(final)<.035 and r.base.pose()[0]>.7 and minimum>.07 and stopped and not r.collision_events
    return {'success':bool(success),'final_door_angle':final,'final_base':r.base.pose().tolist(),
            'whole_robot_min_x_m':minimum,'entered_before_close':True,'initial_entry_min_x_m':entered_minimum,
            'opening_contact':opening,'closing_contact':closing,
            'total_grasp_count':opening['grasp_count']+closing['grasp_count'],
            'safety_releases':opening['safety_releases']+closing['safety_releases'],
            'necessary_phase_releases':2,'final_stopped':bool(stopped),
            'mode':'Oracle tangent-circle wheel/arm expert; oblique physical pinch; safety-triggered regrasp',
            'scope':'Complete five-metre open-enter-reapproach-close-clearance task; unchanged scoring'}
