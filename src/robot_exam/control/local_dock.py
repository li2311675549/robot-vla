"""Physical execution of collision-checked local paths through normal wheel controls."""
import math
import numpy as np
import mujoco

from robot_exam.control.base import wrap_angle
from robot_exam.navigation.local_dock import plan_dock


class PoseClearance:
    """Read-only collision queries in separate MjData; never moves the live robot."""
    def __init__(self,r):
        self.r=r;self.model=r.sim.model;self.scratch=mujoco.MjData(self.model)
        self.scratch.qpos[:]=r.sim.data.qpos
        j=int(self.model.body_jntadr[r.base.body]);self.qadr=int(self.model.jnt_qposadr[j])
        self.z=float(r.sim.data.qpos[self.qadr+2]);self.cache={}
        self.robot=set(r._robot);self.payload_qadr=None
        self.origin=r.base.pose().copy()
        if r.payload and r.payload_body:
            body=self.model.body(r.payload_body).id;self.robot.add(body)
            joint=int(self.model.body_jntadr[body])
            if self.model.jnt_type[joint]==mujoco.mjtJoint.mjJNT_FREE:
                self.payload_qadr=int(self.model.jnt_qposadr[joint])
                self.payload_origin=r.sim.data.qpos[self.payload_qadr:self.payload_qadr+3].copy()
                self.payload_quaternion=r.sim.data.qpos[self.payload_qadr+3:self.payload_qadr+7].copy()
    def __call__(self,pose):
        if getattr(self,'grid',None) is not None and not self.grid.free(self.grid.cell(pose[:2])):
            return False
        key=tuple(np.round(pose,4))
        if key in self.cache:return self.cache[key]
        self.scratch.qpos[self.qadr:self.qadr+3]=[pose[0],pose[1],self.z]
        self.scratch.qpos[self.qadr+3:self.qadr+7]=[math.cos(pose[2]/2),0,0,math.sin(pose[2]/2)]
        if self.payload_qadr is not None:
            angle=pose[2]-self.origin[2];c,s=math.cos(angle),math.sin(angle)
            delta=self.payload_origin[:2]-self.origin[:2]
            # Collision-only proxy: translate the held body's real geometry
            # with the base-relative carry point; orientation remains world
            # fixed, matching tick()'s payload stabilizer. Live qpos untouched.
            self.scratch.qpos[self.payload_qadr:self.payload_qadr+2]=np.asarray(pose[:2])+[c*delta[0]-s*delta[1],s*delta[0]+c*delta[1]]
            if self.r.payload_rotation is None:
                delta_quat=np.array([math.cos(angle/2),0,0,math.sin(angle/2)])
                mujoco.mju_mulQuat(self.scratch.qpos[self.payload_qadr+3:self.payload_qadr+7],delta_quat,self.payload_quaternion)
        mujoco.mj_forward(self.model,self.scratch)
        result=True
        for contact in self.scratch.contact:
            if contact.dist>=-.0005:continue
            a,b=map(int,contact.geom);ba,bb=(int(self.model.geom_bodyid[g]) for g in (a,b))
            if (ba in self.robot)==(bb in self.robot):continue
            other=b if ba in self.robot else a
            name=self.model.geom(other).name or ''
            # Held fruit contact and supporting ground are expected. Other
            # external physical boundaries remain in every clearance query.
            other_body=self.model.body(int(self.model.geom_bodyid[other])).name
            if name=='floor' or other_body==self.r.payload_body:continue
            result=False;break
        self.cache[key]=result
        return result


def execute_local_dock(r,goal,timeout,tolerance,strategy):
    from robot_exam.tasks.runtime import TaskFailure
    goal=np.asarray(goal);start=float(r.sim.data.time);trace=[];plans=[];stable=0.
    previous_mode=getattr(r,'payload_docking',False);r.payload_docking=bool(r.payload)
    previous_precision=getattr(r,'precision_docking',False);r.precision_docking=True
    tracking=r.config.get('local_dock_tracking_version','axis_v1')
    primitive_pivot=tracking=='axis_pivot_v6'
    bounded_axis=tracking in ('axis_bounded_v5','axis_pivot_v6')
    anchored=tracking in ('anchored_v4','axis_bounded_v5','axis_pivot_v6')
    wheel_dofs=[int(r.sim.model.jnt_dofadr[int(r.sim.model.actuator_trnid[a,0])]) for a in r.base.ids]
    last_applied=r.wheel_applied.copy();dt=float(r.sim.model.opt.timestep)
    capture_goal=bool(r.config.get('local_dock_capture_goal_during_primitives',False))
    capture_stable=0.;turn_context=None
    class StrictGoalSettled(Exception):pass
    def command(v,w,mode):
        nonlocal last_applied,stable,capture_stable
        if capture_goal:
            pose=r.base.pose()
            if (np.linalg.norm(pose[:2]-goal[:2])<tolerance
                    and abs(wrap_angle(goal[2]-pose[2]))<.035):
                # Intermediate waypoints are a means to reach the final pose,
                # not a reason to drive away after physically arriving there.
                # Brake normally and independently check actual velocities.
                v=w=0.;mode='goal_capture'
        if r.config.get('local_dock_pi_turn',False) and mode.startswith('turn_') and turn_context is not None:
            anchor,target_yaw=turn_context
            r.base.rotate_velocity(anchor,target_yaw,body_yaw_limit=.25 if r.payload else .4)
            wheels=r.base.wheel_target
            v=float(r.base.wheel_radius*.5*(wheels[0]+wheels[1]));w=float(r.base.wheel_radius/r.base.wheel_track*(wheels[1]-wheels[0]))
        else:r.base.command(v,w)
        desired=r.sim.data.ctrl[r.base.ids].copy()
        if not r.payload:
            last_applied+=np.clip(desired-last_applied,-.8*dt,.8*dt)
            r.sim.data.ctrl[r.base.ids]=last_applied
        pose=r.base.pose();velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
        sampled=round(r.sim.data.time/dt)%10==0
        if sampled:
            trace.append({'time':float(r.sim.data.time),'mode':mode,'pose':pose.tolist(),
                'position_error_m':float(np.linalg.norm(pose[:2]-goal[:2])),
                'heading_error_rad':float(abs(wrap_angle(goal[2]-pose[2]))),
                'linear_speed_m_s':float(np.linalg.norm(velocity[:2])),'yaw_rate_rad_s':float(velocity[5]),
                'linear_command_m_s':float(v),'angular_command_rad_s':float(w),
                'wheel_target_rad_s':desired.tolist(),'wheel_before_payload_limits_rad_s':r.sim.data.ctrl[r.base.ids].tolist(),
                'wheel_velocity_rad_s':r.sim.data.qvel[wheel_dofs].tolist()})
        if r.sim.data.time-start>=timeout:raise TaskFailure('navigation_timeout','Local dock trajectory timed out')
        r.tick()
        if sampled:
            # tick() applies payload speed/rate limits AFTER base.command().
            # The old trace labelled the pre-limit request as the applied
            # wheel command; retain both timing points explicitly.
            trace[-1].update(wheel_applied_rad_s=r.sim.data.ctrl[r.base.ids].tolist(),
                             wheel_actuator_force_nm=r.sim.data.actuator_force[r.base.ids].tolist(),
                             control_after_tick_time=float(r.sim.data.time))
        if r.collision_events:raise TaskFailure('base_collision','Collision during local dock path')
        if capture_goal:
            pose=r.base.pose();velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
            valid=(np.linalg.norm(pose[:2]-goal[:2])<tolerance
                   and abs(wrap_angle(goal[2]-pose[2]))<.035
                   and np.linalg.norm(velocity[:2])<.02 and abs(velocity[5])<.03)
            capture_stable=capture_stable+dt if valid else 0.
            if capture_stable>=2.:
                stable=capture_stable
                raise StrictGoalSettled()
    def steer(target_yaw):
        pose=r.base.pose();err=wrap_angle(target_yaw-pose[2]);velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
        # Same tire-scrub compensation mechanism as the production controller.
        w=float(np.clip(1.8*err-.55*velocity[5],-.55,.55))
        if abs(err)>.018:w+=math.copysign(float(r.config.get('local_dock_angular_feedforward',.35 if r.payload else .45)),err)
        limit=float(r.config.get('local_dock_angular_limit_rad_s',.85))
        return float(np.clip(w,-limit,limit))
    def brake():
        # A slow payload wheel limiter retains a common-mode wheel command
        # while switching from driving to yaw. Drain it first; otherwise a
        # nominal in-place primitive translates tens of centimetres.
        while True:
            velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
            if (np.max(np.abs(r.wheel_applied))<.025 and
                    np.linalg.norm(velocity[:2])<.008 and abs(velocity[5])<.02):break
            command(0.,0.,'brake')
    try:
        if r.config.get('local_dock_preserve_settled_entry',False):
            # A recovered grasp can already be within the ORIGINAL dock
            # tolerances. Planning a line to a residual of a few millimetres
            # introduced a needless large bearing turn and destroyed the
            # valid pose. Verify two real stopped seconds before planning.
            pose=r.base.pose();velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
            at_goal=(np.linalg.norm(pose[:2]-goal[:2])<tolerance and
                     abs(wrap_angle(goal[2]-pose[2]))<.035 and
                     np.linalg.norm(velocity[:2])<.02 and abs(velocity[5])<.03)
            if at_goal:
                while stable<2.:
                    command(0.,0.,'entry_settle')
                    pose=r.base.pose();velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
                    valid=(np.linalg.norm(pose[:2]-goal[:2])<tolerance and
                           abs(wrap_angle(goal[2]-pose[2]))<.035 and
                           np.linalg.norm(velocity[:2])<.02 and abs(velocity[5])<.03)
                    if not valid:
                        stable=0.;break
                    stable+=dt
                if stable>=2.:return
        # At most two physically executed replans; no unlimited residual loops.
        for attempt in range(2):
            clearance=PoseClearance(r)
            grid=getattr(r,'grid',None)
            if anchored and grid is not None and grid.free(grid.cell(goal)) and grid.free(grid.cell(r.base.pose())):
                clearance.grid=grid
            try:plan=plan_dock(r.base.pose(),goal,strategy,clearance)
            except RuntimeError as exc:raise TaskFailure('path_planning',str(exc)) from exc
            plans.append(plan)
            for a,b in zip(plan['poses'][:-1],plan['poses'][1:]):
                a,b=np.asarray(a),np.asarray(b);turn=np.linalg.norm(a[:2]-b[:2])<1e-5
                if anchored:brake()
                if turn:
                    anchor=r.base.pose()[:2].copy()
                    turn_context=(anchor,float(b[2]))
                    while True:
                        velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
                        if abs(wrap_angle(b[2]-r.base.pose()[2]))<.018 and abs(velocity[5])<.03:break
                        pose=r.base.pose();heading=np.array([math.cos(pose[2]),math.sin(pose[2])])
                        drift=float((pose[:2]-anchor)@heading)
                        # A pivot primitive cannot independently repair XY.
                        # Fore/aft anchor corrections alternate wheel common
                        # mode and can stall a skid-steer turn under load.
                        # The optional version leaves XY to the next planned
                        # translation/replan, with the same final tolerances.
                        v=float(np.clip(-1.5*drift-.7*(velocity[:2]@heading),-.045,.045)) if anchored and not primitive_pivot else 0.
                        command(v,steer(b[2]),f'turn_{attempt}')
                else:
                    while True:
                        pose=r.base.pose();delta=b[:2]-pose[:2]
                        heading=np.array([math.cos(b[2]),math.sin(b[2])]);along=float(delta@heading)
                        lateral=float(delta@[-heading[1],heading[0]])
                        velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
                        arrived=(np.linalg.norm(delta)<.016 if anchored and not bounded_axis else abs(along)<.010)
                        if arrived and np.linalg.norm(velocity[:2])<.02:break
                        direction=1 if (b[:2]-a[:2])@heading>=0 else -1
                        # Use the measured post-turn start, not the planned
                        # point; tire scrub can shift it. Retain the terminal
                        # axis only inside the final short approach region.
                        bearing=math.atan2(delta[1],delta[0])+(math.pi if direction<0 else 0.)
                        look_heading=(bearing if anchored and not bounded_axis else b[2]+float(np.clip(direction*2.0*lateral,-.18,.18)))
                        yaw=wrap_angle(look_heading-pose[2]);w=steer(look_heading) if abs(yaw)>.018 else -.55*velocity[5]
                        limit=float(r.config.get('local_dock_linear_speed_m_s',.075 if r.payload else .11))
                        measured_heading=np.array([math.cos(pose[2]),math.sin(pose[2])])
                        v=(float(direction*np.clip(.8*np.linalg.norm(delta)-.55*direction*(velocity[:2]@measured_heading),0,limit))
                           if anchored and not bounded_axis else float(np.clip(.9*along-.55*(velocity[:2]@heading),-limit,limit)))
                        if abs(yaw)>.22:v=0.
                        command(v,w,f'drive_{attempt}')
            # Original acceptance, checked continuously and again by the paired
            # harness after returning. Planning tolerances never replace it.
            window=float(r.sim.data.time)
            while r.sim.data.time-window<4.:
                pose=r.base.pose();velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
                inside=np.linalg.norm(pose[:2]-goal[:2])<tolerance and abs(wrap_angle(goal[2]-pose[2]))<.035
                stopped=np.linalg.norm(velocity[:2])<.02 and abs(velocity[5])<.03
                stable=stable+dt if inside and stopped else 0.
                command(0,0,'settle')
                if stable>=2.:return
            r.retries+=1
        raise TaskFailure('navigation_accuracy','Local path ended outside strict stable dock envelope')
    except StrictGoalSettled:
        return
    finally:
        r.base.command(0,0);r.payload_docking=previous_mode;r.precision_docking=previous_precision
        r.dock_traces.append({'controller':strategy+'_stage10','tracking_version':tracking,'goal':goal.tolist(),
            'position_tolerance_m':tolerance,'heading_tolerance_rad':.035,'stable_seconds':stable,
            'elapsed_seconds':float(r.sim.data.time-start),'plans':plans,'samples':trace,
            'final_pose':r.base.pose().tolist()})
        if r.output:
            import json
            r.output.mkdir(parents=True,exist_ok=True)
            (r.output/'dock.json').write_text(json.dumps(r.dock_traces,indent=2))
