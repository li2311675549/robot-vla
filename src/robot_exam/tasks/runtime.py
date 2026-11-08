"""Shared bounded task runtime, state transitions and physical motion primitives."""
import enum
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from robot_exam.control.arm import ArmController, GripperController
from robot_exam.control.base import BaseController, wrap_angle
from robot_exam.control.grasp_monitor import relative_grasp_pose
from robot_exam.tasks.contracts import CONTRACTS


class State(str, enum.Enum):
    INITIALIZE = "INITIALIZE"
    NAVIGATE = "NAVIGATE"
    OBSERVE = "OBSERVE"
    SELECT_TARGET = "SELECT_TARGET"
    PLAN = "PLAN"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    MANIPULATE = "MANIPULATE"
    PLACE = "PLACE"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"
    RETURN_HOME = "RETURN_HOME"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class TaskFailure(RuntimeError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


class TaskRuntime:
    STOW = np.array([0, -2.0, 1.2, -.77, -1.57, 0])

    def __init__(self, sim, output=None, callback=None, wall_limit=1800, sim_limit=1800):
        self.sim = sim
        self.arm = ArmController(sim)
        self.gripper = GripperController(sim)
        self.grip_force_feedback=None
        self.base = BaseController(sim)
        self.qgoal = self.arm.reference.copy()
        self.callback = callback
        self.output = Path(output) if output else None
        self.transitions, self.rows = [], []
        self.state = State.INITIALIZE
        self.state_started = sim.data.time
        self.state_timeout = CONTRACTS['INITIALIZE'][3]
        self.stage = "initialize"
        self.wall_start = time.monotonic()
        self.wall_limit, self.sim_limit = wall_limit, sim_limit
        self.data_logger=None
        self.payload=False
        self.payload_body=None
        self.payload_rotation=None
        self.payload_local_xy=None
        self.payload_height=None
        self.payload_relative=None
        self.payload_world_relative=None
        self.payload_relative_rotation=None
        self.payload_monitor_count=0
        self.payload_trace=[]
        self.payload_trace_history=[]
        self.payload_faults=[]
        self.ik_diagnostics=[]
        self.base_hold_anchor=None
        self.parking=None
        self.parking_history=[]
        self.wheel_applied=sim.data.ctrl[self.base.ids].copy()
        self.retries = 0
        self.navigation = []
        self.dock_traces=[]
        self.config={}
        self.collision_events = []
        self.dynamic_bodies=[i for i in range(sim.model.nbody) if sim.model.body_jntnum[i] and
                             (sim.model.body(i).name or '').startswith(('fruit_','book_','door_leaf','drawer_'))]
        self._contact_pairs = set()
        self._robot = set()
        husky = sim.model.body("husky").id
        first=int(sim.model.body_jntadr[husky]);count=int(sim.model.body_jntnum[husky])
        free=next((j for j in range(first,first+count) if sim.model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE),None)
        if free is None:raise RuntimeError('Husky free-joint velocity is required for physical docking')
        self.base_velocity_dof=int(sim.model.jnt_dofadr[free])
        for i in range(sim.model.nbody):
            p = i
            while p > 0 and p != husky:
                p = sim.model.body_parentid[p]
            if p == husky:
                self._robot.add(i)

    def transition(self, state, reason, target=None, timeout=None, recovery="bounded retry or FAILURE"):
        self.state = State(state)
        self.state_started = self.sim.data.time
        self.state_timeout = timeout if timeout is not None else CONTRACTS[self.state.value][3]
        self.stage = reason
        self.transitions.append({"time": self.sim.data.time, "state": self.state.value,
                                 "reason": reason, "target": target, "timeout_seconds": self.state_timeout,
                                 "recovery": recovery})
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "fsm.json").write_text(json.dumps(self.transitions, ensure_ascii=False, indent=2))
            checkpoint=np.empty(mujoco.mj_stateSize(self.sim.model,mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(self.sim.model,self.sim.data,checkpoint,mujoco.mjtState.mjSTATE_INTEGRATION)
            directory=self.output/'checkpoints';directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/f'{len(self.transitions):04d}.npz',integration_state=checkpoint,
                                state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),time=self.sim.data.time)

    def tick(self):
        if self.sim.data.time-self.state_started > self.state_timeout:
            raise TaskFailure("state_timeout", f"{self.state.value}/{self.stage} exceeded {self.state_timeout}s")
        if self.sim.data.time > self.sim_limit or time.monotonic()-self.wall_start > self.wall_limit:
            raise TaskFailure("task_timeout", "Global task budget exceeded")
        if self.base_hold_anchor is not None:
            self.base.rotate(self.base_hold_anchor[:2],self.base_hold_anchor[2])
        # A fixed UR5 joint target rotates with the Husky.  During fruit
        # transport this turns a vertical pinch sideways whenever A* steers.
        # Re-solve a base-relative position with a world-fixed tool orientation
        # so the grasp does not rotate with the chassis.
        if self.payload and self.payload_body and self.payload_rotation is not None:
            if round(self.sim.data.time / self.sim.model.opt.timestep) % 10 == 0:
                pose=self.base.pose();c,s=math.cos(pose[2]),math.sin(pose[2])
                offset=self.payload_local_xy
                target=np.array([pose[0]+c*offset[0]-s*offset[1],
                                 pose[1]+s*offset[0]+c*offset[1],self.payload_height])
                try:
                    self.qgoal,_=self.solve_ik(target,self.payload_rotation,seed=self.qgoal,max_iterations=120,context='payload_world_stabilization')
                except RuntimeError as exc:
                    self.base.command(0,0)
                    raise TaskFailure('payload_ik','World-stabilized carry pose became unreachable') from exc
        if self.grip_force_feedback:
            from robot_exam.control.contact_grip import update_contact_feedback
            update_contact_feedback(self)
        arm_speed=float(self.config.get('payload_arm_speed_rad_s',1.2)) if self.payload else .7
        self.arm.command(self.qgoal,max_speed=arm_speed)
        wheel_acceleration=np.zeros(4)
        if self.payload:
            # Long, curved bananas are sensitive to yaw and acceleration
            # disturbances even when both fingers carry substantial normal
            # force. Use a slower physical carry profile for that geometry;
            # this changes only wheel commands, never object state/constraints.
            profile=getattr(self,'payload_motion_profile','default')
            linear_limit=float(self.config.get(
                'payload_banana_linear_speed_m_s' if profile=='banana' else 'payload_linear_speed_m_s',
                .055 if profile=='banana' else .10))
            angular_key=('payload_dock_angular_speed_rad_s' if getattr(self,'payload_docking',False)
                         else 'payload_banana_angular_speed_rad_s' if profile=='banana'
                         else 'payload_angular_speed_rad_s')
            angular_default=.45 if getattr(self,'payload_docking',False) else (.10 if profile=='banana' else .25)
            angular_limit=float(self.config.get(angular_key,angular_default))
            desired=getattr(self.base,'wheel_target',np.zeros(4))
            # Limit yaw separately from translation: clipping each wheel by one
            # scalar made the base unable to clear its staging pose.
            radius=self.base.wheel_radius;track=self.base.wheel_track
            linear=radius*.5*(desired[0]+desired[1])
            angular=radius/track*(desired[1]-desired[0])
            precision_turn=getattr(self.base,'precision_turn',False)
            strict_pivot=bool(self.config.get('payload_strict_pivot_turn',False) and
                              precision_turn and profile!='banana' and profile in
                              self.config.get('payload_strict_pivot_categories',[profile]))
            if strict_pivot:
                # During route heading alignment a differential-drive base
                # cannot cancel lateral error. The old anchor correction
                # plus turn creep could orbit a waypoint indefinitely.
                # Reacquire translation only after the heading aligns.
                linear=0.
            banana_turn=(profile=='banana' and (precision_turn or abs(linear)<.02))
            other_pure_turn=(profile!='banana' and abs(linear)<1e-6)
            if (not getattr(self,'payload_docking',False)
                    and (banana_turn or other_pure_turn) and abs(angular)>.02):
                if banana_turn:
                    # The former .035 m/s creep translated the chassis while
                    # its .035 rad/s yaw target could not overcome tire scrub.
                    # A bounded differential wheel target turns the real
                    # skid-steer wheels without walking into the worktable.
                    angular_limit=max(angular_limit,.45)
                    angular=np.sign(angular)*min(angular_limit,abs(angular)+.35)
                    linear=float(np.clip(linear,-.02,.02)) if precision_turn else 0.
                elif not strict_pivot:
                    linear=float(self.config.get('payload_turn_creep_m_s',.035))
            linear=np.clip(linear,-linear_limit,linear_limit)
            angular=np.clip(angular,-angular_limit,angular_limit)
            left=(linear-angular*track/2)/radius;right=(linear+angular*track/2)/radius
            desired=np.array([left,right,left,right])
            acceleration=float(self.config.get(
                'payload_banana_wheel_acceleration_rad_s2' if profile=='banana' else 'payload_wheel_acceleration_rad_s2',
                .35 if profile=='banana' else .8))
            increment=acceleration*self.sim.model.opt.timestep
            previous_wheels=self.wheel_applied.copy()
            if (getattr(self,'payload_docking',False) and self.config.get('payload_precision_twist_slew',False)
                    and profile in self.config.get('payload_twist_slew_categories',[profile])):
                # Independent wheel ramps retain forward velocity while the
                # outer pose loop requests a pivot. Brake translation as a
                # common-mode signal, independent of differential yaw slew.
                previous_linear=radius*.5*(previous_wheels[0]+previous_wheels[1])
                previous_angular=radius/track*(previous_wheels[1]-previous_wheels[0])
                braking=(linear*previous_linear<=0 or abs(linear)<abs(previous_linear))
                linear_acceleration=.20 if braking else .06
                applied_linear=previous_linear+np.clip(linear-previous_linear,
                    -linear_acceleration*self.sim.model.opt.timestep,linear_acceleration*self.sim.model.opt.timestep)
                applied_angular=previous_angular+np.clip(angular-previous_angular,
                    -.8*self.sim.model.opt.timestep,.8*self.sim.model.opt.timestep)
                left=(applied_linear-applied_angular*track/2)/radius
                right=(applied_linear+applied_angular*track/2)/radius
                self.wheel_applied=np.array([left,right,left,right])
            else:self.wheel_applied+=np.clip(desired-self.wheel_applied,-increment,increment)
            wheel_acceleration=(self.wheel_applied-previous_wheels)/self.sim.model.opt.timestep
            self.sim.data.ctrl[self.base.ids]=self.wheel_applied
        else:self.wheel_applied=self.sim.data.ctrl[self.base.ids].copy()
        if self.data_logger:self.data_logger.capture_pre_step(self.stage,arm_target=self.qgoal,
            arm_reference=self.arm.reference,max_speed=arm_speed)
        self.sim.step(self.sim.data.ctrl)
        if self.payload and self.payload_body and round(self.sim.data.time / self.sim.model.opt.timestep) % 10 == 0:
            body_id=self.sim.model.body(self.payload_body).id
            tool_position,tool_rotation=self.arm.pose()
            object_rotation=self.sim.data.xmat[body_id].reshape(3,3)
            relative,relative_rotation=relative_grasp_pose(tool_position,tool_rotation,
                self.sim.data.xpos[body_id],object_rotation)
            drift=float(np.linalg.norm(relative-self.payload_relative))
            bilateral=self.bilateral(self.payload_body)
            world_relative=self.sim.data.xpos[body_id]-tool_position
            world_drift=float(np.linalg.norm(world_relative-self.payload_world_relative))
            cosine=np.clip((np.trace(self.payload_relative_rotation.T@relative_rotation)-1)/2,-1,1)
            angular_drift=float(np.arccos(cosine))
            contacts=self.contacts(self.payload_body)
            forces={'left':float(sum(c['normal_force_n'] for c in contacts if c['body'].startswith('grip_left'))),
                    'right':float(sum(c['normal_force_n'] for c in contacts if c['body'].startswith('grip_right')))}
            joint=int(self.sim.model.body_jntadr[body_id]);object_dof=int(self.sim.model.jnt_dofadr[joint])
            base_velocity=self.sim.data.qvel[self.base_velocity_dof:self.base_velocity_dof+6].copy()
            base_acceleration=(base_velocity-self.payload_previous_base_velocity)/.02
            self.payload_previous_base_velocity=base_velocity
            finger_positions={name:float(self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint(name).id]])
                              for name in ('grip_left_driver_joint','grip_right_driver_joint')}
            trace_period=float(self.config.get('payload_trace_interval_seconds',.02))
            if len(self.payload_trace)<100000 and (not self.payload_trace or self.sim.data.time-self.payload_trace[-1]['time']>=trace_period-1e-9):
                self.payload_trace.append({'time':float(self.sim.data.time),'relative_error_m':drift,
                    'relative_position_frame':'grip_pinch',
                    'object_position_tool_m':relative.tolist(),
                    'gripper_rotation_world':tool_rotation.tolist(),
                    'legacy_world_offset_error_m':world_drift,
                    'relative_rotation_error_rad':angular_drift,'bilateral_contact':bool(bilateral),
                    'finger_contacts':contacts,'finger_driver_qpos_rad':finger_positions,
                    'left_normal_force_n':forces['left'],'right_normal_force_n':forces['right'],
                    'gripper_closure':float(self.sim.data.ctrl[self.gripper.actuator]/255),
                    'arm_joint_goal_error_rad':float(np.linalg.norm(self.qgoal-self.sim.data.qpos[self.arm.qadr])),
                    'arm_reference_error_rad':float(np.linalg.norm(self.arm.reference-self.sim.data.qpos[self.arm.qadr])),
                    'base_linear_acceleration_m_s2':base_acceleration[:2].tolist(),
                    'base_yaw_acceleration_rad_s2':float(base_acceleration[5]),
                    'object_linear_speed_m_s':float(np.linalg.norm(self.sim.data.qvel[object_dof:object_dof+3])),
                    'object_angular_speed_rad_s':float(np.linalg.norm(self.sim.data.qvel[object_dof+3:object_dof+6])),
                    'base_pose':self.base.pose().tolist(),
                    'wheel_command':self.sim.data.ctrl[self.base.ids].tolist(),
                    'wheel_acceleration_rad_s2':wheel_acceleration.tolist(),
                    'object_position':self.sim.data.xpos[body_id].tolist(),'gripper_position':self.arm.pose()[0].tolist()})
            self.payload_monitor_count=self.payload_monitor_count+1 if (drift>.025 or angular_drift>.35 or not bilateral) else 0
            if self.payload_monitor_count>=3:
                self.payload_faults.append({'time':float(self.sim.data.time),'body':self.payload_body,
                    'translation_error_m':drift,'rotation_error_rad':angular_drift,'bilateral_contact':bool(bilateral),
                    'trigger_dimensions':{'translation':bool(drift>.025),'rotation':bool(angular_drift>.35),
                                          'contact_loss':bool(not bilateral)},
                    'consecutive_20ms_violations':self.payload_monitor_count,'base_pose':self.base.pose().tolist(),
                    'object_position':self.sim.data.xpos[body_id].tolist(),'left_normal_force_n':forces['left'],
                    'right_normal_force_n':forces['right']})
                self.write_trace('payload_faults',self.payload_faults)
                self.base.command(0,0)
                raise TaskFailure('gripper_slip',f'In-motion grasp witness lost: tool-frame relative error {drift:.4f}m, rotation={angular_drift:.4f}rad, bilateral={bilateral}')
        if round(self.sim.data.time/self.sim.model.opt.timestep) % 25 == 0:
            self.check_collisions()
            log_due=(not self.rows or self.sim.data.time-self.rows[-1]['time']>=float(self.config.get('trajectory_log_interval_seconds',.05))-1e-9)
            if log_due:self.rows.append({"time": self.sim.data.time, "state": self.state.value, "stage": self.stage,
                              "base": self.base.pose().tolist(), "tip": self.arm.pose()[0].tolist(),
                              "arm_q": self.sim.data.qpos[self.arm.qadr].tolist(),
                              "arm_qvel": self.sim.data.qvel[self.arm.dofs].tolist(),
                              "ctrl": self.sim.data.ctrl.tolist()})
            if log_due:
                self.rows[-1]['dynamic_objects']={self.sim.model.body(i).name:self.sim.data.xpos[i].tolist() for i in self.dynamic_bodies}
                self.rows[-1]['passive_joints']={name:float(self.sim.data.qpos[self.sim.model.jnt_qposadr[j]])
                    for j in range(self.sim.model.njnt) for name in [self.sim.model.joint(j).name]
                    if name=='door_hinge' or (name and name.startswith('drawer_slide_'))}
        if self.callback:
            self.callback(self.sim, self.state.value + "_" + self.stage)

    def check_collisions(self):
        current = set()
        for contact in self.sim.data.contact:
            a, b = map(int, contact.geom)
            ba, bb = int(self.sim.model.geom_bodyid[a]), int(self.sim.model.geom_bodyid[b])
            if (ba in self._robot) == (bb in self._robot):
                continue
            other = b if ba in self._robot else a
            robot_geom = a if ba in self._robot else b
            name = self.sim.model.geom(other).name or ""
            robot_name=self.sim.model.body(int(self.sim.model.geom_bodyid[robot_geom])).name or ''
            other_body=self.sim.model.body(int(self.sim.model.geom_bodyid[other])).name or ''
            forbidden=(name.startswith(("wall", "door_frame", "obstacle", "cabinet_shell", "table_leg"))
                       or (name.endswith('_top') and not robot_name.startswith('grip_'))
                       or (self.state in (State.NAVIGATE,State.RETURN_HOME) and other_body=='door_leaf'))
            if forbidden:
                pair = tuple(sorted([a, b]))
                current.add(pair)
                if pair not in self._contact_pairs and contact.dist < -.001:
                    self.collision_events.append({"time": self.sim.data.time, "geoms": [a, b], "name": name, "penetration_m": -float(contact.dist)})
        self._contact_pairs = current

    def hold(self, seconds):
        for _ in range(round(seconds/self.sim.model.opt.timestep)):
            self.tick()

    def solve_ik(self,position,rotation,seed=None,max_iterations=300,context=None):
        try:
            q,metrics=self.arm.solve(position,rotation,seed=seed,max_iterations=max_iterations)
            if not metrics.get('approximate_convergence',False):return q,metrics
            # Search other branches before accepting the legacy bounded
            # approximation. The convergence thresholds themselves stay fixed.
            approximate=(q,metrics)
            try:
                multi_q,multi_metrics=self.arm.solve_multi(position,rotation,seed=seed,max_iterations=max(max_iterations,500))
                if not multi_metrics.get('approximate_convergence',False):
                    self.ik_diagnostics.append({"context":context,"first_branch_approximate":metrics,
                                                "multi_branch":self.arm.last_solve_report,"recovered":True})
                    if self.output:
                        self.output.mkdir(parents=True,exist_ok=True)
                        self.write_trace('ik_diagnostics',self.ik_diagnostics)
                    return multi_q,multi_metrics
            except RuntimeError:pass
            return approximate
        except RuntimeError as first:
            failed=dict(self.arm.last_solve_report)
            try:
                q,metrics=self.arm.solve_multi(position,rotation,seed=seed,max_iterations=max(max_iterations,500))
                self.ik_diagnostics.append({"context":context,"first_branch_failure":failed,
                                            "multi_branch":self.arm.last_solve_report,"recovered":True})
                if self.output:
                    self.output.mkdir(parents=True,exist_ok=True)
                    self.write_trace('ik_diagnostics',self.ik_diagnostics)
                return q,metrics
            except RuntimeError as second:
                entry={"context":context,"target_position":np.asarray(position).tolist(),
                       "target_rotation":np.asarray(rotation).tolist(),
                       "measured_joint_position_rad":self.sim.data.qpos[self.arm.qadr].tolist(),
                       "reference_joint_position_rad":self.arm.reference.tolist(),
                       "joint_limits_rad":self.sim.model.jnt_range[self.arm.joints].tolist(),
                       "first_branch_failure":failed,"multi_branch":self.arm.last_solve_report,
                       "failure_reason":str(second)}
                self.ik_diagnostics.append(entry)
                if self.output:
                    self.output.mkdir(parents=True,exist_ok=True)
                    self.write_trace('ik_diagnostics',self.ik_diagnostics)
                raise

    def joint_move(self, q, seconds=3., _collision_checked=False):
        self._ensure_operation_hold()
        if self.config.get('collision_planned_joint_moves',False) and not _collision_checked:
            from robot_exam.control.joint_rrt import plan_joint_rrt
            allowed=[self.sim.model.body(i).name for i in self.dynamic_bodies
                     if self.bilateral(self.sim.model.body(i).name)]
            path,report=plan_joint_rrt(self.arm,q,allowed_bodies=allowed)
            self.ik_diagnostics.append({'context':self.stage+':joint_path','planning':report})
            for waypoint in path:self.joint_move(waypoint,seconds,_collision_checked=True)
            return
        q=np.asarray(q).copy()
        start=self.sim.data.qpos[self.arm.qadr].copy()
        seconds=max(seconds,1.6*float(np.max(np.abs(q-start)))/.7)
        steps=max(1,round(seconds/self.sim.model.opt.timestep))
        # Synchronize all six joints along the same path. Independent saturated
        # joint ramps can sweep through the chassis even when the straight
        # configuration-space segment is collision-free.
        for i in range(steps):
            f=(i+1)/steps;f=f*f*(3-2*f)
            self.qgoal=start+f*(q-start)
            self.tick()
        self.qgoal=q;self.hold(.8)
        error = float(np.max(np.abs(self.sim.data.qpos[self.arm.qadr]-q)))
        if error > .06:
            raise TaskFailure("arm_tracking", f"Joint tracking error {error:.4f}rad")

    def stow(self):
        self.gripper.command(0)
        self.joint_move(self.STOW, 3.5)

    def move(self, position, rotation=None, seconds=2., tolerance=.018):
        self._ensure_operation_hold()
        mujoco.mj_forward(self.sim.model,self.sim.data)
        rotation = np.diag([1., -1., -1.]) if rotation is None else np.asarray(rotation)
        start, start_r = self.arm.pose()
        steps = round(seconds/self.sim.model.opt.timestep)
        # First solve full target orientation; interpolation below is positional.
        try:
            final_q, _ = self.solve_ik(position, rotation, seed=self.qgoal,context=f"{self.stage}:endpoint")
        except RuntimeError as exc:
            raise TaskFailure("ik_unreachable", str(exc)) from exc
        if (self.config.get('collision_planned_reorientation',False) and
                np.arccos(np.clip((np.trace(start_r.T@rotation)-1)/2,-1,1))>.20):
            from robot_exam.control.joint_rrt import plan_joint_rrt
            allowed=[self.sim.model.body(i).name for i in self.dynamic_bodies if self.bilateral(self.sim.model.body(i).name)]
            failures=[]
            for seed in (None,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
                try:
                    q=final_q if seed is None else self.arm.solve(position,rotation,seed=seed,max_iterations=500)[0]
                    path,planning=plan_joint_rrt(self.arm,q,allowed_bodies=allowed)
                    for waypoint in path:self.joint_move(waypoint,seconds,_collision_checked=True)
                    error=float(np.linalg.norm(self.arm.pose()[0]-position))
                    if error>tolerance:raise TaskFailure('arm_tracking',f'Collision-planned reorientation endpoint {error:.4f}m')
                    self.ik_diagnostics.append({'context':self.stage+':reorientation','planning':planning})
                    return
                except TaskFailure:raise
                except RuntimeError as exc:failures.append(str(exc))
            raise TaskFailure('arm_collision_path',' | '.join(failures))
        seconds=max(seconds,1.8*float(np.max(np.abs(final_q-self.arm.reference)))/.7)
        steps=round(seconds/self.sim.model.opt.timestep)
        for i in range(steps):
            if i % 20 == 0:
                f = min(1., (i+20)/steps)
                f = f*f*(3-2*f)
                goal = start + f*(np.asarray(position)-start)
                # Quaternion interpolation avoids an instantaneous 90-degree
                # orientation request during side grasp transitions.
                qa, qb, qr = np.empty(4), np.empty(4), np.empty(4)
                mujoco.mju_mat2Quat(qa, start_r.ravel())
                mujoco.mju_mat2Quat(qb, rotation.ravel())
                if qa@qb < 0:
                    qb *= -1
                qr[:] = (1-f)*qa + f*qb
                qr /= np.linalg.norm(qr)
                mat = np.empty(9)
                mujoco.mju_quat2Mat(mat, qr)
                try:
                    self.qgoal, _ = self.solve_ik(goal, mat.reshape(3, 3), seed=self.qgoal, max_iterations=160,context=f"{self.stage}:cartesian_waypoint")
                except RuntimeError as exc:
                    raise TaskFailure("ik_unreachable", str(exc)) from exc
            self.tick()
        # The mobile base can settle under arm reaction forces. Recompute in
        # the current base frame instead of restoring a stale pre-motion IK q.
        for j in range(round(.45/self.sim.model.opt.timestep)):
            if j%20==0:
                self.qgoal,_=self.solve_ik(position,rotation,seed=self.qgoal,max_iterations=160,context=f"{self.stage}:settle")
            self.tick()
        error = float(np.linalg.norm(self.arm.pose()[0]-position))
        max_corrections=round(float(self.config.get('arm_endpoint_convergence_seconds',8.0))/.04)
        correction_history=[]
        for correction in range(max_corrections):
            if error<=tolerance:break
            # Re-seed from measured joints to avoid repeatedly returning the
            # same near-singular branch while the physical arm lags behind it.
            self.qgoal,_=self.solve_ik(position,rotation,seed=self.sim.data.qpos[self.arm.qadr].copy(),
                                       max_iterations=180,context=f"{self.stage}:endpoint_feedback_{correction}")
            self.hold(.04)
            error=float(np.linalg.norm(self.arm.pose()[0]-position))
            if correction%10==0:correction_history.append({"seconds":(correction+1)*.04,"position_error_m":error})
        if error > tolerance:
            report={"context":self.stage,"target_position":np.asarray(position).tolist(),
                    "target_rotation":rotation.tolist(),"actual_position":self.arm.pose()[0].tolist(),
                    "position_error_m":error,"tolerance_m":tolerance,
                    "measured_joint_position_rad":self.sim.data.qpos[self.arm.qadr].tolist(),
                    "reference_joint_position_rad":self.arm.reference.tolist(),
                    "correction_history":correction_history,"failure_reason":"physical_endpoint_tracking_timeout"}
            self.ik_diagnostics.append(report)
            if self.output:
                self.output.mkdir(parents=True,exist_ok=True)
                self.write_trace('ik_diagnostics',self.ik_diagnostics)
            raise TaskFailure("arm_tracking", f"Cartesian error {error:.4f}m at {self.stage}; target={np.asarray(position).tolist()} actual={self.arm.pose()[0].tolist()}")
        return error

    def contacts(self, target_body):
        target = self.sim.model.body(target_body).id
        contacts = []
        for i, c in enumerate(self.sim.data.contact):
            a, b = [int(self.sim.model.geom_bodyid[g]) for g in c.geom]
            if a != target and b != target:
                continue
            other = b if a == target else a
            name = self.sim.model.body(other).name
            if name.startswith("grip_"):
                force = np.empty(6)
                mujoco.mj_contactForce(self.sim.model, self.sim.data, i, force)
                point=np.asarray(c.pos,dtype=float)
                normal=np.asarray(c.frame,dtype=float).reshape(3,3)[0]
                tangential=float(np.linalg.norm(force[1:3]))
                # The contact-frame normal may point toward either geometry;
                # torque magnitude is invariant to that sign convention.
                world_force=np.asarray(c.frame,dtype=float).reshape(3,3).T@force[:3]
                lever=point-self.sim.data.xipos[target]
                contacts.append({"body": name, "normal_force_n": float(force[0]),
                    "tangential_force_n":tangential,
                    "friction_utilization":tangential/max(float(force[0]),1e-6),
                    "position_world_m":point.tolist(),"normal_world":normal.tolist(),
                    "lever_arm_m":lever.tolist(),
                    "torque_about_object_com_nm":float(np.linalg.norm(np.cross(lever,world_force)))})
        return contacts

    def bilateral(self, body):
        return self.gripper.bilateral(self.contacts(body))

    def stable_grasp(self, body, seconds=.5):
        """Temporal contact witness, robust to single contact-solver gaps."""
        relative=[];contacts=[]
        target=self.sim.model.body(body).id
        for _ in range(max(1,round(seconds/.02))):
            self.hold(.02)
            contacts.append(self.bilateral(body))
            tip,rotation=self.arm.pose()
            relative.append(rotation.T@(self.sim.data.xpos[target]-tip))
        drift=float(np.linalg.norm(np.asarray(relative)[-1]-relative[0]))
        witness={'time':self.sim.data.time,'body':body,'bilateral_fraction':float(np.mean(contacts)),
                 'relative_drift_m':drift,'relative_distance_m':float(np.linalg.norm(relative[-1]))}
        if not hasattr(self,'grasp_witnesses'):self.grasp_witnesses=[]
        self.grasp_witnesses.append(witness)
        return witness['bilateral_fraction']>=.9 and drift<.015 and witness['relative_distance_m']<.075

    def begin_payload(self,body,rotation,*,world_stabilize=True):
        """Track a physical payload while compensating base yaw; never constrains the object."""
        self.payload_body=body
        self.payload_rotation=np.asarray(rotation,dtype=float).copy() if world_stabilize else None
        pose=self.base.pose();tip=self.arm.pose()[0]
        c,s=math.cos(pose[2]),math.sin(pose[2])
        dx,dy=tip[0]-pose[0],tip[1]-pose[1]
        self.payload_local_xy=np.array([c*dx+s*dy,-s*dx+c*dy])
        self.payload_height=float(tip[2])
        body_id=self.sim.model.body(body).id
        self.payload_relative,self.payload_relative_rotation=relative_grasp_pose(
            tip,self.arm.pose()[1],self.sim.data.xpos[body_id],self.sim.data.xmat[body_id])
        self.payload_world_relative=self.sim.data.xpos[body_id]-tip
        self.payload_previous_base_velocity=self.sim.data.qvel[self.base_velocity_dof:self.base_velocity_dof+6].copy()
        self.payload_monitor_count=0
        self.payload_trace=[]
        self.payload=True

    def end_payload(self):
        self.payload=False
        if self.payload_trace:
            self.payload_trace_history.append({'body':self.payload_body,'samples':self.payload_trace})
        if self.output and self.payload_trace_history:
            self.output.mkdir(parents=True,exist_ok=True)
            self.write_trace('payload_traces',self.payload_trace_history)
        self.payload_body=None
        self.payload_motion_profile='default'
        self.payload_rotation=None
        self.payload_relative=None
        self.payload_world_relative=None
        self.payload_relative_rotation=None

    def _ensure_operation_hold(self):
        if not self.config.get('stationary_manipulation',False):return
        from robot_exam.control.parking import WheelParkingHold
        if self.parking is None:self.parking=WheelParkingHold(self.sim,
            sample_seconds=float(self.config.get('parking_trace_interval_seconds',.05)))
        if self.parking.active and self.parking.armed:return
        self.parking.begin();started=self.sim.data.time
        while not self.parking.ready:
            if self.sim.data.time-started>5.:raise TaskFailure('parking_not_stable','Base failed the 2-second parking gate')
            self.tick()

    def _release_operation_hold(self):
        if self.parking is not None and self.parking.active:
            self.parking_history.append({'anchor':self.parking.anchor.tolist(),'samples':self.parking.samples.copy()})
            self.parking.release();self.parking.samples=[]
            self.wheel_applied=self.sim.data.ctrl[self.base.ids].copy()

    def dock_legacy(self, goal, timeout=15., tolerance=.023, _settle_attempt=0):
        self._release_operation_hold()
        """Frozen Stage 6 controller for paired physical comparisons."""
        start = self.sim.data.time
        trace=[]
        while self.sim.data.time-start < timeout:
            pose = self.base.pose()
            delta = np.asarray(goal[:2])-pose[:2]
            angle = abs(wrap_angle(goal[2]-pose[2]))
            distance = np.linalg.norm(delta)
            along_goal=delta @ [math.cos(goal[2]),math.sin(goal[2])]
            lateral_goal=delta @ [-math.sin(goal[2]),math.cos(goal[2])]
            if round(self.sim.data.time/self.sim.model.opt.timestep)%100==0:
                wheel_dofs=[int(self.sim.model.jnt_dofadr[int(self.sim.model.actuator_trnid[a,0])])
                            for a in self.base.ids]
                trace.append({'time':float(self.sim.data.time),'pose':pose.tolist(),
                              'position_error_m':float(distance),'heading_error_rad':float(angle),
                              'along_error_m':float(along_goal),'lateral_error_m':float(lateral_goal),
                              'wheel_target':getattr(self.base,'wheel_target',np.zeros(4)).tolist(),
                              'wheel_velocity_rad_s':self.sim.data.qvel[wheel_dofs].tolist()})
            base_velocity=self.sim.data.qvel[self.base_velocity_dof:self.base_velocity_dof+3]
            yaw_rate=float(self.sim.data.qvel[self.base_velocity_dof+5])
            if distance<tolerance:
                yaw_error=wrap_angle(goal[2]-pose[2])
                self.base.command(0,np.clip(3.0*yaw_error-.75*yaw_rate,-.8,.8))
            else:
                # Point the skid-steer chassis at the position error (or use a
                # reverse approach when that is the shorter turn). Position-
                # only control against the final dock heading can overshoot
                # and oscillate when the target lies behind the chassis.
                bearing=math.atan2(delta[1],delta[0])
                direction=1
                heading_error=wrap_angle(bearing-pose[2])
                if abs(heading_error)>math.pi/2:
                    direction=-1
                    heading_error=wrap_angle(bearing+math.pi-pose[2])
                yaw_command=np.clip(3.0*heading_error-.75*yaw_rate,-1.0,1.0)
                if abs(heading_error)>.09:
                    self.base.command(0,yaw_command)
                else:
                    heading=np.array([math.cos(pose[2]),math.sin(pose[2])])
                    forward_speed=float(base_velocity[:2]@heading)
                    directed_speed=direction*forward_speed
                    limit=.045 if distance<.12 else (.10 if self.payload else .14)
                    speed=np.clip(.9*distance-.8*directed_speed,-limit,limit)
                    self.base.command(direction*float(speed),yaw_command)
            self.tick()
            if distance < tolerance and angle < .035:
                break
        else:
            self.base.command(0, 0)
            self.dock_traces.append({'goal':list(goal),'position_tolerance_m':tolerance,
                'timeout':True,'elapsed_seconds':float(self.sim.data.time-start),
                'final_pose':self.base.pose().tolist(),
                'final_position_error_m':float(np.linalg.norm(self.base.pose()[:2]-goal[:2])),
                'final_heading_error_rad':float(abs(wrap_angle(self.base.pose()[2]-goal[2]))),
                'command_trace':trace[-100:]})
            if self.output:
                self.output.mkdir(parents=True,exist_ok=True)
                self.write_trace('dock',self.dock_traces)
            raise TaskFailure("navigation_timeout", f"Docking failed: {self.base.pose()} -> {goal}")
        self.base.command(0, 0)
        self.hold(.6)
        settled=self.base.pose().copy()
        position_error=float(np.linalg.norm(settled[:2]-np.asarray(goal[:2])))
        heading_error=float(abs(wrap_angle(settled[2]-goal[2])))
        self.dock_traces.append({'goal':list(goal),'position_tolerance_m':tolerance,
            'settle_attempt':_settle_attempt,'settled_pose':settled.tolist(),
            'position_error_m':position_error,'heading_error_rad':heading_error})
        if position_error>tolerance or heading_error>.035:
            if _settle_attempt<2:
                return self.dock_legacy(goal,timeout=max(2.,timeout*.55),tolerance=tolerance,
                                 _settle_attempt=_settle_attempt+1)
            raise TaskFailure('navigation_accuracy',
                f'Dock pose drift after bounded settle corrections: position={position_error:.4f}m heading={heading_error:.4f}rad goal={goal}')

    def dock(self, goal, timeout=15., tolerance=.023):
        self._release_operation_hold()
        """Wheel-driven terminal pose control with hysteresis and a 2 s rest check.

        The position mode enters heading control inside an inner radius and
        returns only after leaving a larger radius. This prevents the Stage 6
        command reversal at the single 23 mm threshold. The final tolerances
        are the same as ``dock_legacy``; actual velocity is checked as well.
        """
        strategy=self.config.get('local_dock_strategy')
        if strategy and (self.payload or not self.config.get('local_dock_payload_only',False)):
            from robot_exam.control.local_dock import execute_local_dock
            budget=max(timeout,float(self.config.get('local_dock_timeout_seconds',timeout)))
            return execute_local_dock(self,goal,budget,tolerance,strategy)
        goal=np.asarray(goal,dtype=float)
        if goal.shape!=(3,) or not np.isfinite(goal).all():
            raise ValueError('Dock goal must be a finite x, y, yaw pose')
        enter=tolerance
        leave=max(enter+.025,2.0*tolerance)
        start=float(self.sim.data.time)
        previous_payload_docking=getattr(self,'payload_docking',False)
        self.payload_docking=bool(self.payload)
        mode='position'
        stable_for=0.;trace=[];switches=[];residual_stall=0.
        wheel_dofs=[int(self.sim.model.jnt_dofadr[int(self.sim.model.actuator_trnid[a,0])])
                    for a in self.base.ids]
        previous_applied=self.sim.data.ctrl[self.base.ids].copy()
        dt=float(self.sim.model.opt.timestep)
        try:
            while self.sim.data.time-start<timeout:
                pose=self.base.pose()
                delta=goal[:2]-pose[:2]
                distance=float(np.linalg.norm(delta))
                yaw_error=wrap_angle(goal[2]-pose[2])
                free_velocity=self.sim.data.qvel[self.base_velocity_dof:self.base_velocity_dof+6]
                linear_speed=float(np.linalg.norm(free_velocity[:2]))
                yaw_rate=float(free_velocity[5])
                inside=(distance<tolerance and abs(yaw_error)<.035)
                stopped=(linear_speed<.02 and abs(yaw_rate)<.03)
                residual_stall=(residual_stall+dt if mode=='heading' and
                                distance>=tolerance and abs(yaw_error)<.035 and stopped else 0.)
                reentry_after=float(self.config.get('dock_residual_reentry_seconds',0.))
                if mode=='settle' and not inside:
                    new_mode='position' if distance>leave else 'heading'
                elif mode=='position' and distance<enter:
                    new_mode='heading'
                elif mode=='heading' and distance>leave:
                    new_mode='position'
                elif mode=='heading' and reentry_after>0 and residual_stall>=reentry_after:
                    # Heading-only feedback cannot remove a lateral residual
                    # trapped between the entry and exit hysteresis radii.
                    # Re-enter position control only after measured rest;
                    # retain the original pose/velocity acceptance limits.
                    new_mode='position'
                elif mode=='heading' and inside and stopped:
                    new_mode='settle'
                else:new_mode=mode
                if new_mode!=mode:
                    switches.append({'time':float(self.sim.data.time),'from':mode,'to':new_mode,
                                     'position_error_m':distance,'heading_error_rad':float(abs(yaw_error))})
                    mode=new_mode
                    stable_for=0.;residual_stall=0.
                heading=np.array([math.cos(pose[2]),math.sin(pose[2])])
                forward_speed=float(free_velocity[:2]@heading)
                if mode=='position':
                    goal_heading=np.array([math.cos(goal[2]),math.sin(goal[2])])
                    along=float(delta@goal_heading)
                    lateral=float(delta@[-goal_heading[1],goal_heading[0]])
                    if distance<.18 and abs(yaw_error)<.9 and abs(lateral)<.020:
                        # Final approach follows the requested docking frame.
                        # Turning to face a bearing that swings rapidly near
                        # the goal caused an unnecessary skid pivot at 3 cm.
                        linear=float(np.clip(.9*along-.55*forward_speed,-.055,.055))
                        lateral_sign=(-1. if along<0 and self.config.get('dock_reverse_lateral_sign',False) else 1.)
                        angular=float(np.clip(2.3*yaw_error+2.0*lateral_sign*lateral-.55*yaw_rate,-.75,.75))
                    else:
                        bearing=math.atan2(delta[1],delta[0])
                        heading_error=wrap_angle(bearing-pose[2])
                        direction=1.
                        if abs(heading_error)>math.pi/2:
                            direction=-1.
                            heading_error=wrap_angle(bearing+math.pi-pose[2])
                        angular=float(np.clip(2.0*heading_error-.55*yaw_rate,-.75,.75))
                        if abs(heading_error)>.12:linear=0.
                        else:
                            limit=.06 if distance<.12 else (.10 if self.payload else .14)
                            linear=direction*float(np.clip(1.0*distance-.75*direction*forward_speed,0.,limit))
                elif mode=='heading':
                    along=float(delta@heading)
                    linear=float(np.clip(.6*along-.5*forward_speed,-.025,.025))
                    angular=float(np.clip(1.8*yaw_error-.55*yaw_rate,-.55,.55))
                    if (self.config.get('banana_staging_pure_yaw',False)
                            and self.payload and getattr(self,'payload_motion_profile','')=='banana'
                            and tolerance>=.08 and abs(yaw_error)>.25):
                        # The staging dock may enter heading control with a
                        # large yaw error at its 85 mm handoff radius. Its
                        # small forward creep then sweeps the base away from
                        # the target while carrying a banana. Keep this
                        # development option limited to that staging phase.
                        linear=0.
                else:
                    linear=angular=0.
                if mode!='settle' and abs(angular)>.03:
                    # The four wheel skid-steer needs a finite differential
                    # wheel target to overcome static tire scrub. This term
                    # is actuator feedback, not a direct base pose change.
                    feedforward=.35 if self.payload else .45
                    max_angular=.85 if self.payload else 1.25
                    angular=float(np.clip(angular+feedforward*np.sign(angular),-max_angular,max_angular))
                self.base.command(linear,angular)
                desired=self.sim.data.ctrl[self.base.ids].copy()
                if not self.payload:
                    step=.8*dt
                    applied=previous_applied+np.clip(desired-previous_applied,-step,step)
                    self.sim.data.ctrl[self.base.ids]=applied
                    previous_applied=applied
                if round(self.sim.data.time/dt)%10==0:
                    trace.append({'time':float(self.sim.data.time),'mode':mode,
                        'pose':pose.tolist(),'goal':goal.tolist(),
                        'position_error_m':distance,'heading_error_rad':float(abs(yaw_error)),
                        'linear_speed_m_s':linear_speed,'yaw_rate_rad_s':yaw_rate,
                        'linear_command_m_s':linear,'angular_command_rad_s':angular,
                        'wheel_target_rad_s':desired.tolist(),
                        'wheel_applied_rad_s':self.sim.data.ctrl[self.base.ids].tolist(),
                        'wheel_velocity_rad_s':self.sim.data.qvel[wheel_dofs].tolist()})
                self.tick()
                if mode=='settle':
                    stable_for=stable_for+dt if stopped else 0.
                    if stable_for>=2.:
                        break
            else:
                self.base.command(0,0)
                category='navigation_timeout'
                raise TaskFailure(category,f'Docking failed: {self.base.pose()} -> {goal.tolist()}')
            self.base.command(0,0)
            pose=self.base.pose()
            velocity=self.sim.data.qvel[self.base_velocity_dof:self.base_velocity_dof+6]
            final_position=float(np.linalg.norm(pose[:2]-goal[:2]))
            final_heading=float(abs(wrap_angle(pose[2]-goal[2])))
            final_linear=float(np.linalg.norm(velocity[:2]))
            final_angular=float(abs(velocity[5]))
            if (final_position>=tolerance or final_heading>=.035
                    or final_linear>=.02 or final_angular>=.03):
                raise TaskFailure('navigation_accuracy','Dock drifted during the 2 s stability window')
        finally:
            self.payload_docking=previous_payload_docking
            self.dock_traces.append({'controller':'hysteresis_v1','goal':goal.tolist(),
                'position_tolerance_m':tolerance,'heading_tolerance_rad':.035,
                'elapsed_seconds':float(self.sim.data.time-start),'mode_switches':switches,
                'stable_seconds':stable_for,'final_pose':self.base.pose().tolist(),
                'samples':trace})
            if self.output:
                self.output.mkdir(parents=True,exist_ok=True)
                self.write_trace('dock',self.dock_traces)

    def navigate(self, goal, grid, timeout=120., max_replans=None):
        self._release_operation_hold()
        from robot_exam.navigation.grid import NoPath
        if max_replans is None:max_replans=self.config.get('maximum_navigation_replans',2)
        start_time = self.sim.data.time
        origin = self.base.pose().copy()
        path = []
        for attempt in range(max_replans+1):
            try:
                approach=np.asarray(goal[:2])-.40*np.array([math.cos(goal[2]),math.sin(goal[2])])
                current=self.base.pose()
                direct_banana=(self.payload and getattr(self,'payload_motion_profile','') in
                               self.config.get('payload_direct_visible_categories',['banana'])
                               and grid.free(grid.cell(goal))
                               and grid.visible(grid.cell(current[:2]),grid.cell(goal)))
                if direct_banana:
                    # The inflated occupancy grid certifies the straight
                    # corridor. The old mandatory pre-goal waypoint caused
                    # slow banana carries to loop backward near the bin.
                    path=[current[:2].tolist(),list(goal[:2])]
                elif grid.free(grid.cell(approach)) and grid.visible(grid.cell(approach),grid.cell(goal)):
                    path=grid.plan(self.base.pose(),approach)
                    path.append(list(goal[:2]))
                else:
                    path=grid.plan(self.base.pose(),goal)
                for segment_index,point in enumerate(path[1:]):
                    segment_start = self.sim.data.time
                    segment_budget=55.
                    if self.payload and getattr(self,'payload_motion_profile','')=='banana':
                        segment_origin=self.base.pose()
                        segment_distance=float(np.linalg.norm(segment_origin[:2]-point))
                        segment_heading=math.atan2(point[1]-segment_origin[1],point[0]-segment_origin[0])
                        segment_turn=abs(wrap_angle(segment_heading-segment_origin[2]))
                        safe_speed=max(.025,float(self.config.get('payload_banana_linear_speed_m_s',.035)))
                        # A 90-degree physical skid turn and a 1.5 m slow
                        # carry cannot fit the old fixed 55 s waypoint clock.
                        # This adjusts only the bounded progress budget; pose
                        # and contact success conditions remain unchanged.
                        # Near the final station the wheel controller may
                        # spend several seconds cancelling skid before it
                        # makes forward progress. Replanning at the old
                        # per-waypoint deadline sent a nearly arrived robot
                        # back toward the first waypoint. Keep the global
                        # navigation timeout and final dock tolerance intact.
                        segment_budget=max(90.,10.+segment_distance/safe_speed+segment_turn/.035)
                    turn_anchor=None
                    final_banana_handoff=(self.payload and getattr(self,'payload_motion_profile','') in
                                          self.config.get('payload_visible_handoff_categories',['banana'])
                                          and segment_index==len(path)-2)
                    waypoint_tolerance=.085 if final_banana_handoff else .055
                    while True:
                        pose = self.base.pose()
                        distance = np.linalg.norm(pose[:2]-point)
                        if distance < waypoint_tolerance:
                            break
                        if (self.payload and getattr(self,'payload_motion_profile','')=='banana'
                                and segment_index<len(path)-2 and distance<.15
                                and grid.visible(grid.cell(pose[:2]),grid.cell(path[segment_index+2]))):
                            # The inflated map already certifies the next
                            # segment. Avoid orbiting a waypoint that a
                            # physical skid-steer chassis cannot hit exactly.
                            break
                        if (final_banana_handoff and distance<.20
                                and grid.visible(grid.cell(pose[:2]),grid.cell(goal))):
                            # The path follower repeatedly overshot its last
                            # low-speed point while carrying a banana. Hand
                            # the final local correction to the already
                            # wheel-driven pose controller. This is only a
                            # navigation handoff; the task later docks at
                            # its original 30 mm placement tolerance.
                            self.base.command(0,0)
                            self.dock(goal,timeout=45.,tolerance=.085)
                            break
                        if self.sim.data.time-start_time > timeout or self.sim.data.time-segment_start > segment_budget:
                            raise TaskFailure("navigation_timeout",
                                f"Waypoint progress timeout segment={segment_index} "
                                f"point={point} distance={distance:.3f}m "
                                f"budget={segment_budget:.1f}s")
                        heading = math.atan2(point[1]-pose[1], point[0]-pose[0])
                        error=wrap_angle(heading-pose[2])
                        if abs(error)>.08:
                            if turn_anchor is None:turn_anchor=pose[:2].copy()
                            self.base.rotate(turn_anchor,heading)
                        else:
                            turn_anchor=None
                            self.base.command(min(.20,1.1*distance),np.clip(10*error+(.4*np.sign(error) if abs(error)>.02 else 0),-2,2))
                        self.tick()
                self.base.command(0, 0)
                anchor = self.base.pose()[:2].copy()
                for _ in range(round(12/self.sim.model.opt.timestep)):
                    if abs(wrap_angle(goal[2]-self.base.pose()[2])) < .03:
                        break
                    self.base.rotate(anchor, goal[2])
                    self.tick()
                self.base.command(0, 0)
                self.hold(.6)
                if np.linalg.norm(self.base.pose()[:2]-goal[:2]) > .09:
                    raise TaskFailure("navigation_accuracy", "Position drift after final turn")
                break
            except (NoPath, TaskFailure) as exc:
                if isinstance(exc,TaskFailure) and exc.category not in (
                        'navigation_timeout','navigation_accuracy','path_planning'):
                    # Payload slip, arm IK and state timeouts originate inside
                    # tick(). Replanning the chassis cannot cure them, and
                    # replacing their category with path_planning hid the
                    # fruit controller's bounded grasp-recovery branch.
                    raise
                if attempt == max_replans:
                    raise TaskFailure("path_planning", str(exc)) from exc
                self.retries += 1
                self.transition(State.RECOVER, "navigation_replan", target=list(goal), timeout=timeout)
        samples = [r for r in self.rows if r["time"] >= start_time]
        poses = np.array([r["base"][:2] for r in samples])
        actual_length = float(np.linalg.norm(np.diff(poses, axis=0), axis=1).sum()) if len(poses)>1 else 0
        segments = [(np.asarray(a), np.asarray(b)) for a,b in zip(path[:-1], path[1:])]
        def path_error(p):
            return min(np.linalg.norm(p-(a+np.clip((p-a)@(b-a)/max(np.sum((b-a)**2),1e-12),0,1)*(b-a))) for a,b in segments)
        result = {"start": origin.tolist(), "goal": list(goal), "path": path,
                  "planned_length_m": float(sum(math.dist(a,b) for a,b in zip(path[:-1],path[1:]))),
                  "actual_length_m": actual_length, "mean_cross_track_error_m": float(np.mean([path_error(p) for p in poses])) if len(poses) and segments else 0,
                  "position_error_m": float(np.linalg.norm(self.base.pose()[:2]-goal[:2])),
                  "heading_error_rad": float(abs(wrap_angle(self.base.pose()[2]-goal[2]))),
                  "sim_seconds": self.sim.data.time-start_time, "replans": attempt,
                  "collision_events": len(self.collision_events)}
        self.navigation.append(result)
        return result

    def backout(self, x, timeout=10):
        self._release_operation_hold()
        """Leave a work bay backwards before global planning; no lateral dock."""
        start=self.sim.data.time
        if self.payload:
            profile=getattr(self,'payload_motion_profile','default')
            speed_key='payload_banana_linear_speed_m_s' if profile=='banana' else 'payload_linear_speed_m_s'
            speed=float(self.config.get(speed_key,.035 if profile=='banana' else .10))
            timeout=max(timeout,1.5*max(0.,self.base.pose()[0]-x)/max(speed,.01)+2.)
        while self.base.pose()[0]>x+.012:
            if self.sim.data.time-start>timeout:raise TaskFailure('navigation_timeout','Work-bay reverse clearance timeout')
            yaw=self.base.pose()[2]
            w=np.clip(-6*yaw-.4*np.sign(yaw),-1.5,1.5) if abs(yaw)>.025 else 0
            self.base.command(-.12,w);self.tick()
        self.base.command(0,0);self.hold(.5)

    def write_trace(self,name,data):
        """Optional lossless compression of NEW bulk logs; never alters control."""
        if not self.output:return
        self.output.mkdir(parents=True,exist_ok=True)
        compressed=(self.config.get('compress_bulk_logs',False) and
                    name in ('trajectory','dock','payload_traces','parking','ik_diagnostics'))
        if compressed:
            import gzip
            with gzip.open(self.output/(name+'.json.gz'),'wt',encoding='utf-8') as stream:
                json.dump(data,stream,ensure_ascii=False)
        else:(self.output/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False))

    def save(self):
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            for name, data in (("trajectory", self.rows), ("fsm", self.transitions), ("navigation", self.navigation), ("dock",self.dock_traces), ("ik_diagnostics",self.ik_diagnostics), ("collisions", self.collision_events), ("grasp_witnesses", getattr(self,'grasp_witnesses',[]))):
                self.write_trace(name,data)
            if self.parking_history or (self.parking is not None and self.parking.active):
                history=self.parking_history.copy()
                if self.parking is not None and self.parking.active:
                    history.append({'anchor':self.parking.anchor.tolist(),'samples':self.parking.samples})
                self.write_trace('parking',history)
            if self.payload_trace_history or self.payload_trace:
                traces=self.payload_trace_history.copy()
                if self.payload_trace and (not traces or traces[-1]['samples'] is not self.payload_trace):
                    traces.append({'body':self.payload_body,'samples':self.payload_trace})
                self.write_trace('payload_traces',traces)
