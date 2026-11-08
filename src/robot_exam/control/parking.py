"""Bounded wheel-encoder parking servo and independent chassis-motion guard.

Only existing wheel actuator targets are written. The floating base, ground
contacts, friction and actuator force limits are never modified.
"""
import numpy as np

from robot_exam.control.base import BaseController, wrap_angle


class ParkingViolation(RuntimeError):
    category = 'parking_drift'


class WheelParkingHold:
    def __init__(self, sim, position_tolerance=.010, heading_tolerance=.020,
                 linear_limit=.020, angular_limit=.030, stable_seconds=2.,
                 wheel_position_gain=8., wheel_damping=.4, max_wheel_speed=.75,
                 sample_seconds=.05):
        self.sim=sim;self.base=BaseController(sim)
        joints=sim.model.actuator_trnid[self.base.ids,0].astype(int)
        self.qadr=sim.model.jnt_qposadr[joints];self.dofs=sim.model.jnt_dofadr[joints]
        joint=sim.model.joint('husky_free').id
        self.base_dof=int(sim.model.jnt_dofadr[joint])
        self.position_tolerance=position_tolerance;self.heading_tolerance=heading_tolerance
        self.linear_limit=linear_limit;self.angular_limit=angular_limit;self.required_stable_seconds=stable_seconds
        self.kp=wheel_position_gain;self.kd=wheel_damping;self.max_wheel_speed=max_wheel_speed
        if not np.isfinite(sample_seconds) or sample_seconds<.05:
            raise ValueError('Parking output sample period must be finite and at least0.05s')
        self.sample_seconds=sample_seconds
        self.active=False;self.ready=False;self.armed=False;self.faulted=False;self.samples=[];self.stable_seconds=0.;self.bad_seconds=0.

    def begin(self):
        if self.active:return
        self.anchor=self.base.pose().copy();self.wheel_anchor=self.sim.data.qpos[self.qadr].copy()
        self.active=True;self.ready=False;self.armed=False;self.faulted=False;self.stable_seconds=0.;self.bad_seconds=0.
        self.started=float(self.sim.data.time)
        self.sim.add_step_controller(self)
        self.before_step()

    def release(self):
        if not self.active:return
        self.sim.remove_step_controller(self);self.active=False;self.ready=False;self.armed=False
        self.base.command(0,0)

    def metrics(self):
        pose=self.base.pose();velocity=self.sim.data.qvel[self.base_dof:self.base_dof+6]
        return {'time':float(self.sim.data.time),'pose':pose.tolist(),
                'position_error_m':float(np.linalg.norm(pose[:2]-self.anchor[:2])),
                'heading_error_rad':float(abs(wrap_angle(pose[2]-self.anchor[2]))),
                'linear_speed_m_s':float(np.linalg.norm(velocity[:2])),
                'yaw_rate_rad_s':float(abs(velocity[5]))}

    def before_step(self):
        if self.active:
            if self.faulted:
                self.sim.data.ctrl[self.base.ids]=0
                return
            error=self.wheel_anchor-self.sim.data.qpos[self.qadr]
            target=self.kp*error-self.kd*self.sim.data.qvel[self.dofs]
            pose=self.base.pose();heading=np.array([np.cos(pose[2]),np.sin(pose[2])])
            velocity=self.sim.data.qvel[self.base_dof:self.base_dof+6]
            # Encoders cannot distinguish rolling from tire scrub. Add bounded
            # odometric longitudinal/yaw damping; never request lateral motion.
            linear=-2.*((pose[:2]-self.anchor[:2])@heading)-float(velocity[:2]@heading)
            angular=-4.*wrap_angle(pose[2]-self.anchor[2])-2.*float(velocity[5])
            correction=np.array([linear-angular*self.base.wheel_track/2,
                                 linear+angular*self.base.wheel_track/2]*2)/self.base.wheel_radius
            target=np.clip(target+correction,-self.max_wheel_speed,self.max_wheel_speed)
            self.sim.data.ctrl[self.base.ids]=target

    def after_step(self):
        if not self.active or self.faulted:return
        m=self.metrics();dt=float(self.sim.model.opt.timestep)
        valid=(m['position_error_m']<self.position_tolerance and
               m['heading_error_rad']<self.heading_tolerance and
               m['linear_speed_m_s']<self.linear_limit and m['yaw_rate_rad_s']<self.angular_limit)
        self.stable_seconds=self.stable_seconds+dt if valid else 0.
        self.ready=self.stable_seconds+1e-9>=self.required_stable_seconds
        self.armed=self.armed or self.ready
        self.bad_seconds=0. if valid else self.bad_seconds+dt
        if not self.samples or m['time']-self.samples[-1]['time']>=self.sample_seconds-.001:
            m.update(stable_seconds=self.stable_seconds,ready=self.ready,
                     wheel_reference=self.sim.data.ctrl[self.base.ids].tolist(),
                     wheel_velocity=self.sim.data.qvel[self.dofs].tolist())
            m['nonfloor_contacts']=[{'bodies':[self.sim.model.body(int(self.sim.model.geom_bodyid[g])).name for g in c.geom],
                'distance_m':float(c.dist)} for c in self.sim.data.contact
                if c.dist<-.0001 and all(int(self.sim.model.geom_bodyid[g])!=0 for g in c.geom)]
            self.samples.append(m)
        if self.bad_seconds>.10:
            self.faulted=True;self.ready=False;self.armed=False
            self.sim.data.ctrl[self.base.ids]=0
            raise ParkingViolation(f'Physical parking envelope exceeded: {m}')
