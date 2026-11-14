"""Husky skid-steer wheel velocity controller; base pose is read-only."""
import math
import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


class BaseController:
    wheel_radius = .1651
    wheel_track = .555

    def __init__(self, sim):
        self.sim = sim
        self.body = sim.model.body("husky").id
        self.ids = [sim.model.actuator(f"{a}_{s}_wheel_velocity").id
                    for a in ("front", "rear") for s in ("left", "right")]
        self.precision_turn = False

    def pose(self):
        p = self.sim.data.xpos[self.body]
        r = self.sim.data.xmat[self.body].reshape(3, 3)
        return np.array([p[0], p[1], math.atan2(r[1, 0], r[0, 0])])

    def command(self, linear, angular):
        if not np.isfinite([linear, angular]).all():
            raise ValueError("Nonfinite base command")
        self.precision_turn = False
        left = (linear - angular*self.wheel_track/2) / self.wheel_radius
        right = (linear + angular*self.wheel_track/2) / self.wheel_radius
        self.wheel_target = np.clip([left, right, left, right], -12, 12)
        self.sim.data.ctrl[self.ids] = self.wheel_target

    def track(self, goal, max_linear=.25):
        x, y, yaw = self.pose()
        dx, dy = goal[0] - x, goal[1] - y
        distance = math.hypot(dx, dy)
        bearing = wrap_angle(math.atan2(dy, dx) - yaw)
        if distance > .012:
            v = min(max_linear, 1.2*distance) if abs(bearing) < .08 else 0
            w = np.clip(10*bearing + (.45*np.sign(bearing) if abs(bearing) > .02 else 0), -2, 2)
        else:
            v = 0
            error = wrap_angle(goal[2] - yaw)
            w = np.clip(4*error + (.45*np.sign(error) if abs(error) > .015 else 0), -2, 2)
        self.command(v, w)
        return distance, abs(wrap_angle(goal[2] - yaw))

    def rotate(self, anchor, yaw, max_angular=2., linear_limit=.25,
               velocity_damping=0., yaw_damping=0., feedforward=.45):
        """Correct skid-induced drift along the heading while rotating."""
        x, y, heading = self.pose()
        along = (x-anchor[0])*math.cos(heading) + (y-anchor[1])*math.sin(heading)
        error = wrap_angle(yaw-heading)
        forward_velocity=yaw_rate=0.
        if velocity_damping or yaw_damping:
            joint=int(self.sim.model.body_jntadr[self.body]);dof=int(self.sim.model.jnt_dofadr[joint])
            velocity=self.sim.data.qvel[dof:dof+6]
            forward_velocity=float(velocity[:2]@[math.cos(heading),math.sin(heading)])
            yaw_rate=float(velocity[5])
        self.command(np.clip(-3*along-velocity_damping*forward_velocity,-linear_limit,linear_limit),
                     np.clip(4*error + (feedforward*np.sign(error) if abs(error) > .015 else 0)-yaw_damping*yaw_rate,-max_angular,max_angular))
        self.precision_turn = True

    def rotate_velocity(self,anchor,yaw,body_yaw_limit=.8):
        """Opt-in measured-twist PI turn, with bounded integral compensation.

        A payload's asymmetric wheel loading creates forward motion during
        differential steering. Regulate actual forward velocity, rather than
        treating a zero wheel-command mean as proof of a stationary pivot.
        """
        pose=self.pose();anchor=np.asarray(anchor);heading=np.array([math.cos(pose[2]),math.sin(pose[2])])
        joint=int(self.sim.model.body_jntadr[self.body]);dof=int(self.sim.model.jnt_dofadr[joint])
        velocity=self.sim.data.qvel[dof:dof+6];forward=float(velocity[:2]@heading)
        if not self.precision_turn or not np.array_equal(anchor,getattr(self,'_turn_anchor',None)):
            self._turn_anchor=anchor.copy();self._turn_integral=np.zeros(2);self._turn_yaw_reference=0.
            wheels=self.sim.data.ctrl[self.ids]
            self._turn_last_request=np.array([self.wheel_radius*.5*(wheels[0]+wheels[1]),self.wheel_radius/self.wheel_track*(wheels[1]-wheels[0])])
        along=float((pose[:2]-anchor)@heading);error=wrap_angle(yaw-pose[2])
        dt=float(self.sim.model.opt.timestep)
        # Precision/load paths have an additional wheel-command ramp. Shape
        # the body reference to that bandwidth instead of integrating into
        # an actuator delay and reversing full yaw targets at the endpoint.
        precision=body_yaw_limit<=.4
        rate=.3 if precision else 1.2;gain=1.2 if precision else 1.5
        wanted=float(np.clip(gain*error,-body_yaw_limit,body_yaw_limit))
        self._turn_yaw_reference+=float(np.clip(wanted-self._turn_yaw_reference,-rate*dt,rate*dt))
        reference=np.array([np.clip(-2*along,-.2,.2),self._turn_yaw_reference])
        measured=np.array([forward,float(velocity[5])])
        self._turn_integral+=np.array([3.,2.])*(reference-measured)*dt
        if hasattr(self,'_turn_last_request'):
            wheels=self.sim.data.ctrl[self.ids]
            applied=np.array([self.wheel_radius*.5*(wheels[0]+wheels[1]),self.wheel_radius/self.wheel_track*(wheels[1]-wheels[0])])
            self._turn_integral+=5.*(applied-self._turn_last_request)*dt
        self._turn_integral=np.clip(self._turn_integral,[-.5,-1.5],[.5,1.5])
        command=reference*np.array([1.,5.])+np.array([1.5,2.])*(reference-measured)+self._turn_integral
        self.command(float(np.clip(command[0],-.6,.6)),float(np.clip(command[1],-6,6)))
        wheels=self.wheel_target
        self._turn_last_request=np.array([self.wheel_radius*.5*(wheels[0]+wheels[1]),self.wheel_radius/self.wheel_track*(wheels[1]-wheels[0])])
        self.precision_turn=True
