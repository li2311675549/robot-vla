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

    def rotate(self, anchor, yaw):
        """Correct skid-induced drift along the heading while rotating."""
        x, y, heading = self.pose()
        along = (x-anchor[0])*math.cos(heading) + (y-anchor[1])*math.sin(heading)
        error = wrap_angle(yaw-heading)
        self.command(np.clip(-3*along, -.25, .25),
                     np.clip(4*error + (.45*np.sign(error) if abs(error) > .015 else 0), -2, 2))
        self.precision_turn = True
