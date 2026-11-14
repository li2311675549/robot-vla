"""Opt-in finite-horizon wheel-command path tracking in the inflated map."""
import math
import numpy as np
from robot_exam.control.base import wrap_angle


def pursuit_twist(pose,target,speed,final=False,max_yaw=2.0):
    delta=np.asarray(target)-np.asarray(pose[:2]);distance=float(np.linalg.norm(delta))
    error=wrap_angle(math.atan2(delta[1],delta[0])-pose[2]);direction=1.
    if abs(error)>math.pi*.6:
        direction=-1.;error=wrap_angle(error+math.pi)
    if abs(error)>.9:return 0.,float(np.clip(3*error+.45*np.sign(error),-max_yaw,max_yaw)),True
    velocity=speed*max(.15,math.cos(error)**2)
    if final:velocity=min(velocity,1.8*distance)
    yaw=float(np.clip(4*error+(.4*np.sign(error)if abs(error)>.02 else 0.),-max_yaw,max_yaw))
    return direction*velocity,yaw,False


def arc_is_free(grid,pose,linear,angular,horizon=.35):
    x,y,yaw=map(float,pose)
    for _ in range(7):
        x+=linear*math.cos(yaw)*horizon/7;y+=linear*math.sin(yaw)*horizon/7;yaw+=angular*horizon/7
        if not grid.free(grid.cell([x,y])):return False
    return True


def velocity_pursuit_twist(pose,target,speed,forward_velocity,yaw_rate):
    """Curvature-limited body velocity reference plus measured twist feedback.

    The skid-steer wheel yaw request is distinct from the measured body yaw.
    Never interpret this empirical compensation as a holonomic displacement.
    """
    delta=np.asarray(target)-np.asarray(pose[:2]);distance=float(np.linalg.norm(delta))
    error=wrap_angle(math.atan2(delta[1],delta[0])-pose[2])
    if abs(error)>.9:return 0.,0.,True
    curvature=2*math.sin(error)/max(distance,.12)
    reference_v=min(speed,1.5*distance,.55/max(abs(curvature),1e-6))
    reference_v*=max(.1,math.cos(error))
    reference_w=reference_v*curvature
    v=float(np.clip(reference_v+1.2*(reference_v-forward_velocity),-.4,speed))
    w=5.5*reference_w+1.8*(reference_w-yaw_rate)
    if abs(reference_w)>.015:w+=math.copysign(.45,reference_w)
    return v,float(np.clip(w,-6.,6.)),False
