"""Small finite waypoint search for camera-arch clearance (not a general RRT)."""
import itertools
import mujoco
import numpy as np


def clear_joint_path(arm, target):
    model=arm.sim.model
    scratch=mujoco.MjData(model);scratch.qpos[:]=arm.sim.data.qpos
    start=arm.sim.data.qpos[arm.qadr].copy()
    arm_names={'shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link','tool0'}
    def segment(a,b):
        count=max(2,int(np.max(np.abs(b-a))/.06)+2)
        for q in np.linspace(a,b,count):
            scratch.qpos[arm.qadr]=q;mujoco.mj_forward(model,scratch)
            for contact in scratch.contact:
                if contact.dist>=-.0003:continue
                names=[model.body(int(model.geom_bodyid[g])).name or '' for g in contact.geom]
                if not any(n in arm_names or n.startswith('grip_') for n in names):continue
                if all(n.startswith('grip_') for n in names):continue
                return False
        return True
    if segment(start,target):return [target]
    for shoulder,elbow,wrist in itertools.product([-1.57,-2.1,-1.1,-2.6], [2.4,2.8,1.6,.6,-1.6], [-1.57,0,1.57]):
        before=np.array([start[0],shoulder,elbow,wrist,target[4],target[5]])
        after=before.copy();after[0]=target[0]
        if segment(start,before) and segment(before,after) and segment(after,target):
            return [before,after,target]
    raise RuntimeError('No camera-arch-clear path in the finite joint waypoint library')
