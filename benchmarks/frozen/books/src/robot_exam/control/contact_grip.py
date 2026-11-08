"""Ordinary position-actuator commands with bounded measured-contact feedback."""
import numpy as np


def enabled_for(runtime,category):
    return bool(runtime.config.get('fruit_contact_force_feedback',False) and
                category in runtime.config.get('fruit_force_feedback_categories',['apple','banana','orange']))


def finger_forces(runtime,body):
    contacts=runtime.contacts(body)
    return {side:float(sum(c['normal_force_n'] for c in contacts if c['body'].startswith('grip_'+side)))
            for side in ('left','right')}


def close_until_bilateral_contact(runtime,body,target_force_n=12.):
    r=runtime
    for closure in np.linspace(.03,1.,66):
        r.gripper.command(float(closure));r.hold(.025)
        force=finger_forces(r,body)
        if min(force.values())>=3.:
            r.grip_force_feedback={'body':body,'target_force_n':target_force_n,
                'force_deadband_n':float(r.config.get('fruit_contact_force_deadband_n',0.))}
            r.hold(.6)
            return
    raise RuntimeError('Bounded physical closure found no bilateral contact')


def update_contact_feedback(runtime):
    r=runtime;feedback=getattr(r,'grip_force_feedback',None)
    if not feedback:return
    closure=float(r.sim.data.ctrl[r.gripper.actuator]/255)
    if closure<.01:
        r.grip_force_feedback=None
        return
    if round(r.sim.data.time/r.sim.model.opt.timestep)%10:return
    forces=finger_forces(r,feedback['body']);measured=min(forces.values())
    error=feedback['target_force_n']-measured
    deadband=float(feedback.get('force_deadband_n',0.))
    if not np.isfinite(deadband) or deadband<0 or deadband>feedback['target_force_n']*.25:
        raise ValueError('Contact feedback deadband must be within25% of the requested normal force')
    error=np.sign(error)*max(0.,abs(error)-deadband)
    delta=float(np.clip(.0002*error,-.01,.004))
    # Actuator gain/force ranges are unchanged. No forces are applied to the
    # object; only the existing normalized gripper position target is updated.
    r.gripper.command(float(np.clip(closure+delta,.03,1.)))
