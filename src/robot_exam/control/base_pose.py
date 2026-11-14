"""Scratch-state evaluation of physically navigable manipulation base poses."""
from types import SimpleNamespace

import mujoco
import numpy as np

from robot_exam.control.arm import ArmController


def evaluate_operational_base_poses(runtime, nominal, targets, offsets=None):
    """Rank nearby Husky poses by IK margin, arm motion, path length and collision.

    Candidate free-base configurations are written only into copied planning
    data. The live simulation is never teleported. ``targets`` is a list of
    ``(name, position, rotation)`` tuples that all need to be reachable from
    the selected base pose.
    """
    sim=runtime.sim;model=sim.model;data=mujoco.MjData(model)
    mujoco.mj_copyData(data,model,sim.data)
    husky=model.body('husky').id
    first=int(model.body_jntadr[husky]);count=int(model.body_jntnum[husky])
    free=next((j for j in range(first,first+count) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE),None)
    if free is None:raise RuntimeError('Husky has no free joint for scratch base-pose evaluation')
    qadr=int(model.jnt_qposadr[free]);dof=int(model.jnt_dofadr[free])
    start=runtime.base.pose().copy();nominal=np.asarray(nominal,dtype=float)
    offsets=offsets or [(0,0),( -.08,0),(-.16,0),(.08,0),(0,-.08),(0,.08),(-.08,-.08),(-.08,.08)]
    robot=set()
    for body in range(model.nbody):
        parent=body
        while parent>0 and parent!=husky:parent=int(model.body_parentid[parent])
        if parent==husky:robot.add(body)
    attempts=[]
    for offset in offsets:
        dx,dy=offset[:2];dyaw=offset[2] if len(offset)>2 else 0.
        pose=nominal.copy();pose[0]+=dx;pose[1]+=dy;pose[2]+=dyaw
        data.qpos[:]=sim.data.qpos
        data.qvel[:]=sim.data.qvel
        data.ctrl[:]=sim.data.ctrl
        data.qpos[qadr:qadr+3]=[pose[0],pose[1],sim.data.qpos[qadr+2]]
        data.qpos[qadr+3:qadr+7]=[np.cos(pose[2]/2),0,0,np.sin(pose[2]/2)]
        mujoco.mj_forward(model,data)
        proxy=SimpleNamespace(model=model,data=data)
        arm=ArmController(proxy);arm.check_self_collisions=bool(runtime.config.get('ik_check_self_collisions',False))
        seed=sim.data.qpos[runtime.arm.qadr].copy()
        ik=[];feasible=True
        for target_spec in targets:
            name,position,rotation=target_spec[:3]
            allowed=target_spec[3] if len(target_spec)>3 else ()
            try:
                q,metrics=arm.solve_multi(position,rotation,seed=seed,max_iterations=500,allowed_bodies=allowed)
                limits=model.jnt_range[arm.joints]
                margin=np.minimum(q-limits[:,0],limits[:,1]-q)
                ik.append({'target':name,'q':q.tolist(),'metrics':metrics,'minimum_limit_margin_rad':float(margin.min())})
                seed=q
                arm.scratch.qpos[arm.qadr]=q;mujoco.mj_forward(model,arm.scratch)
            except RuntimeError as exc:
                feasible=False;ik.append({'target':name,'failure':str(exc),'solver_report':arm.last_solve_report});break
        if not feasible:
            attempts.append({'pose':pose.tolist(),'feasible':False,'ik':ik});continue
        collision=[]
        for c in arm.scratch.contact:
            a,b=map(int,c.geom);ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
            if (ba in robot)==(bb in robot):continue
            other=b if ba in robot else a;name=model.geom(other).name or ''
            blocking=(name.startswith(('wall','door_frame','obstacle','cabinet_shell','table_leg')) or name.endswith('_top'))
            if blocking and float(c.dist)<-.002:collision.append({'geom':name,'penetration_m':-float(c.dist)})
        try:
            if getattr(runtime,'grid',None) is not None:
                # Plan the A* part to a collision-inflated free staging pose;
                # the final short dock is checked in the full contact model.
                staging=pose[:2]-.40*np.array([np.cos(pose[2]),np.sin(pose[2])])
                path=runtime.grid.plan(start,staging)
                path.append(pose[:2].tolist())
            else:path=[start[:2].tolist(),pose[:2].tolist()]
            path_length=float(sum(np.linalg.norm(np.asarray(b)-a) for a,b in zip(path,path[1:])))
        except Exception as exc:
            path=[];path_length=float('inf');path_error=str(exc)
        else:path_error=None
        joint_cost=float(sum(np.linalg.norm(np.asarray(x['q'])-runtime.arm.reference) for x in ik))
        margin_cost=float(sum(1/max(x['minimum_limit_margin_rad'],.02) for x in ik))
        score=path_length+.15*joint_cost+.02*margin_cost+100*len(collision)
        attempts.append({'pose':pose.tolist(),'feasible':bool(not collision and np.isfinite(path_length)),'path_length_m':path_length,
                         'path_error':path_error,'joint_motion_cost':joint_cost,'joint_limit_cost':margin_cost,
                         'collisions':collision,'ik':ik,'score':score})
    valid=[x for x in attempts if x['feasible']]
    report={'nominal':nominal.tolist(),'start':start.tolist(),'target_names':[x[0] for x in targets],
            'candidates':attempts,'selected':min(valid,key=lambda x:x['score']) if valid else None}
    return report
