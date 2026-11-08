"""Independent truth-based scoring. Never used as a vision detection result."""
import numpy as np
import mujoco


def body_bounds(sim, name):
    model,data=sim.model,sim.data
    body=model.body(name).id
    lower,upper=np.full(3,np.inf),np.full(3,-np.inf)
    for g in range(model.ngeom):
        if model.geom_bodyid[g]!=body or not model.geom_contype[g]:continue
        R=data.geom_xmat[g].reshape(3,3);size=model.geom_size[g]
        kind=model.geom_type[g]
        if kind==mujoco.mjtGeom.mjGEOM_BOX:extent=np.abs(R)@size
        elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:extent=np.sqrt((R**2)@(size**2))
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=np.abs(R[:,2])*size[1]+size[0]
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:extent=np.repeat(size[0],3)
        else:extent=np.repeat(model.geom_rbound[g],3)
        lower=np.minimum(lower,data.geom_xpos[g]-extent);upper=np.maximum(upper,data.geom_xpos[g]+extent)
    return lower,upper


def fruit_score(sim,metadata):
    result=[]
    for obj in metadata['objects']:
        lower,upper=body_bounds(sim,obj['body']);bininfo=metadata['bins'][obj['category']]
        center=np.asarray(bininfo['center']);half=np.asarray(bininfo['inside_halfsize'])
        dof=sim.model.jnt_dofadr[sim.model.joint(obj['body']+'_free').id]
        speed=float(np.linalg.norm(sim.data.qvel[dof:dof+3]))
        contained=bool(np.all(lower[:2]>=center[:2]-half-.003) and np.all(upper[:2]<=center[:2]+half+.003)
                       and lower[2]>=.55-.005 and upper[2]<=bininfo['rim_z']+.01 and speed<.02)
        result.append({'body':obj['body'],'category':obj['category'],'correct_bin':contained,'bounds':[lower.tolist(),upper.tolist()],'speed':speed})
    return {'success':all(x['correct_bin'] for x in result),'correct':sum(x['correct_bin'] for x in result),'total':10,'objects':result}


def drawer_score(sim,metadata,initial):
    records=[]
    for obj in metadata['objects']:
        rank=obj['rank'];lower,upper=body_bounds(sim,obj['body']);opening=float(sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint(f'drawer_slide_{rank}').id]])
        bottom=metadata['drawer_bottoms'][rank]
        dof=sim.model.jnt_dofadr[sim.model.joint(obj['body']+'_free').id]
        inside=bool(lower[0]>.58-opening+.008 and upper[0]<.86-opening and lower[1]>.31 and upper[1]<.69 and lower[2]>bottom-.003 and upper[2]<bottom+.14 and np.linalg.norm(sim.data.qvel[dof:dof+3])<.02)
        records.append({'book':obj['body'],'rank':rank,'inside_correct_layer':inside,'opening':opening})
    body=sim.model.body('husky').id;R=sim.data.xmat[body].reshape(3,3);pose=np.r_[sim.data.xpos[body,:2],np.arctan2(R[1,0],R[0,0])]
    from robot_exam.control.base import wrap_angle
    returned=np.linalg.norm(pose[:2]-initial[:2])<.10 and abs(wrap_angle(pose[2]-initial[2]))<.08
    return {'success':all(r['inside_correct_layer'] and abs(r['opening'])<.012 for r in records) and bool(returned),'books':records,'returned':bool(returned),'final_base':pose.tolist()}
