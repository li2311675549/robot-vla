"""Bounded bidirectional RRT for a fixed-base original UR5, scratch collisions only."""
import time
import mujoco
import numpy as np


def plan_joint_rrt(arm,target,seed=101,max_nodes=4000,wall_seconds=20.,allowed_bodies=(),held_body=None):
    model=arm.sim.model;scratch=mujoco.MjData(model);scratch.qpos[:]=arm.sim.data.qpos
    start=arm.sim.data.qpos[arm.qadr].copy();target=np.asarray(target)
    limits=model.jnt_range[arm.joints];rng=np.random.default_rng(seed);clock=time.monotonic();queries=0
    links={'shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link','tool0','wrist_rgb_mount'}
    # Optional rigid-offset prediction is ONLY for collision queries. It adds
    # the object to the swept assembly, never attaches it in live simulation.
    if held_body is not None:
        body_id=model.body(held_body).id
        joint=int(model.body_jntadr[body_id])
        if joint<0 or model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Held-object prediction requires an independent free body')
        adr=int(model.jnt_qposadr[joint]);tip,rotation=arm.pose()
        relative_p=rotation.T@(arm.sim.data.xpos[body_id]-tip)
        relative_r=rotation.T@arm.sim.data.xmat[body_id].reshape(3,3)
        links.add(held_body)
    def update_scratch(q):
        scratch.qpos[arm.qadr]=q;mujoco.mj_forward(model,scratch)
        if held_body is not None:
            tip=scratch.site_xpos[arm.site]
            rotation=scratch.site_xmat[arm.site].reshape(3,3)
            scratch.qpos[adr:adr+3]=tip+rotation@relative_p
            mujoco.mju_mat2Quat(scratch.qpos[adr+3:adr+7],(rotation@relative_r).ravel())
            mujoco.mj_forward(model,scratch)
    def valid(q):
        nonlocal queries
        queries+=1
        if not np.isfinite(q).all() or np.any(q<limits[:,0]) or np.any(q>limits[:,1]):return False
        update_scratch(q)
        for contact in scratch.contact:
            if contact.dist>=-.0003:continue
            names=[model.body(int(model.geom_bodyid[g])).name or '' for g in contact.geom]
            relevant=[n in links or n.startswith('grip_') for n in names]
            if not any(relevant):continue
            if all(n.startswith('grip_') for n in names):continue
            if any(n in allowed_bodies for n in names) and any(n.startswith('grip_') for n in names):continue
            return False
        return True
    def segment(a,b):
        return all(valid(q) for q in np.linspace(a,b,max(2,int(np.max(np.abs(b-a))/.055)+2)))
    def invalid_contacts():
        details=[]
        for contact in scratch.contact:
            if contact.dist>=-.0003:continue
            names=[model.body(int(model.geom_bodyid[g])).name or '' for g in contact.geom]
            if not any(n in links or n.startswith('grip_') for n in names):continue
            if all(n.startswith('grip_') for n in names):continue
            if any(n in allowed_bodies for n in names) and any(n.startswith('grip_') for n in names):continue
            geoms=[model.geom(int(g)).name or f'geom_{int(g)}' for g in contact.geom]
            details.append('/'.join(names)+f' penetration={-float(contact.dist):.6f}m geoms='+('/'.join(geoms)))
        return '; '.join(details[:5]) or 'joint-limit or non-finite target'
    if not valid(start):raise RuntimeError('RRT start is in physical collision: '+invalid_contacts())
    if not valid(target):raise RuntimeError('RRT goal is in physical collision: '+invalid_contacts())
    if segment(start,target):return [target],{'nodes':2,'queries':queries,'wall_seconds':time.monotonic()-clock}
    trees=[([start],[-1]),([target],[-1])]
    def extend(tree,q):
        nodes,parents=tree;nearest=int(np.argmin([np.linalg.norm(v-q) for v in nodes]));v=nodes[nearest]
        delta=q-v;nxt=v+delta*min(1.,.28/max(np.linalg.norm(delta),1e-10))
        if not segment(v,nxt):return None
        nodes.append(nxt);parents.append(nearest);return len(nodes)-1
    def history(tree,index):
        result=[]
        while index!=-1:result.append(tree[0][index]);index=tree[1][index]
        return result[::-1]
    for iteration in range(max_nodes):
        if time.monotonic()-clock>wall_seconds:break
        side=iteration%2;a,b=trees[side],trees[1-side]
        if rng.random()<.2:sample=b[0][rng.integers(len(b[0]))]
        elif rng.random()<.75:sample=np.clip((start+target)/2+rng.normal(0,1.4,6),limits[:,0],limits[:,1])
        else:sample=rng.uniform(limits[:,0],limits[:,1])
        index=extend(a,sample)
        if index is None:continue
        meet=a[0][index]
        for _ in range(35):
            other=extend(b,meet)
            if other is None:break
            if np.linalg.norm(b[0][other]-meet)<1e-7:
                pa,pb=history(a,index),history(b,other)
                path=pa+pb[-2::-1] if side==0 else pb+pa[-2::-1]
                # Deterministic shortcutting preserves collision checks.
                short=[path[0]];i=0
                while i<len(path)-1:
                    j=len(path)-1
                    while j>i+1 and not segment(path[i],path[j]):j-=1
                    short.append(path[j]);i=j
                return short[1:],{'nodes':sum(len(t[0]) for t in trees),'queries':queries,'wall_seconds':time.monotonic()-clock}
    raise RuntimeError(f'Joint RRT bounded search exhausted: {sum(len(t[0]) for t in trees)} nodes, {queries} queries')
