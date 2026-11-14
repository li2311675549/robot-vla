"""Explicit Oracle push/side-pinch skill, using real robot/object contact only."""
import numpy as np
from robot_exam.tasks.runtime import State,TaskFailure
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.control.joint_rrt import plan_joint_rrt


def book_centerline_halfwidth(rotation,size):
    """X extent of the actual rectangle at world y=center_y, not its AABB.

    For a horizontal book and a centre-line side approach, local coordinates
    are dx*R[0,i]. Each must remain inside its own half-size.
    """
    rotation=np.asarray(rotation).reshape(3,3)
    if abs(rotation[2,2])<.98:raise TaskFailure('book_edge_orientation','Book is not flat enough for a horizontal edge approach')
    return float(np.min(np.asarray(size[:2])*.5/np.maximum(np.abs(rotation[0,:2]),1e-10)))


def stable_edge_book(r,obj,reference_relative=None,seconds=.6):
    """Retain temporal contact/drift checks without a fruit-sized radius cap.

    A rectangular book is pinched near an edge, not its centre. Bound the
    centre by its actual half diagonal plus the existing18mm endpoint error,
    and independently require bilateral contact and bounded relative motion.
    """
    body=r.sim.model.body(obj['body']).id
    positions=[];contacts=[]
    for _ in range(max(1,round(seconds/.02))):
        r.hold(.02);tip,rotation=r.arm.pose()
        positions.append(rotation.T@(r.sim.data.xpos[body]-tip))
        contacts.append(r.bilateral(obj['body']))
    values=np.asarray(positions)
    reference=values[0] if reference_relative is None else np.asarray(reference_relative)
    maximum_drift=float(np.max(np.linalg.norm(values-reference,axis=1)))
    maximum_distance=float(np.max(np.linalg.norm(values,axis=1)))
    geometry_bound=float(np.linalg.norm(np.asarray(obj['size']))/2+.018)
    row={'time':float(r.sim.data.time),'body':obj['body'],'bilateral_fraction':float(np.mean(contacts)),
         'relative_drift_m':maximum_drift,'relative_distance_m':maximum_distance,
         'geometry_bound_m':geometry_bound,'size_m':list(obj['size']),
         'criterion':'edge_book_geometry_and_original_temporal_contact_drift_v1'}
    if not hasattr(r,'grasp_witnesses'):r.grasp_witnesses=[]
    r.grasp_witnesses.append(row)
    return row['bilateral_fraction']>=.9 and maximum_drift<.015 and maximum_distance<geometry_bound


def long_edge_grasp_frame(rotation,size):
    """Pinch normal to the long edge so later drawer alignment clears wrists."""
    axis=np.asarray(rotation).reshape(3,3)[:,1].copy();axis[2]=0.
    axis/=max(np.linalg.norm(axis),1e-12)
    if axis[0]<0:axis=-axis
    support=float(axis[0]*size[1]/2)
    if support<.032:
        raise TaskFailure('book_edge_orientation','Long edge cannot overhang the west table side with its centre safely supported')
    angle=np.arctan2(axis[1],axis[0]);c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@side_rotation()
    return axis,support,rotation


def supported_push_halfwidth(rotation,size,prefer_normal,allow_fallback=False):
    """Re-evaluate the actual rotated edge, keeping the support threshold.

    A book may rotate during physical pushing. Eligibility established before
    contact is not a guarantee that the long edge stays suitable afterward.
    """
    if abs(np.asarray(rotation).reshape(3,3)[2,2])<.98:
        raise TaskFailure('book_edge_orientation','Book is not flat enough for a supported edge fallback')
    if prefer_normal:
        try:return long_edge_grasp_frame(rotation,size)[1],True
        except TaskFailure as exc:
            if not allow_fallback or exc.category!='book_edge_orientation':raise
    return book_centerline_halfwidth(rotation,size),False


def pick_pushed_book(r,obj,grid):
    raised=(r.config.get('book_push_pose_profile')=='raised_tilt_v2' and
            obj.get('rank') in r.config.get('book_raised_tilt_push_ranks',[1,2]))
    if not raised:return _pick_pushed_book(r,obj,grid,False)
    class PushClearanceGuard:
        def before_step(self):pass
        def after_step(self):
            model=r.sim.model
            for contact in r.sim.data.contact:
                a,b=map(int,contact.geom);ba,bb=map(int,model.geom_bodyid[[a,b]])
                if (ba in r._robot)==(bb in r._robot):continue
                other=b if ba in r._robot else a
                if (model.geom(other).name or '').endswith('_top') and contact.dist<-.002:
                    raise TaskFailure('book_push_table_collision',
                        f'{model.geom(a).name}/{model.geom(b).name}: penetration {-float(contact.dist):.6f}m')
    guard=PushClearanceGuard();r.sim.add_step_controller(guard)
    try:return _pick_pushed_book(r,obj,grid,True)
    finally:r.sim.remove_step_controller(guard)


def _pick_pushed_book(r,obj,grid,raised):
    normal_grasp=(r.config.get('book_edge_grasp_profile')=='long_edge_normal_v2'
                  and obj.get('rank') in r.config.get('book_edge_normal_grasp_ranks',[2]))
    body=r.sim.model.body(obj['body']).id
    def refresh_edge(rotation,phase):
        nonlocal normal_grasp
        width,still_normal=supported_push_halfwidth(rotation,obj['size'],normal_grasp,
            allow_fallback=bool(r.config.get('book_edge_normal_fallback_centerline',False)))
        if normal_grasp and not still_normal and r.output:
            row={'time':float(r.sim.data.time),'phase':phase,'body':obj['body'],
                 'requested':'long_edge_normal_v2','actual':'original_centerline_side_grasp',
                 'reason':'Physical push rotated the long-edge normal outside supported geometry',
                 'book_rotation':np.asarray(rotation).tolist(),'centerline_halfwidth_m':width,
                 'minimum_long_edge_support_m':.032,'no_object_pose_assignment':True}
            with (r.output/('book_edge_dynamic_fallback_'+obj['body']+'.jsonl')).open('a') as stream:
                stream.write(__import__('json').dumps(row)+'\n')
        normal_grasp=still_normal
        return width
    if normal_grasp and r.config.get('book_edge_normal_fallback_centerline',False):
        rotation=r.sim.data.xmat[body].reshape(3,3)
        supported=abs(rotation[0,1])*obj['size'][1]/2>=.032
        if not supported:
            normal_grasp=False
            if r.output:
                (r.output/('book_edge_profile_'+obj['body']+'.json')).write_text(__import__('json').dumps({
                    'requested':'long_edge_normal_v2','actual':'original_centerline_side_grasp',
                    'reason':'Long edge cannot overhang while retaining table support; retain original physical edge strategy and strict insertion preflight'},indent=2))
    center=r.sim.data.xpos[body].copy()
    safe_y=float(np.clip(center[1],float(r.config.get('book_safe_bay_lower_y',-.80)),
                        float(r.config.get('book_safe_bay_upper_y',-.65))))
    r.transition(State.NAVIGATE,'oracle_book_safe_operation_bay',target=[0,safe_y,0],timeout=120)
    r.stow();r.backout(-.45);r.navigate([-.45,safe_y,0],grid)
    push_rotation=np.diag([1.,-1.,-1.]);entry_height=.731
    if raised:
        angle=-np.pi/6
        push_rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])@push_rotation
        entry_height=.755
        if r.config.get('book_raised_push_entry_by_rank') is not None:
            entry_height=float(r.config['book_raised_push_entry_by_rank'][int(obj['rank'])])
        # Collision prechecks must use the real closed pusher geometry.
        r.gripper.command(1);r.hold(.5)
    dock_pose=[0,safe_y,0]
    if r.config.get('book_push_base_precheck',False):
        from robot_exam.control.base_pose import evaluate_operational_base_poses
        object_r=r.sim.data.xmat[body].reshape(3,3)
        hx=float(np.sum(np.abs(object_r[0,:2])*np.asarray(obj['size'][:2])/2))
        far_point=center[0]+hx+.035;rotation=push_rotation
        targets=[('above_book_push',np.array([far_point,center[1],.84]),rotation,[obj['body']]),
                 ('book_push_entry',np.array([far_point,center[1],entry_height]),rotation,[obj['body']])]
        previous_grid=getattr(r,'grid',None);r.grid=None
        try:
            # The global route already reaches the same collision-inflated
            # safe staging pose. This search is for the short physical dock
            # and arm margin, not an A* path through the inflated table.
            report=evaluate_operational_base_poses(r,[.045,safe_y,0],targets,
                offsets=[(0,0),(.02,0),(.035,0)])
        finally:r.grid=previous_grid
        if r.output:
            (r.output/('book_push_base_'+obj['body']+'.json')).write_text(__import__('json').dumps(report,indent=2))
        if report['selected'] is None:
            raise TaskFailure('ik_unreachable','No collision-safe physical push bay reaches both above-book and contact entry targets')
        dock_pose=report['selected']['pose']
    if r.config.get('book_use_certified_stopped_bay',False) and r.config.get('book_push_base_precheck',False):
        _certified_push_bay(r,dock_pose,targets,obj['body'])
    else:r.dock(dock_pose,timeout=45)
    r.transition(State.MANIPULATE,'oracle_push_book_edge',target=obj['body'],timeout=180)
    r.gripper.command(1);r.hold(.5)
    center=r.sim.data.xpos[body].copy();initial=center.copy()
    R=r.sim.data.xmat[body].reshape(3,3);halfx=float(np.sum(np.abs(R[0,:2])*np.asarray(obj['size'][:2])/2))
    far=center[0]+halfx+.035
    r.move([far,center[1],.84],push_rotation,seconds=3);r.move([far,center[1],entry_height],push_rotation,seconds=2)
    push_trace=[]
    if r.config.get('book_feedback_push',False):
        # Oracle geometry expert: stop on the actual near edge, not on a
        # commanded hand displacement. The old open-loop push could move a
        # small book's centre of mass past the 0.28 m table support boundary.
        for step in range(80):
            center_now=r.sim.data.xpos[body].copy();Rnow=r.sim.data.xmat[body].reshape(3,3)
            hx=(refresh_edge(Rnow,'during_push') if normal_grasp and r.config.get('book_dynamic_edge_fallback',False) else
                long_edge_grasp_frame(Rnow,obj['size'])[1] if normal_grasp else
                book_centerline_halfwidth(Rnow,obj['size']) if r.config.get('book_push_centerline_clearance',False)
                else float(np.sum(np.abs(Rnow[0,:2])*np.asarray(obj['size'][:2])/2)))
            near=float(center_now[0]-hx)
            push_trace.append({'time':float(r.sim.data.time),'near_edge_x_m':near,
                               'center':center_now.tolist(),'pinch_x_m':float(r.arm.pose()[0][0])})
            if near<=.252:break
            if center_now[0]<.282 or center_now[2]<.725:
                raise TaskFailure('book_fell','Feedback push lost support before grasp clearance')
            r.move([far-.005*(step+1),center[1],entry_height],push_rotation,seconds=.12)
        else:raise TaskFailure('push_timeout','Bounded edge feedback did not reach safe overhang')
    else:r.move([.230+2*halfx,center[1],entry_height],push_rotation,seconds=4)
    r.move(r.arm.pose()[0]+[0,0,.12],push_rotation,seconds=2);r.hold(.4)
    center=r.sim.data.xpos[body].copy()
    if center[2]<.725:raise TaskFailure('book_fell','Book lost table support during push')
    R=r.sim.data.xmat[body].reshape(3,3)
    if normal_grasp and r.config.get('book_dynamic_edge_fallback',False):
        refresh_edge(R,'after_withdrawal_before_side_grasp')
    halfx=(book_centerline_halfwidth(R,obj['size']) if r.config.get('book_push_centerline_clearance',False)
           else float(np.sum(np.abs(R[0,:2])*np.asarray(obj['size'][:2])/2)))
    if normal_grasp:
        approach,_,rotation=long_edge_grasp_frame(R,obj['size'])
        grasp=center-approach*(obj['size'][1]/2-.015);pre=grasp-.075*approach
        if r.output:
            (r.output/('book_normal_grasp_'+obj['body']+'.json')).write_text(__import__('json').dumps({
                'approach':approach.tolist(),'grasp':grasp.tolist(),'pregrasp':pre.tolist(),
                'rotation':rotation.tolist(),'book_rotation':R.tolist(),'size':obj['size'],
                'reason':'Grasp long-edge normal; later book-axis alignment can leave the bulky gripper outside the drawer front'},indent=2))
    else:
        grasp=center.copy();grasp[0]-=halfx-.015;pre=grasp-[.075,0,0];rotation=side_rotation()
    if grasp[0]>.27:raise TaskFailure('edge_clearance','Pushed edge still lacks lower-finger clearance')
    if r.config.get('book_separate_push_grasp_profiles',False):
        r.transition(State.APPROACH,'side_book_edge_grasp',target=obj['body'],timeout=120)
    r.stow();r.backout(-.30);r.gripper.command(0)
    r.move([pre[0],pre[1],.90],rotation,seconds=3)
    path=None;failures=[]
    for seed in (r.qgoal,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
        try:
            q,ik=r.arm.solve(pre,rotation,seed=seed,max_iterations=500)
            path,planning=plan_joint_rrt(r.arm,q,allowed_bodies=[obj['body']]);break
        except RuntimeError as exc:failures.append(str(exc))
    if path is None:raise TaskFailure('book_joint_path',' | '.join(failures))
    if r.config.get('retime_checked_joint_paths',False):r.joint_path(path,seconds=2,_collision_checked=True)
    else:
        for q in path:r.joint_move(q,seconds=2)
    r.move(grasp,rotation,seconds=1.5);r.gripper.command(1);r.hold(.8)
    if not r.bilateral(obj['body']):raise TaskFailure('grasp_contact','No physical bilateral book contact')
    tip,actual_rotation=r.arm.pose()
    relative_before_lift=actual_rotation.T@(r.sim.data.xpos[body]-tip)
    if r.config.get('book_separate_push_grasp_profiles',False):
        r.transition(State.MANIPULATE,'side_book_lift_hold',target=obj['body'],timeout=60)
    r.move(grasp+[0,0,.10],rotation,seconds=2)
    lift=float(r.sim.data.xpos[body,2]-center[2])
    stable=(stable_edge_book(r,obj,relative_before_lift) if r.config.get('book_geometry_aware_grasp_witness',False)
            else r.stable_grasp(obj['body'],seconds=.6)) if lift>.055 else False
    if lift<=.055 or not stable:
        raise TaskFailure('gripper_slip','Book side grip did not retain a physical lift')
    return {'rotation':rotation,'lift_m':lift,'candidate_count':len(failures)+1,
            'scope':'Oracle physical push-and-side-pinch','push_displacement_m':(center-initial).tolist(),
            'joint_path_planning':planning,'ik':ik,'push_feedback_trace':push_trace}


def _certified_push_bay(r,nominal,targets,body):
    """Choose a real stopped, IK-feasible bay instead of chasing an arbitrary XY.

    Nominal workspace coordinates are design variables, not task scoring.
    The new actual pose must pass the same collision/IK tests and original
    strict dock stability criteria. This is not a claim of reaching nominal.
    """
    from robot_exam.control.base_pose import evaluate_operational_base_poses
    from robot_exam.control.base import wrap_angle
    goal=np.asarray(nominal);start=float(r.sim.data.time)
    while True:
        pose=r.base.pose();dx=goal[0]-pose[0];dy=goal[1]-pose[1]
        if abs(dx)<.035 and abs(dy)<.04:break
        if r.sim.data.time-start>10.:
            r.base.command(0,0);return r.dock(nominal,timeout=45)
        yaw=wrap_angle(goal[2]-pose[2])
        velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
        v=float(np.clip(1.2*dx-.4*velocity[0],-.22,.22)) if abs(yaw)<.12 else 0.
        r.base.command(v,float(np.clip(3*yaw+2*dy-.5*velocity[5],-.7,.7)));r.tick()
        if r.collision_events:raise TaskFailure('base_collision','Collision while approaching physical push workspace')
    r.base.command(0,0);r.hold(.6);actual=r.base.pose().copy()
    previous_grid=getattr(r,'grid',None);r.grid=None
    try:report=evaluate_operational_base_poses(r,actual,targets,offsets=[(0,0,0)])
    finally:r.grid=previous_grid
    if r.output:
        (r.output/f'certified_push_bay_{body}.json').write_text(__import__('json').dumps(
            {'nominal':goal.tolist(),'actual_selected':actual.tolist(),'feasibility':report,
             'scope':'New feasible operating pose, not nominal dock accuracy'},indent=2)+'\n')
    if report['selected'] is None:return r.dock(nominal,timeout=45)
    return r.dock(actual,timeout=10.,tolerance=.023)
