"""Reusable physical object and constrained-joint skills for task FSMs."""
import numpy as np
from robot_exam.tasks.runtime import State,TaskFailure
from robot_exam.perception.objects import grasp_candidates


def pick(runtime,body,detection,candidate_index=0):
    r=runtime
    r.transition(State.PLAN,'multi_candidate_grasp',target=detection,timeout=30)
    category=detection.get('category','')
    oracle_clearance=bool(r.config.get('oracle_grasp_clutter_clearance',False))
    if oracle_clearance and r.config.get('collection_expert_source')!='oracle':
        raise ValueError('Scene-truth grasp clearance requires a declared Oracle collection expert')
    candidates=grasp_candidates(r.arm,{**detection,'target_body':body,
        'oracle_avoid_other_fruit_contacts':oracle_clearance,
        'grasp_z_offset_m':float(r.config.get(category+'_grasp_z_offset_m',0.))})
    if not candidates:raise TaskFailure('ik_unreachable','No width- and IK-feasible grasp candidate')
    candidate=candidates[candidate_index%len(candidates)];p=candidate['position'];R=candidate['rotation']
    r.gripper.command(0)
    r.transition(State.APPROACH,'pregrasp',target=p.tolist())
    r.move(candidate.get('pregrasp',p+[0,0,.085]),R,seconds=2.0)
    r.move(p,R,seconds=1.5)
    r.transition(State.GRASP,'close_and_check_contact',target=body,timeout=3)
    from robot_exam.control.contact_grip import enabled_for
    if enabled_for(r,category):
        from robot_exam.control.contact_grip import close_until_bilateral_contact
        try:close_until_bilateral_contact(r,body,float(r.config.get('fruit_target_contact_force_n',12.)))
        except RuntimeError as exc:raise TaskFailure('grasp_contact',str(exc)) from exc
    else:r.gripper.command(1);r.hold(.8)
    if not r.bilateral(body):raise TaskFailure('grasp_contact','No bilateral target contacts')
    initial=float(r.sim.data.xpos[r.sim.model.body(body).id,2]) # scoring/contact verification only
    r.transition(State.MANIPULATE,'lift_and_hold',target=body)
    r.move(p+[0,0,.095],R,seconds=1.8)
    r.hold(.6)
    height=float(r.sim.data.xpos[r.sim.model.body(body).id,2]-initial)
    if height<.055 or not r.bilateral(body):raise TaskFailure('gripper_slip',f'Lift/contact verification failed: {height:.4f}m')
    return {'candidate_score':candidate['score'],'candidate_count':len(candidates),'rotation':R,'lift_m':height,'position':p}


def release(runtime,position,rotation,retreat_offsets=None):
    r=runtime;r.transition(State.PLACE,'release',target=list(position))
    r.move(position,rotation,seconds=1.8)
    retreat=np.asarray(position)+[0,0,.06]
    if retreat_offsets is not None:
        # Choose a reachable exit before opening the gripper. The normal
        # vertical exit remains first; a short backward exit can avoid the
        # UR5 wrist limit at a physically imperfect bin dock.
        retreat=None
        for offset in retreat_offsets:
            candidate=np.asarray(position)+np.asarray(offset)
            try:r.solve_ik(candidate,rotation,seed=r.qgoal,context='fruit_release_retreat_precheck')
            except RuntimeError:continue
            retreat=candidate;break
        if retreat is None:
            raise TaskFailure('ik_unreachable','No feasible release retreat before opening gripper')
    r.gripper.command(0);r.hold(.6)
    r.move(retreat,rotation,seconds=1.3)
    r.hold(.6)


def side_rotation():
    # Gripper +Z forward(+X), closing direction +Y; used on horizontal pulls.
    return np.array([[0,0,1],[1,0,0],[0,1,0.]])


def drawer_operation(runtime,layer,opening,metadata):
    """Oracle geometric constraint expert. Passive drawer, no drawer actuator."""
    r=runtime;m,d=r.sim.model,r.sim.data
    joint=m.joint(f'drawer_slide_{layer}').id;adr=m.jnt_qposadr[joint]
    initial=float(d.qpos[adr]);z=metadata['drawer_bottoms'][layer]+metadata.get('drawer_handle_offset',.085)
    if layer==0 and initial>.10:
        # Back up physically to preserve shoulder clearance at the open handle.
        r.dock([-.30,.50,0],timeout=8)
    if layer==2 and initial>.10:
        # Open upper handle is too close for the side-grasp wrist branch.
        # Recover working distance with real wheels before re-approaching.
        r.backout(-.35)
    handle=np.array([.58-.055-initial,.50,z])
    # Fingers close vertically around the horizontal Y handle.
    R=np.array([[0,0,1],[1,0,0],[0,1,0.]])
    r.transition(State.APPROACH,'drawer_handle_'+str(layer),target=handle.tolist())
    # Reconfigure above the base before lowering the wrist in front of it.
    # The alternate wrist branch prevents forearm/chassis intersection.
    staging=np.array([.43,.5,max(.84,z+.06)])
    if layer==2 and initial>.10:staging[0]=handle[0]-.09
    safe_q,_=r.solve_ik(staging,R,seed=[0,-1.5,2,-2,1.57,0],context=f'drawer_{layer}_staging')
    r.joint_move(safe_q,5)
    r.gripper.command(0)
    feedback=bool(r.config.get('drawer_handle_feedback',False))
    def refresh_handle(phase):
        nonlocal handle
        measured=float(d.qpos[adr]);actual=np.array([.58-.055-measured,.50,z])
        if r.output and (feedback or r.config.get('drawer_handle_geometry_trace',False)):
            row={'time':float(d.time),'phase':phase,'layer':layer,'measured_opening':measured,
                 'previous_target':handle.tolist(),'measured_target':actual.tolist(),
                 'target_shift_m':float(np.linalg.norm(actual-handle)),'feedback_enabled':feedback}
            with (r.output/'drawer_handle_feedback.jsonl').open('a') as stream:stream.write(__import__('json').dumps(row)+'\n')
        if feedback:handle=actual
    refresh_handle('after_staging_before_pregrasp')
    if layer==2:
        pre_q,_=r.solve_ik(handle+[-.09,0,0],R,seed=safe_q,context=f'drawer_{layer}_pregrasp')
        r.joint_move(pre_q,3.)
    else:r.move(handle+[-.09,0,0],R,seconds=2)
    refresh_handle('after_pregrasp_before_contact')
    r.move(handle,R,seconds=1.5)
    r.gripper.command(1);r.hold(.8)
    if not r.bilateral(f'drawer_{layer}'):
        raise TaskFailure('handle_grasp','No bilateral drawer handle contact')
    if feedback:initial=float(d.qpos[adr])
    r.transition(State.MANIPULATE,'drawer_slide_'+str(layer),target=float(opening),timeout=60)
    for value in np.linspace(initial,opening,18)[1:]:
        target=np.array([.58-.055-value,.50,z])
        r.move(target,R,seconds=.30,tolerance=.025)
    r.hold(.4)
    actual=float(d.qpos[adr])
    if opening<.01 and actual>.010:
        # A pull grip can slip longitudinally while closing. Use bounded
        # physical fingertip pressure on the front, stopping on actual slide
        # closure rather than demanding penetration through a closed panel.
        r.transition(State.MANIPULATE,'drawer_contact_push_'+str(layer),target=0.)
        for extra in np.linspace(.005,.085,20):
            goal=np.array([.525+extra,.50,z])
            r.qgoal,_=r.solve_ik(goal,R,seed=r.qgoal,context=f'drawer_{layer}_closure_push')
            r.hold(.18)
            actual=float(d.qpos[adr])
            if actual<.010:break
    r.gripper.command(0);r.hold(.4)
    if layer==2 and r.config.get('collision_planned_joint_moves',False):
        # The old near-base Cartesian withdrawal folded wrist_2 into the
        # forearm. Release first, then gain clearance with real wheel motion;
        # collision-plan the stow from the retained safe handle configuration.
        r.transition(State.NAVIGATE,'upper_drawer_released_clearance',timeout=60)
        r.backout(-.50);r.stow()
        actual=float(d.qpos[adr])
        if abs(actual-opening)>.025:raise TaskFailure('drawer_stuck',f'Drawer {layer}: requested {opening}, actual {actual}')
        return {'layer':layer,'initial':initial,'target':opening,'actual':actual,'error':abs(actual-opening),
                'expert':'oracle joint constraint geometry; released wheel-clearance retreat'}
    r.move(r.arm.pose()[0]+[-(.07 if layer==2 else .04),0,0],R,seconds=1.0)
    retreat=r.arm.pose()[0]+[0,0,.10]
    retreat[2]=min(retreat[2],.92 if layer==2 else .94)
    r.move(retreat,R,seconds=1.2)
    if abs(actual-opening)>.025:raise TaskFailure('drawer_stuck',f'Drawer {layer}: requested {opening}, actual {actual}')
    return {'layer':layer,'initial':initial,'target':opening,'actual':actual,'error':abs(actual-opening),'expert':'oracle joint constraint geometry'}
