"""Books sorted by observed planar area, passive drawers, return-home FSM."""
import numpy as np
from robot_exam.tasks.runtime import State,TaskFailure
from robot_exam.tasks.manipulation import pick,release,drawer_operation
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.perception.objects import observe_objects
from robot_exam.tasks.scoring import drawer_score
from robot_exam.control.base_pose import evaluate_operational_base_poses
from robot_exam.perception.objects import grasp_candidates


def execute(runtime,metadata,mode='vision'):
    r=runtime;initial=np.asarray(metadata['start']);grid=OccupancyGrid();r.grid=grid
    for box in metadata['obstacles']:grid.rectangle(*box)
    r.hold(.6);r.transition(State.NAVIGATE,'five_metre_book_table')
    r.navigate([-.40,-.65,0],grid);r.dock(metadata['book_dock'])
    camera=RGBDCamera(r.sim);operations=[]
    try:
        r.transition(State.OBSERVE,'rgbd_book_size_sort')
        rgb,depth=camera.capture()
        if r.output:camera.save(r.output/'books_observation',rgb,depth)
        workspace=metadata.get('observation_workspace',(.34,.64,-1.,-.30))
        observed=observe_objects(camera,rgb,depth,workspace=workspace,kind='book')
        if mode=='oracle':
            observed=[{'center':r.sim.data.xpos[r.sim.model.body(o['body']).id].tolist(),'category':o['body'],'yaw':o['yaw'],
                       'length':o['size'][0],'width':o['size'][1],'area':o['area'],'pca_ratio':4} for o in metadata['objects']]
        if len(observed)!=3:raise TaskFailure('perception',f'Expected 3 books, observed {len(observed)}')
        observed.sort(key=lambda o:o['area'])
        for rank,book in enumerate(observed):
            r.transition(State.NAVIGATE,'cabinet_open_layer_'+str(rank))
            r.stow();r.backout(-.40);r.navigate([-.45,.50,0],grid);r.dock(metadata['cabinet_dock'])
            opened=drawer_operation(r,rank,.26,metadata)
            # Drawer contact operation has ended. Its 60 s deadline must not
            # keep running during the subsequent drive back to the worktable.
            r.transition(State.NAVIGATE,'return_to_book_table',timeout=240)
            r.backout(-.50);r.stow();r.navigate([-.40,-.65,0],grid);r.dock(metadata['book_dock'])
            r.transition(State.OBSERVE,'reobserve_selected_book')
            rgb,depth=camera.capture();detections=observe_objects(camera,rgb,depth,workspace=workspace,kind='book')
            if mode=='vision':
                matches=[d for d in detections if d['category']==book['category']]
                if not matches:raise TaskFailure('perception','Selected book lost from RGB-D')
                book=matches[0]
            # Truth association is for contact instrumentation, not target generation.
            target=min(metadata['objects'],key=lambda o:np.linalg.norm(r.sim.data.xpos[r.sim.model.body(o['body']).id]-book['center']))
            book={**book,'target_body':target['body']}
            if r.config.get('book_pick_strategy')=='oracle_push_side':
                if mode!='oracle':raise TaskFailure('policy_scope','Oracle book push strategy requires explicit oracle mode')
                from robot_exam.skills.book_edge import pick_pushed_book
                grasp=pick_pushed_book(r,target,grid)
                # Continue the same physical transport/place/close/home flow.
                _place_side_book(r,metadata,rank,book,target,grasp,grid,opened,operations)
                continue
            candidates=grasp_candidates(r.arm,book,solve_ik=False)
            if not candidates:raise TaskFailure('ik_unreachable','No book grasp candidate before base-pose search')
            r.backout(-.55)
            nominal=np.asarray(metadata['book_dock'],dtype=float)
            reports=[]
            for candidate_index,candidate in enumerate(candidates):
                report=evaluate_operational_base_poses(r,nominal,
                    [('book_pregrasp',candidate['pregrasp'],candidate['rotation'],[target['body']]),
                     ('book_grasp',candidate['position'],candidate['rotation'],[target['body']])],
                    offsets=[(0,0),(-.08,0),(.08,0),(0,-.06),(0,.06),
                             (0,0,-.20),(0,0,.20),(-.06,0,-.12),(-.06,0,.12)])
                reports.append({'candidate_index':candidate_index,'position':candidate['position'].tolist(),
                                'grasp_type':candidate['grasp_type'],
                                'approach_axis':candidate.get('approach_axis'),
                                'report':report})
            feasible=[(x['report']['selected']['score']+candidates[x['candidate_index']]['score'],x)
                      for x in reports if x['report']['selected'] is not None]
            base_report={'grasp_candidates':reports,'selected':None}
            if feasible:
                _,chosen_trial=min(feasible,key=lambda x:x[0]);base_report['selected']=chosen_trial
            if r.output:(r.output/'book_base_pose_trials.json').write_text(__import__('json').dumps(base_report,indent=2))
            if base_report['selected'] is None:
                raise TaskFailure('ik_unreachable','No collision-free operational base pose can reach any selected book grasp candidate')
            best=candidates[base_report['selected']['candidate_index']]
            chosen=np.asarray(base_report['selected']['report']['selected']['pose']);approach=chosen.copy();approach[0]-=.40*np.cos(approach[2]);approach[1]-=.40*np.sin(approach[2])
            r.transition(State.NAVIGATE,'optimize_base_pose_for_book_grasp',target=chosen.tolist())
            r.navigate(approach,grid,timeout=90);r.dock(chosen,timeout=20)
            reachable=grasp_candidates(r.arm,book)
            if not reachable:raise TaskFailure('ik_unreachable','Selected base pose lost all collision-free book IK candidates')
            selected_index=int(np.argmin([np.linalg.norm(c['position']-best['position']) for c in reachable]))
            grasp=pick(r,target['body'],book,candidate_index=selected_index)
            r.joint_move(r.STOW,4.) # keep gripper closed while driving
            r.backout(-.40);r.navigate([-.45,.50,0],grid);r.dock(metadata['cabinet_dock'])
            if not r.bilateral(target['body']):raise TaskFailure('gripper_slip','Book slipped during mobile transport')
            if rank==2:
                # The raised open-drawer handle intersects a low STOW payload.
                # Lift the carried book above it before advancing the wheels.
                r.move(r.arm.pose()[0]+[0,0,.10],r.arm.pose()[1],seconds=2.)
                r.dock([0,.50,0])
                r.move([.45,.50,.94],grasp['rotation'],seconds=2.)
            height=.725+2*(book['center'][2]-.725) # observed top surface -> observed thickness
            thickness=max(.008,2*(book['center'][2]-.725))
            destination=np.array([.45,.50,metadata['drawer_bottoms'][rank]+.008+thickness/2+.003])
            r.transition(State.APPROACH,'drawer_insertion_'+str(rank))
            pre=destination.copy();pre[2]=max(.80,destination[2]+.085)
            r.move(pre,grasp['rotation'],seconds=2.8)
            release(r,destination,grasp['rotation'])
            if rank==2:r.move([destination[0],.50,.94],grasp['rotation'],seconds=2.)
            r.stow()
            if rank==2:r.backout(-.20)
            closed=drawer_operation(r,rank,0.,metadata)
            operations.append({'rank_from_visual_area':rank,'book':target['body'],'area':book['area'],'opened':opened,'closed':closed,'lift_m':grasp['lift_m']})
            r.backout(-.50);r.stow()
        r.transition(State.RETURN_HOME,'astar_return_to_saved_start')
        r.backout(-.50);r.navigate(initial,grid,timeout=160)
        r.hold(1);score=drawer_score(r.sim,metadata,initial)
        return {'success':score['success'],'score':score,'operations':operations,'mode':mode,'area_rule':'observed length*width'}
    except Exception as exc:
        exc.partial={'operations':operations,'score':drawer_score(r.sim,metadata,initial)};raise
    finally:camera.close()


def _place_side_book(r,metadata,rank,book,target,grasp,grid,opened,operations):
    """Development side-grip placement; all endpoints remain physically scored."""
    from robot_exam.tasks.manipulation import drawer_operation
    r.transition(State.NAVIGATE,'side_book_transport',timeout=180)
    station=np.asarray(metadata['cabinet_dock'],dtype=float).copy()
    station[0]=float(r.config.get('side_book_cabinet_x',station[0]))
    world_carry=bool(r.config.get('side_book_world_carry',False));previous_precision=getattr(r,'payload_docking',False)
    joint_carry=bool(r.config.get('side_book_joint_carry',False))
    if r.config.get('side_book_joint_carry_rank') is not None:
        joint_carry=joint_carry and rank==int(r.config['side_book_joint_carry_rank'])
    if world_carry or joint_carry:
        r.payload_motion_profile='book'
        r.begin_payload(target['body'],grasp['rotation'],world_stabilize=not joint_carry)
        r.payload_docking=True
    else:r.joint_move(r.STOW,4)
    try:
        r.backout(-.45)
        staging_x=min(-.45,station[0]-.15)
        if joint_carry:
            # A large side-held book at table height otherwise hits the open
            # top drawer's front and physically pushes it closed before drop.
            # Stay clear of that front, then raise with real arm actuators.
            staging_x=min(staging_x,-.55)
        r.navigate([staging_x,.50,0],grid)
        if joint_carry:
            from robot_exam.tasks.scoring import body_bounds
            lower,_=body_bounds(r.sim,target['body'])
            # Include the protruding handle and supports, not just the panel.
            _,drawer_upper=body_bounds(r.sim,f'drawer_{rank}')
            dz=max(0.,drawer_upper[2]+.04-lower[2])
            if dz:
                tip,rotation=r.arm.pose()
                r.move(tip+[0,0,dz],rotation,seconds=3.)
        r.dock(station,timeout=30)
    finally:
        if world_carry or joint_carry:r.end_payload();r.payload_docking=previous_precision
    if not r.bilateral(target['body']):raise TaskFailure('gripper_slip','Side-held book lost during transport')
    if joint_carry:
        adr=r.sim.model.jnt_qposadr[r.sim.model.joint(f'drawer_slide_{rank}').id]
        if r.sim.data.qpos[adr]<.18:
            raise TaskFailure('drawer_not_open','Open drawer was displaced before book release')
    r.transition(State.PLACE,'side_book_drawer_insertion',timeout=120)
    height=metadata['drawer_bottoms'][rank]+.008+target['size'][2]/2+float(r.config.get('side_book_release_clearance_m',.025))
    place_rotation=np.asarray(grasp['rotation']).copy()
    pre_height=min(max(.90,height+.16),float(r.config.get('side_book_max_pre_height_m',2.0)))
    if pre_height<height+.02:raise TaskFailure('placement_unreachable','No clearance above the release plane')
    # Restore horizontal book orientation above the drawer before lowering.
    def checked_move(position):
        if not r.config.get('side_book_place_joint_path',False):
            return r.move(position,place_rotation,seconds=3)
        from robot_exam.control.joint_rrt import plan_joint_rrt
        errors=[]
        for seed in (r.qgoal,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
            try:
                q,_=r.arm.solve(position,place_rotation,seed=seed,max_iterations=500)
                path,_=plan_joint_rrt(r.arm,q,allowed_bodies=[target['body']])
                for waypoint in path:r.joint_move(waypoint,seconds=2.)
                error=float(np.linalg.norm(r.arm.pose()[0]-position))
                if error>.018:raise TaskFailure('arm_tracking',f'Collision-planned book endpoint error {error}')
                return
            except RuntimeError as exc:
                if isinstance(exc,TaskFailure):raise
                errors.append(str(exc))
        raise TaskFailure('book_joint_path',' | '.join(errors))
    checked_move(np.array([.36,.50,pre_height]))
    if not r.bilateral(target['body']):
        raise TaskFailure('gripper_slip','Book contact lost during above-drawer reorientation')
    if r.config.get('side_book_align_long_axis',False):
        # Explicit Oracle geometry feedback: align the rectangular long axis
        # with the drawer width before dropping past the cabinet roof edge.
        object_rotation=r.sim.data.xmat[r.sim.model.body(target['body']).id].reshape(3,3)
        yaw=np.arctan2(object_rotation[1,0],object_rotation[0,0]);angle=np.pi/2-yaw
        c,s=np.cos(angle),np.sin(angle)
        place_rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])@place_rotation
        checked_move(np.array([.36,.50,pre_height]))
    offset=r.sim.data.xpos[r.sim.model.body(target['body']).id]-r.arm.pose()[0]
    if not r.bilateral(target['body']) or np.linalg.norm(offset)>.20:
        raise TaskFailure('gripper_slip','Invalid held-book offset after orientation; not an IK reachability failure')
    destination=np.array([float(r.config.get('side_book_drop_center_x',.47)),.50,height])-offset
    checked_move(destination)
    r.gripper.command(0);r.hold(.8)
    r.move(destination+[-.14,0,.025],place_rotation,seconds=2)
    r.stow();closed=drawer_operation(r,rank,0.,metadata)
    operations.append({'rank_from_visual_area':rank,'book':target['body'],'area':book['area'],
                       'opened':opened,'closed':closed,'lift_m':grasp['lift_m'],'strategy':grasp['scope']})
    r.backout(-.50);r.stow()
