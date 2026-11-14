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
    if runtime.config.get('book_cabinet_station') is not None:
        station=np.asarray(runtime.config['book_cabinet_station'],dtype=float)
        if station.shape!=(3,) or not np.isfinite(station).all():raise ValueError('Finite cabinet operating pose required')
        metadata={**metadata,'cabinet_dock':station.tolist()}
    r=runtime;initial=np.asarray(metadata['start']);grid=OccupancyGrid();r.grid=grid
    for box in metadata['obstacles']:grid.rectangle(*box)
    initial_cabinet=bool(mode=='oracle' and r.config.get('book_oracle_initial_cabinet_approach',False))
    r.hold(.6);r.transition(State.NAVIGATE,'five_metre_cabinet_preparation' if initial_cabinet else 'five_metre_book_table')
    if initial_cabinet:
        r.navigate([-.45,metadata['cabinet_dock'][1],0],grid);r.dock(metadata['cabinet_dock'])
    else:r.navigate([-.40,-.65,0],grid);r.dock(metadata['book_dock'])
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
            r.stow()
            if not(rank==0 and initial_cabinet):
                r.backout(-.40);r.navigate([-.45,metadata['cabinet_dock'][1],0],grid);r.dock(metadata['cabinet_dock'])
            opened=drawer_operation(r,rank,.26,metadata)
            # Drawer contact operation has ended. Its 60 s deadline must not
            # keep running during the subsequent drive back to the worktable.
            r.transition(State.NAVIGATE,'return_to_book_table',timeout=240)
            r.backout(-.50);r.stow()
            direct_push=(mode=='oracle' and r.config.get('book_pick_strategy')=='oracle_push_side'
                         and r.config.get('book_direct_push_bay',False))
            if not direct_push:r.navigate([-.40,-.65,0],grid);r.dock(metadata['book_dock'])
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
        if joint_carry and r.config.get('book_compact_held_transport',False):
            # Keep the jaws closed. The normal stow() opens them, so use a
            # collision-checked held-book joint path, never a rigid attachment.
            from robot_exam.control.joint_rrt import plan_joint_rrt
            compact_q=r.STOW
            if r.config.get('book_compact_target_in_base') is not None:
                relative=np.asarray(r.config['book_compact_target_in_base'],dtype=float)
                if relative.shape!=(3,) or not np.isfinite(relative).all():
                    raise ValueError('Finite compact carry x/y offsets and absolute height required')
                pose=r.base.pose();c,s=np.cos(pose[2]),np.sin(pose[2])
                xy=pose[:2]+np.array([[c,-s],[s,c]])@relative[:2]
                compact_q,_=r.solve_ik(np.r_[xy,relative[2]],r.arm.pose()[1],
                                       context='held_book_compact_pose')
            path,report=plan_joint_rrt(r.arm,compact_q,allowed_bodies=[target['body']],
                                       held_body=target['body'],wall_seconds=5.)
            r.ik_diagnostics.append({'context':'held_book_compact_transport','planning':report})
            r.joint_path(path,seconds=2.,_collision_checked=True)
            if not r.bilateral(target['body']):
                raise TaskFailure('gripper_slip','Book lost while folding for transport')
        r.backout(-.45)
        staging_x=min(-.45,station[0]-.15)
        if joint_carry:
            # A large side-held book at table height otherwise hits the open
            # top drawer's front and physically pushes it closed before drop.
            # Stay clear of that front, then raise with real arm actuators.
            staging_x=min(staging_x,-.55)
            if r.config.get('side_book_braking_clearance_staging',False):
                from robot_exam.tasks.scoring import body_bounds
                _,upper=body_bounds(r.sim,target['body']);tool=r.sim.model.body('tool0').id
                for bid in range(r.sim.model.nbody):
                    parent=bid
                    while parent>0 and parent!=tool:parent=int(r.sim.model.body_parentid[parent])
                    if parent==tool:
                        _,extent=body_bounds(r.sim,r.sim.model.body(bid).name)
                        if np.isfinite(extent).all():upper=np.maximum(upper,extent)
                front,_=body_bounds(r.sim,f'drawer_{rank}')
                carry_extent=float(upper[0]-r.base.pose()[0])
                speed=float(r.config.get('payload_linear_speed_m_s',.1))
                braking=float(r.config.get('payload_linear_braking_m_s2',.2))
                gap=speed*speed/(2*max(braking,.05))+.06
                staging_x=min(staging_x,float(front[0])-carry_extent-gap)
                if r.output:(r.output/f'book_braking_staging_{rank}.json').write_text(__import__('json').dumps(
                    {'drawer_front_x_m':float(front[0]),'assembly_extent_x_m':carry_extent,
                     'braking_and_margin_m':gap,'selected_staging_x_m':staging_x},indent=2)+'\n')
        workspace_transport=bool(r.config.get('book_workspace_transport',False) and joint_carry)
        if workspace_transport:
            from robot_exam.control.workspace_approach import corridor_to_y
            corridor_to_y(r,grid,float(station[1]),x_corridor=-1.,linear_limit=.25)
            grasp['rotation']=r.arm.pose()[1].copy()
        else:r.navigate([staging_x,station[1],0],grid)
        if joint_carry:
            from robot_exam.tasks.scoring import body_bounds
            lower,_=body_bounds(r.sim,target['body'])
            if r.config.get('side_book_assembly_clearance',False):
                # Clearance must include the physical couplers, gripper base
                # and camera bracket, not only the held rectangular book.
                model=r.sim.model;tool=model.body('tool0').id
                for body_id in range(model.nbody):
                    parent=body_id
                    while parent>0 and parent!=tool:parent=int(model.body_parentid[parent])
                    if parent==tool:
                        lo,_=body_bounds(r.sim,model.body(body_id).name)
                        if np.isfinite(lo).all():lower=np.minimum(lower,lo)
            # Include the protruding handle and supports, not just the panel.
            _,drawer_upper=body_bounds(r.sim,f'drawer_{rank}')
            dz=max(0.,drawer_upper[2]+.04-lower[2])
            if r.output:
                (r.output/('book_carry_clearance_'+target['body']+'.json')).write_text(__import__('json').dumps({
                    'time':float(r.sim.data.time),'lowest_carried_geometry_z':float(lower[2]),
                    'drawer_upper_z':float(drawer_upper[2]),'required_raise_m':float(dz),
                    'includes_end_effector':bool(r.config.get('side_book_assembly_clearance',False))},indent=2))
            if dz:
                tip,rotation=r.arm.pose()
                forward_shift=float(r.config.get('side_book_clearance_forward_shift_m',0.)) if r.config.get('book_compact_held_transport',False) else 0.
                if not 0.<=forward_shift<=.2:raise ValueError('Clearance approach shift must be in0–.2m')
                clearance_target=tip+[forward_shift,0,dz]
                if r.config.get('side_book_collision_planned_clearance',False):
                    _checked_side_book_move(r,clearance_target,rotation,target['body'])
                else:r.move(clearance_target,rotation,seconds=3.)
        if workspace_transport:
            from robot_exam.control.workspace_approach import approach_feasible_workspace
            body_id=r.sim.model.body(target['body']).id;tip,current_R=r.arm.pose()
            relative=current_R.T@(r.sim.data.xpos[body_id]-tip)
            object_R=r.sim.data.xmat[body_id].reshape(3,3)
            angle=rectangular_axis_alignment(np.arctan2(object_R[1,0],object_R[0,0]))
            c,s=np.cos(angle),np.sin(angle);release_R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@current_R
            clearance=float(r.config['side_book_release_clearance_by_rank'][rank])
            height=metadata['drawer_bottoms'][rank]+.008+target['size'][2]/2+clearance
            drop_x=float(r.config['side_book_drop_center_x_by_rank'][rank]);release_tip=np.array([drop_x,.5,height])-release_R@relative
            pre_height=min(max(.90,height+.16),float(r.config.get('side_book_max_pre_height_m',2.)))
            targets=[('above_drawer',[.36,.5,pre_height],current_R,[target['body']]),
                     ('release',release_tip,release_R,[target['body']]),
                     ('retreat',release_tip+[0,0,.06],release_R,[target['body']])]
            approach_feasible_workspace(r,targets,minimum_x=-.24,maximum_x=0.,linear_limit=.25,
                                        tolerance=.023,name=f'book_workspace_{rank}')
        else:r.dock(station,timeout=30)
    finally:
        if world_carry or joint_carry:r.end_payload();r.payload_docking=previous_precision
    if not r.bilateral(target['body']):raise TaskFailure('gripper_slip','Side-held book lost during transport')
    if joint_carry:
        adr=r.sim.model.jnt_qposadr[r.sim.model.joint(f'drawer_slide_{rank}').id]
        if r.sim.data.qpos[adr]<.18:
            raise TaskFailure('drawer_not_open','Open drawer was displaced before book release')
    return _insert_side_book(r,metadata,rank,book,target,grasp,opened,operations)


def _checked_side_book_move(r,position,rotation,body,allow_held_contact=True):
    from robot_exam.control.joint_rrt import plan_joint_rrt
    from robot_exam.control.book_path import plan_book_lowering, rotation_distance
    start,current_rotation=r.arm.pose()
    if (r.config.get('side_book_lowering_cartesian_path',False) and allow_held_contact
            and np.asarray(position)[2]<start[2]-.04 and rotation_distance(current_rotation,rotation)<.04):
        path,report=plan_book_lowering(r.arm,position,rotation,body)
        body_id=r.sim.model.body(body).id
        relative_p=current_rotation.T@(r.sim.data.xpos[body_id]-start)
        relative_r=current_rotation.T@r.sim.data.xmat[body_id].reshape(3,3)
        class HeldBookGuard:
            last=-float('inf');violations=0
            def before_step(self):pass
            def after_step(self):
                if r.sim.data.time-self.last<.02-1e-9:return
                self.last=float(r.sim.data.time)
                tip,actual_r=r.arm.pose()
                drift=float(np.linalg.norm(actual_r.T@(r.sim.data.xpos[body_id]-tip)-relative_p))
                turn=rotation_distance(relative_r,actual_r.T@r.sim.data.xmat[body_id].reshape(3,3))
                bad=not r.bilateral(body) or drift>.025 or turn>.35
                self.violations=self.violations+1 if bad else 0
                if self.violations>=3:
                    raise TaskFailure('gripper_slip',f'Book lowering contact guard: drift={drift:.4f}m turn={turn:.4f}rad')
        guard=HeldBookGuard();r.sim.add_step_controller(guard)
        try:
            r.joint_path(path,seconds=.7,
                _collision_checked=bool(r.config.get('side_book_execute_checked_path',False)))
        finally:r.sim.remove_step_controller(guard)
        error=float(np.linalg.norm(r.arm.pose()[0]-position))
        if error>.018:raise TaskFailure('arm_tracking',f'Cartesian book lowering endpoint error {error}')
        r.ik_diagnostics.append({'context':r.stage+':cartesian_book_lowering','planning':report})
        return
    errors=[]
    for seed in (r.qgoal,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
        try:
            q,_=r.arm.solve(position,rotation,seed=seed,max_iterations=500)
            path,report=plan_joint_rrt(r.arm,q,allowed_bodies=[body] if allow_held_contact else [],
                held_body=body if allow_held_contact and r.config.get('side_book_full_assembly_joint_path',False) else None)
            r.joint_path(path,seconds=2.,
                _collision_checked=bool(r.config.get('side_book_execute_checked_path',False)))
            error=float(np.linalg.norm(r.arm.pose()[0]-position))
            if error>.018:raise TaskFailure('arm_tracking',f'Collision-planned book endpoint error {error}')
            r.ik_diagnostics.append({'context':r.stage+':checked_book_move','planning':report})
            return
        except RuntimeError as exc:
            if isinstance(exc,TaskFailure):raise
            errors.append(str(exc))
    raise TaskFailure('book_joint_path',' | '.join(errors))


def _insert_side_book(r,metadata,rank,book,target,grasp,opened,operations):
    """Shared live insertion, also callable in documented checkpoint probes."""
    r.transition(State.PLACE,'side_book_drawer_insertion',timeout=120)
    def record_geometry(phase):
        if r.output:
            body=r.sim.model.body(target['body']).id;tip,rotation=r.arm.pose()
            row={'phase':phase,'time':float(r.sim.data.time),'tool_position':tip.tolist(),
                 'tool_rotation':rotation.tolist(),'book_position':r.sim.data.xpos[body].tolist(),
                 'book_rotation':r.sim.data.xmat[body].reshape(3,3).tolist()}
            with (r.output/('book_placement_geometry_'+target['body']+'.jsonl')).open('a') as stream:
                stream.write(__import__('json').dumps(row)+'\n')
    record_geometry('before_initial_restore')
    release_clearance=float(r.config.get('side_book_release_clearance_m',.025))
    if r.config.get('side_book_release_clearance_by_rank') is not None:
        release_clearance=float(r.config['side_book_release_clearance_by_rank'][rank])
    height=metadata['drawer_bottoms'][rank]+.008+target['size'][2]/2+release_clearance
    place_rotation=np.asarray(grasp['rotation']).copy()
    pre_height=min(max(.90,height+.16),float(r.config.get('side_book_max_pre_height_m',2.0)))
    if pre_height<height+.02:raise TaskFailure('placement_unreachable','No clearance above the release plane')
    # Restore horizontal book orientation above the drawer before lowering.
    def checked_move(position,allow_held_contact=True):
        if not r.config.get('side_book_place_joint_path',False):
            return r.move(position,place_rotation,seconds=3)
        return _checked_side_book_move(r,position,place_rotation,target['body'],allow_held_contact)
    checked_move(np.array([.36,.50,pre_height]))
    record_geometry('after_initial_restore')
    if not r.bilateral(target['body']):
        raise TaskFailure('gripper_slip','Book contact lost during above-drawer reorientation')
    if r.config.get('side_book_align_long_axis',False):
        alignment_waypoints=None
        # Explicit Oracle geometry feedback: align the rectangular long axis
        # with the drawer width before dropping past the cabinet roof edge.
        object_rotation=r.sim.data.xmat[r.sim.model.body(target['body']).id].reshape(3,3)
        yaw=np.arctan2(object_rotation[1,0],object_rotation[0,0]);angle=np.pi/2-yaw
        if r.config.get('side_book_axis_alignment_mod_pi',False):angle=rectangular_axis_alignment(yaw)
        if r.config.get('side_book_axis_path_precheck',False):
            from robot_exam.control.book_path import held_book_planning_arm,plan_book_lowering,check_release_forearm_clearance
            from robot_exam.control.joint_rrt import plan_joint_rrt
            candidates=[];selected_rotation=None
            drop_x=float(r.config.get('side_book_drop_center_x',.47))
            if r.config.get('side_book_drop_center_x_by_rank') is not None:
                drop_x=float(r.config['side_book_drop_center_x_by_rank'][rank])
            tip,current_r=r.arm.pose()
            relative=current_r.T@(r.sim.data.xpos[r.sim.model.body(target['body']).id]-tip)
            tilts=(r.config.get('side_book_insertion_tilt_degrees',[0.])
                   if rank in r.config.get('side_book_insertion_tilt_ranks',[2]) else [0.])
            if len(tilts)>4 or any(not 0<=float(t)<=35 for t in tilts):
                raise ValueError('At most four insertion tilts in0–35degrees')
            for trial_angle,tilt_deg in [(a,t) for a in (angle,angle+np.pi,angle-np.pi) for t in tilts]:
                c,s=np.cos(trial_angle),np.sin(trial_angle)
                candidate_r=np.array([[c,-s,0],[s,c,0],[0,0,1]])@place_rotation
                tilt=np.deg2rad(float(tilt_deg));ct,st=np.cos(tilt),np.sin(tilt)
                candidate_r=np.array([[ct,0,st],[0,1,0],[-st,0,ct]])@candidate_r
                row={'angle_rad':float(trial_angle),'tilt_deg':float(tilt_deg),'feasible':False,'errors':[]}
                for seed in (r.qgoal,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
                    try:
                        q,_=r.arm.solve([.36,.50,pre_height],candidate_r,seed=seed,max_iterations=500)
                        alignment_path,alignment_report=plan_joint_rrt(r.arm,q,allowed_bodies=[target['body']],wall_seconds=3.,
                            held_body=target['body'] if r.config.get('side_book_full_assembly_joint_path',False) else None)
                        planning=held_book_planning_arm(r.arm,q,candidate_r,target['body'])
                        destination=np.array([drop_x,.50,height])-candidate_r@relative
                        lowering_path,lowering_report=plan_book_lowering(planning,destination,candidate_r,target['body'])
                        release_planning=held_book_planning_arm(planning,lowering_path[-1],candidate_r,target['body'])
                        release_report=check_release_forearm_clearance(release_planning)
                        row.update(feasible=True,alignment=alignment_report,lowering=lowering_report,release=release_report)
                        selected_rotation=candidate_r;alignment_waypoints=alignment_path;break
                    except RuntimeError as exc:row['errors'].append(str(exc))
                candidates.append(row)
                if selected_rotation is not None:break
            if r.output:
                (r.output/('book_axis_path_precheck_'+target['body']+'.json')).write_text(__import__('json').dumps(candidates,indent=2))
            if selected_rotation is None:
                raise TaskFailure('book_placement_path','No equivalent book-axis orientation admits a collision- and orientation-checked lowering')
            place_rotation=selected_rotation
        else:
            c,s=np.cos(angle),np.sin(angle)
            place_rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])@place_rotation
        if alignment_waypoints is None:
            checked_move(np.array([.36,.50,pre_height]))
        else:
            # Keep the exact preflighted IK branch. Re-solving from the old
            # seed can silently select the wrist-folded branch just rejected
            # by the complete lowering-path check.
            r.joint_path(alignment_waypoints,seconds=2.,
                _collision_checked=bool(r.config.get('side_book_execute_checked_path',False)))
            if np.linalg.norm(r.arm.pose()[0]-[.36,.50,pre_height])>.018:
                raise TaskFailure('arm_tracking','Validated book-alignment branch did not physically arrive')
        record_geometry('after_yaw_alignment')
    offset=r.sim.data.xpos[r.sim.model.body(target['body']).id]-r.arm.pose()[0]
    if not r.bilateral(target['body']) or np.linalg.norm(offset)>.20:
        raise TaskFailure('gripper_slip','Invalid held-book offset after orientation; not an IK reachability failure')
    drop_x=float(r.config.get('side_book_drop_center_x',.47))
    if r.config.get('side_book_drop_center_x_by_rank') is not None:
        drop_x=float(r.config['side_book_drop_center_x_by_rank'][rank])
    if r.config.get('side_book_near_front_placement',False):
        from robot_exam.tasks.scoring import body_bounds
        lo,_=body_bounds(r.sim,target['body'])
        half_x=float(r.sim.data.xpos[r.sim.model.body(target['body']).id,0]-lo[0])
        front=r.sim.model.geom(f'drawer_{rank}_front').id
        inner_front=float(r.sim.data.geom_xpos[front,0]+r.sim.model.geom_size[front,0])
        drop_x=inner_front+half_x+float(r.config.get('side_book_front_margin_m',.015))
    destination=np.array([drop_x,.50,height])-offset
    record_geometry('before_lowering')
    checked_move(destination)
    record_geometry('before_release')
    trace=[]
    class ReleaseObserver:
        def before_step(self):pass
        def after_step(self):
            model,data=r.sim.model,r.sim.data
            body=model.body(target['body']).id
            dof=int(model.jnt_dofadr[model.body_jntadr[body]])
            contacts=[]
            for contact_id,contact in enumerate(data.contact):
                names=[model.body(int(model.geom_bodyid[g])).name or '' for g in contact.geom]
                if target['body'] not in names:continue
                force=np.empty(6);__import__('mujoco').mj_contactForce(model,data,contact_id,force)
                contacts.append({'bodies':names,'distance_m':float(contact.dist),'normal_force_n':float(force[0])})
            trace.append({'time':float(data.time),'position':data.xpos[body].tolist(),
                          'velocity':data.qvel[dof:dof+6].tolist(),'contacts':contacts})
    observer=ReleaseObserver()
    if r.config.get('book_release_contact_trace',False):r.sim.add_step_controller(observer)
    try:
        r.gripper.command(0);r.hold(.8)
    finally:
        r.sim.remove_step_controller(observer)
        if trace and r.output:
            (r.output/('book_release_trace_'+target['body']+'.json')).write_text(__import__('json').dumps(trace,indent=2))
    record_geometry('after_release')
    if r.config.get('side_book_collision_planned_withdrawal',False):
        # Withdrawal changes the arm configuration even after the book has
        # been released. Validate this path as well as the insertion path;
        # a Cartesian endpoint alone does not exclude wrist/forearm contact.
        retreat=(r.arm.pose()[0]+[0,0,.06] if r.config.get('side_book_vertical_withdrawal',False)
                 else destination+[-.14,0,.025])
        checked_move(retreat,allow_held_contact=False)
    else:
        r.move(destination+[-.14,0,.025],place_rotation,seconds=2)
    r.stow()
    if r.config.get('side_book_settle_released_book',False):
        _settle_released_book(r,target,metadata['drawer_bottoms'][rank])
    closed=drawer_operation(r,rank,0.,metadata)
    operations.append({'rank_from_visual_area':rank,'book':target['body'],'area':book['area'],
                       'opened':opened,'closed':closed,'lift_m':grasp['lift_m'],'strategy':grasp['scope']})
    r.backout(-.50);r.stow()


def _settle_released_book(r,target,bottom):
    """Oracle contact push to flatten a tall book, never a pose assignment."""
    from robot_exam.tasks.scoring import body_bounds
    name=target['body'];model,data=r.sim.model,r.sim.data;body=model.body(name).id
    lower,upper=body_bounds(r.sim,name)
    if upper[2]<bottom+.14:return
    r.transition(State.MANIPULATE,'settle_released_book_flat',target=name,timeout=60)
    center=data.xpos[body].copy();height=float(upper[2]-.03)
    if lower[2]<bottom-.003 or height>1.05:
        raise TaskFailure('book_settle_geometry','Released book is not safely supported for toppling')
    # An upright book needs room to fall forward into the drawer. Reject the
    # action if its final extent would cross the physical back panel.
    width=float(target['size'][1]);drawer_x=float(data.xpos[model.body(f"drawer_{target['rank']}").id,0])
    if center[0]+width>drawer_x+.265:
        raise TaskFailure('book_settle_geometry','Insufficient forward room to flatten the released book')
    rotation=np.diag([1.,-1.,-1.]);pre=np.array([center[0]-.055,center[1],upper[2]+.035])
    if r.config.get('side_book_settle_base_precheck',False):
        # A released upright book can stop farther into the drawer than the
        # nominal release centre. Plan the complete pressing reach before
        # moving the pusher; do not relax IK or relocate the book.
        targets=[('settle_above',pre,rotation,[name]),
                 ('settle_contact',np.array([pre[0],pre[1],height]),rotation,[name]),
                 ('settle_forward',np.array([pre[0]+.18,pre[1],height]),rotation,[name])]
        previous_grid=getattr(r,'grid',None);r.grid=None
        try:
            report=evaluate_operational_base_poses(r,r.base.pose().copy(),targets,
                offsets=[(dx,dy,0.) for dx in (0.,.04,.08,.12) for dy in (0.,-.04,.04)])
        finally:r.grid=previous_grid
        if r.output:
            (r.output/('book_settle_base_'+name+'.json')).write_text(__import__('json').dumps(report,indent=2))
        if report['selected'] is None:
            raise TaskFailure('book_settle_reach','No collision-safe base pose can reach the complete upright-book press path')
        goal=np.asarray(report['selected']['pose'])
        if np.linalg.norm(goal-r.base.pose())>.003:
            r.stow();r.transition(State.NAVIGATE,'upright_book_press_reposition',target=goal.tolist(),timeout=45)
            r.dock(goal,timeout=40)
            r.transition(State.MANIPULATE,'settle_released_book_flat_after_reposition',target=name,timeout=60)
    r.gripper.command(1);r.hold(.5)
    _checked_side_book_move(r,pre,rotation,name)
    r.move([pre[0],pre[1],height],rotation,seconds=2.)
    trace=[]
    for step in range(9):
        goal=np.array([pre[0]+.02*(step+1),pre[1],height])
        r.move(goal,rotation,seconds=.5)
        lower,upper=body_bounds(r.sim,name)
        normal=data.xmat[body].reshape(3,3)[:,2]
        trace.append({'time':float(data.time),'book_position':data.xpos[body].tolist(),
                      'lower':lower.tolist(),'upper':upper.tolist(),'normal_z':float(normal[2])})
        if r.output:(r.output/('book_settle_'+name+'.json')).write_text(__import__('json').dumps(trace,indent=2))
        if abs(normal[2])>.92 and upper[2]<bottom+.14:break
    else:
        raise TaskFailure('book_settle_contact','Bounded physical toppling did not flatten the released book')
    # The book is already below the tool after it falls flat. A fixed upward
    # Cartesian retreat can exceed the original UR5 workspace here. Use the
    # existing collision-planned stow instead of that unreachable waypoint.
    r.stow();r.hold(.5)
    if r.output:(r.output/('book_settle_'+name+'.json')).write_text(__import__('json').dumps(trace,indent=2))


def rectangular_axis_alignment(yaw,target=np.pi/2):
    """Shortest physical alignment of an unoriented rectangular long axis."""
    return float((target-yaw+np.pi/2)%np.pi-np.pi/2)
