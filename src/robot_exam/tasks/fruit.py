import numpy as np
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.perception.objects import observe_objects,observe_bins
from robot_exam.tasks.runtime import State,TaskFailure
from robot_exam.tasks.manipulation import pick,release
from robot_exam.tasks.scoring import fruit_score
from robot_exam.control.base_pose import evaluate_operational_base_poses


def execute(runtime,metadata,mode='vision'):
    if len(metadata.get('objects',[]))!=10:
        raise TaskFailure('scene_contract','Full fruit classification requires exactly10objects;3/5fruit curricula are short skills only')
    r=runtime;grid=OccupancyGrid();r.grid=grid
    for box in metadata['obstacles']:grid.rectangle(*box)
    r.hold(.7);r.transition(State.NAVIGATE,'five_metre_table_approach')
    r.navigate(metadata['staging'],grid);r.dock(metadata['dock'])
    camera=RGBDCamera(r.sim)
    attempts=[];counts={'apple':0,'banana':0,'orange':0};processed=set()
    try:
        bins={}
        if mode=='vision':
            # Known station approach waypoints are navigation map information;
            # class assignment and metric bin centres come from visible labels.
            if r.config.get('fruit_central_bin_survey',False):
                # A single physical camera station sees all three labels in
                # two development scenes. Fall back to the three stations if
                # any label is cropped or occluded in a randomized scene.
                r.transition(State.NAVIGATE,'survey_bin_labels_central')
                r.stow();r.backout(-.40)
                middle=float(metadata['bin_stations'][1][1])
                r.navigate([-.40,middle,.25],grid);r.dock([0.,middle,.25])
                rgb,depth=camera.capture();seen=observe_bins(camera,rgb,depth)
                bins={category:entry for category,entry in seen.items() if entry['pixels']>=200}
            if len(bins)!=3:
                bins={}
                for station in metadata['bin_stations']:
                    r.transition(State.NAVIGATE,'survey_bin_label_station')
                    r.stow();r.backout(-.40);r.navigate([-.40,station[1],0],grid);r.dock(station)
                    rgb,depth=camera.capture();seen=observe_bins(camera,rgb,depth)
                    for category,entry in seen.items():
                        if abs(entry['center'][1]-station[1])<.20:bins[category]=entry
            if len(bins)!=3:raise TaskFailure('perception',f'Label survey found {len(bins)} classes')
            r.stow();r.backout(-.40);r.navigate(metadata['staging'],grid)
            try:r.dock(metadata['dock'])
            except TaskFailure as dock_error:
                if dock_error.category!='navigation_timeout':raise
                r.base.command(0,0);r.hold(.6)
                actual=r.base.pose();delta=actual[:2]-np.asarray(metadata['dock'][:2])
                velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+3]
                evidence={'dock_timeout':str(dock_error),'actual_pose':actual.tolist(),
                          'position_error_m':float(np.linalg.norm(delta)),
                          'heading_error_rad':float(abs((actual[2]-metadata['dock'][2]+np.pi)%(2*np.pi)-np.pi)),
                          'linear_speed_m_s':float(np.linalg.norm(velocity[:2])),
                          'yaw_rate_rad_s':float(r.sim.data.qvel[r.base_velocity_dof+5])}
                acceptable=(evidence['position_error_m']<=.04 and evidence['heading_error_rad']<=.12
                            and evidence['linear_speed_m_s']<=.03 and abs(evidence['yaw_rate_rad_s'])<=.08)
                evidence['accepted_as_settled_staging_pose']=bool(acceptable)
                if r.output:
                    path=r.output/'staging_dock_recoveries.json'
                    old=__import__('json').loads(path.read_text()) if path.exists() else []
                    old.append(evidence);path.write_text(__import__('json').dumps(old,indent=2))
                if not acceptable:raise
        for operation in range(10):
            r.transition(State.OBSERVE,'rgbd_remaining_fruit',timeout=5)
            rgb,depth=camera.capture()
            if r.output and operation==0:camera.save(r.output/'observation',rgb,depth)
            detections=observe_objects(camera,rgb,depth,split_round_instances=bool(r.config.get("fruit_split_round_instances",False)))
            if mode=='oracle':
                detections=[]
                for obj in metadata['objects']:
                    if obj['body'] in processed:continue
                    p=r.sim.data.xpos[r.sim.model.body(obj['body']).id]
                    detections.append({'category':obj['category'],'center':p.tolist(),'yaw':obj['yaw'],'length':.08 if obj['category']=='banana' else .055,'width':.03 if obj['category']=='banana' else .055,'pca_ratio':3 if obj['category']=='banana' else 1})
                detections.sort(key=lambda x:x['center'][0])
            if not detections:raise TaskFailure('perception','No remaining fruit visible on table')
            detection=detections[0];center=np.asarray(detection['center'])
            # Matching truth is solely for contact logging and independent scoring,
            # never replaces detection.center or determines the selected pose.
            available=[o for o in metadata['objects'] if o['body'] not in processed]
            selected=min(available,key=lambda o:np.linalg.norm(r.sim.data.xpos[r.sim.model.body(o['body']).id]-center))
            body=selected['body'];category=detection['category']
            r.transition(State.SELECT_TARGET,'select_visible_fruit',target=detection)
            max_attempts=r.config.get('maximum_fruit_grasp_attempts',3)
            from robot_exam.control.contact_grip import enabled_for
            contact_feedback=enabled_for(r,category)
            for retry in range(max_attempts):
                try:
                    grasp=pick(r,body,detection,candidate_index=retry)
                    if category in ('banana','apple') and not contact_feedback:
                        # Paired physical checkpoint trials found that full
                        # closure rolled both a curved banana and a round
                        # apple toward the pad edge. Regulate the tendon
                        # target only after a real lift and recheck contact.
                        r.gripper.command(float(r.config.get(f'{category}_hold_closure',.70)))
                        if not r.stable_grasp(body,seconds=.5):
                            raise TaskFailure('gripper_slip',f'{category} lost after post-lift hold regulation')
                    if contact_feedback and not r.stable_grasp(body,seconds=.8):
                        raise TaskFailure('gripper_slip','Measured-contact grip failed the post-lift dwell')
                    attempts.append({'body':body,'success':True,'category':category,'lift_m':grasp['lift_m'],
                                     'candidate_count':grasp['candidate_count'],'retry':retry,'placement_verified':False})
                    break
                except TaskFailure as exc:
                    attempts.append({'body':body,'success':False,'category':exc.category,'reason':str(exc),'retry':retry})
                    if retry==max_attempts-1:raise
                    r.retries+=1;r.transition(State.RECOVER,'reobserve_after_grasp_failure')
                    r.gripper.command(0);r.stow();rgb,depth=camera.capture()
                    options=observe_objects(camera,rgb,depth,split_round_instances=bool(r.config.get("fruit_split_round_instances",False)))
                    if not options:raise
                    detection=min(options,key=lambda o:np.linalg.norm(np.asarray(o['center'])-center))
            # Raise from the verified lift without sweeping laterally across
            # the table edge or chassis.  Only after this arm motion is complete
            # do we enable the base-yaw carry stabilizer.
            carry_start=r.arm.pose()[0].copy()
            carry_raise=None
            for lift_extra in (.04,.02,0.):
                candidate=carry_start+[0,0,lift_extra]
                try:
                    r.solve_ik(candidate,grasp['rotation'],seed=r.qgoal,
                               context=f'fruit_carry_clearance_{lift_extra:.2f}')
                except RuntimeError:
                    continue
                carry_raise=lift_extra
                if lift_extra>0:r.move(candidate,grasp['rotation'],seconds=1.2)
                break
            if carry_raise is None:
                raise TaskFailure('ik_unreachable','No feasible post-grasp carry height')
            if r.output:
                with (r.output/'carry_clearance.jsonl').open('a') as log:
                    log.write(__import__('json').dumps({'time':float(r.sim.data.time),
                        'body':body,'extra_lift_m':carry_raise,
                        'tip_height_m':float(r.arm.pose()[0][2]),
                        'fruit_height_m':float(r.sim.data.xpos[r.sim.model.body(body).id,2]),
                        'bilateral_contact':bool(r.bilateral(body))})+'\n')
            if carry_raise<.04 and not r.stable_grasp(body,seconds=.2):
                raise TaskFailure('gripper_slip','Post-grasp carry height lost stable contact')
            bincenter=np.asarray(bins[category]['center'] if mode=='vision' else metadata['bins'][category]['center'])
            index=counts[category]
            offset=np.array([(-.032 if index%2==0 else .032),(-.032 if index//2==0 else .032),0])
            target=bincenter+offset;target[2]=.744
            transport_timeout=(float(r.config.get('banana_transport_state_timeout_seconds',220))
                               if category=='banana' else 120.)
            r.transition(State.NAVIGATE,'carry_fruit_to_observed_bin',target=category,timeout=transport_timeout)
            transport_attempts=1+int(r.config.get('maximum_payload_transport_recoveries',2))
            selected_base=None
            for transport_try in range(transport_attempts):
                # Preserve the grasp orientation in world coordinates as the
                # skid-steer base changes heading.  Slip recovery is bounded:
                # stop, re-clamp with the physical gripper, re-verify contact,
                # then replan from the actual base pose.
                r.payload_motion_profile=category
                r.begin_payload(body,grasp['rotation'],
                                world_stabilize=bool(r.config.get('fruit_world_stabilize',True)))
                if transport_try==0:
                    r.backout(-.85)
                    desired=np.asarray([-.25,float(bincenter[1]),0.])
                    place_targets=[('approach',target+[0,0,.045],grasp['rotation'],[body]),
                                   ('release',target,grasp['rotation'],[body]),
                                   ('retreat',target+[0,0,.06],grasp['rotation'],[body])]
                    # Search farther-in poses as well as offsets around the
                    # nominal dock. The nominal arm reach was too close to a
                    # singular branch: centimetres of physical dock error
                    # erased the remaining placement IK margin.
                    base_report=evaluate_operational_base_poses(
                        r,desired,place_targets,
                        offsets=[(dx,dy,dyaw) for dx in (.12,.16,.20)
                                 for dy in (0,-.06,.06) for dyaw in (-.3,0,.3)])
                    if r.output:
                        (r.output/'base_pose_trials.json').write_text(__import__('json').dumps(base_report,indent=2))
                    if base_report['selected'] is None:
                        raise TaskFailure('placement_unreachable','No collision-free, A*-reachable base pose can reach the bin release path')
                    selected_base=base_report['selected']['pose']
                try:
                    staging=np.asarray(selected_base);staging=staging.copy();staging[0]-=.40*np.cos(staging[2]);staging[1]-=.40*np.sin(staging[2])
                    navigation_timeout=(float(r.config.get('banana_navigation_timeout_seconds',180))
                                        if category=='banana' else 120.)
                    if r.config.get('fruit_split_transport_deadlines',False):
                        r.transition(State.NAVIGATE,'carry_global_navigation',target=category,timeout=navigation_timeout)
                    r.navigate(staging,grid,timeout=navigation_timeout)
                    r.payload_docking=True
                    try:
                        if r.config.get('fruit_split_transport_deadlines',False):
                            r.transition(State.NAVIGATE,'carry_precision_dock',target=selected_base,timeout=50.)
                        if r.config.get('fruit_guarded_precision_dock',False):
                            from robot_exam.control.payload_dock import dock_held_object
                            dock_held_object(r,body,selected_base,grasp['rotation'],float(selected['initial'][2])+.055,
                                             strategy='direct',timeout=45.,maximum_recoveries=1,
                                             evidence_name=f'guarded_dock_{body}_{transport_try}.json')
                        else:
                            r.dock(selected_base,timeout=45,tolerance=.03)
                    except TaskFailure as dock_error:
                        if dock_error.category!='navigation_timeout':raise
                        if r.config.get('fruit_guarded_precision_dock',False):raise
                        # The chassis may stop within a few centimetres of the
                        # nominal pose while the wheel controller rejects a
                        # final yaw correction. Settle physically, then accept
                        # that pose only if the complete approach/release/
                        # retreat path is IK- and collision-feasible there.
                        r.base.command(0,0);r.hold(.6)
                        nominal_actual=r.base.pose().copy()
                        previous_grid=getattr(r,'grid',None);r.grid=None
                        try:
                            actual_pose_report=evaluate_operational_base_poses(
                                r,nominal_actual,place_targets,offsets=[(0,0,0)])
                        finally:r.grid=previous_grid
                        if r.output:
                            (r.output/'dock_actual_pose_recovery.json').write_text(
                                __import__('json').dumps({
                                    'dock_timeout':str(dock_error),
                                    'settled_actual_pose':nominal_actual.tolist(),
                                    'operational_pose_evaluation':actual_pose_report},indent=2))
                        if actual_pose_report['selected'] is None:
                            raise TaskFailure('placement_unreachable',
                                'Dock timed out and the settled physical base pose cannot reach the full release path') from dock_error
                        selected_base=actual_pose_report['selected']['pose']
                    finally:r.payload_docking=False
                    if r.config.get('fruit_guarded_precision_dock',False):
                        r.transition(State.VERIFY,'verify_lifted_payload_after_strict_dock',target=body,timeout=3.)
                    if not r.stable_grasp(body):raise TaskFailure('gripper_slip','Fruit lost during mobile transport')
                    break
                except TaskFailure as exc:
                    r.end_payload()
                    if exc.category!='gripper_slip' or transport_try==transport_attempts-1:raise
                    r.retries+=1
                    r.transition(State.RECOVER,'stop_reclamp_and_replan_after_payload_slip',target=body,timeout=5)
                    r.base.command(0,0)
                    if r.config.get('fruit_preserve_settled_grip_recovery',False):
                        # The original guard may trigger on a settling rotation
                        # while both jaws still hold the fruit. Stop first and
                        # revalidate a lifted, stationary, bounded physical grip
                        # before reacquiring its measured frame. Do not blindly
                        # open and drop it. All original guard events remain in
                        # the payload trace and count toward bounded retries.
                        from robot_exam.control.grasp_recovery import stop_and_revalidate_grasp
                        witness=stop_and_revalidate_grasp(r,body,float(selected['initial'][2])+.055)
                        if not hasattr(r,'transport_recovery_witnesses'):r.transport_recovery_witnesses=[]
                        r.transport_recovery_witnesses.append(witness)
                        if r.output:
                            (r.output/'transport_recovery_witnesses.json').write_text(__import__('json').dumps(r.transport_recovery_witnesses,indent=2))
                        if witness['success']:
                            attempts[-1]['transport_recoveries']=attempts[-1].get('transport_recoveries',0)+1
                            attempts[-1].setdefault('transport_recovery_modes',[]).append('stop_revalidate_without_release')
                            r.transition(State.NAVIGATE,'revalidated_lifted_grip_resume',target=category,timeout=transport_timeout)
                            continue
                    # A short open-close cycle can recapture an object that has
                    # shifted but is still inside the physical finger aperture.
                    r.gripper.command(0);r.hold(.12);r.gripper.command(1);r.hold(.18)
                    if not r.stable_grasp(body):
                        raise TaskFailure('gripper_slip','Bounded re-clamp could not restore a stable physical grasp') from exc
                    attempts[-1]['transport_recoveries']=attempts[-1].get('transport_recoveries',0)+1
                    r.transition(State.NAVIGATE,'replanned_fruit_transport',target=category,timeout=120)
            r.end_payload()
            if r.config.get('fruit_split_transport_deadlines',False):
                # Dock's finite deadline ends when transport ends. It must
                # not expire during the separate arm approach/release skill.
                r.transition(State.APPROACH,'approach_observed_bin_after_transport',
                             target=target.tolist(),timeout=30.)
            r.move(target+[0,0,.045],grasp['rotation'],seconds=2.4)
            release(r,target,grasp['rotation'],
                    retreat_offsets=[(0,0,.06),(-.03,0,.06),(.03,0,.06)])
            r.transition(State.VERIFY,'verify_physical_bin_placement',target=body)
            r.hold(.5)
            score=fruit_score(r.sim,metadata);item=next(x for x in score['objects'] if x['body']==body)
            attempts[-1]['placement_verified']=item['correct_bin']
            if not item['correct_bin']:raise TaskFailure('wrong_bin_or_unstable',f'{body} not stably contained in correct bin')
            processed.add(body);counts[category]+=1
            if len(processed)<10:
                r.transition(State.NAVIGATE,'return_to_table_after_placement',target=body,timeout=120)
                r.stow();r.backout(-.40);r.navigate(metadata['staging'],grid)
                try:r.dock(metadata['dock'])
                except TaskFailure as dock_error:
                    if dock_error.category!='navigation_timeout':raise
                    r.base.command(0,0);r.hold(.6)
                    actual=r.base.pose();delta=actual[:2]-np.asarray(metadata['dock'][:2])
                    velocity=r.sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+3]
                    evidence={'dock_timeout':str(dock_error),'actual_pose':actual.tolist(),
                              'position_error_m':float(np.linalg.norm(delta)),
                              'heading_error_rad':float(abs((actual[2]-metadata['dock'][2]+np.pi)%(2*np.pi)-np.pi)),
                              'linear_speed_m_s':float(np.linalg.norm(velocity[:2])),
                              'yaw_rate_rad_s':float(r.sim.data.qvel[r.base_velocity_dof+5])}
                    acceptable=(evidence['position_error_m']<=.04 and evidence['heading_error_rad']<=.12
                                and evidence['linear_speed_m_s']<=.03 and abs(evidence['yaw_rate_rad_s'])<=.08)
                    evidence['accepted_as_settled_staging_pose']=bool(acceptable)
                    if r.output:
                        path=r.output/'staging_dock_recoveries.json'
                        old=__import__('json').loads(path.read_text()) if path.exists() else []
                        old.append(evidence);path.write_text(__import__('json').dumps(old,indent=2))
                    if not acceptable:raise
        r.hold(1)
        score=fruit_score(r.sim,metadata)
        if not score['success']:raise TaskFailure('leftover','Final strict 10/10 fruit score failed')
        return {'success':True,'score':score,'grasp_attempts':attempts,'mode':mode,'completed':len(processed)}
    except Exception as exc:
        exc.partial={'grasp_attempts':attempts,'completed':len(processed),'score':fruit_score(r.sim,metadata)}
        raise
    finally:camera.close()
