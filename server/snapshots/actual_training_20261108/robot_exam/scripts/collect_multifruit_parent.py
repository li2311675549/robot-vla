#!/usr/bin/env python3
"""Bounded Oracle pilot in the real ten-fruit scene, with two parked clips.

This is a short-skill collection fixture, not a five-metre complete-task run.
Only the initial base pose is set before physics starts. Subsequent manipulation
and transport use the existing controllers. Failed attempts are kept and are
never admitted as positive parent demonstrations.
"""
import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from build_tasks import build
from robot_exam.control.base_pose import evaluate_operational_base_poses
from robot_exam.data.episodes import EpisodeRecorder, validate_episode
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.simulation import Simulation
from robot_exam.tasks.manipulation import pick, release
from robot_exam.tasks.runtime import State, TaskFailure, TaskRuntime
from robot_exam.tasks.scoring import fruit_score


def visible_target(sim, metadata, category, output, selector='leftmost',require_hardware=False):
    """Oracle mask for expert selection/caption QA, never a policy input."""
    camera = RGBDCamera(sim)
    try:
        rgb, depth = camera.capture()
        from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER
        graphics={'vendor':glGetString(GL_VENDOR).decode(),'renderer':glGetString(GL_RENDERER).decode()}
        if require_hardware and any(x in graphics['renderer'].lower() for x in ('llvmpipe','softpipe','swiftshader','software rasterizer')):
            raise RuntimeError(f'Hardware-rendered dataset requested, got{graphics}')
        camera.save(output / 'initial_observation', rgb, depth)
        renderer = camera.renderer
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera=camera.id)
        segmentation = renderer.render().copy()
        rows = []
        for obj in metadata['objects']:
            if obj['category'] != category:
                continue
            body = sim.model.body(obj['body']).id
            geoms = np.flatnonzero(sim.model.geom_bodyid == body)
            mask = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[..., 0], geoms)
            yy, xx = np.nonzero(mask)
            rows.append({**obj, 'visible_pixels': len(xx),
                         'image_u': float(np.mean(xx)) if len(xx) else None})
        visible = sorted((x for x in rows if x['visible_pixels'] >= 30), key=lambda x: x['image_u'])
        if not visible:
            raise TaskFailure('target_visibility', 'No sufficiently visible target of the requested class')
        # Do not silently skip the true leftmost object if it is too occluded.
        all_visible = sorted((x for x in rows if x['visible_pixels']), key=lambda x: x['image_u'])
        index=0 if selector=='leftmost' else -1
        if visible[index]['body'] != all_visible[index]['body']:
            raise TaskFailure('target_ambiguity', f'{selector} target is too occluded for an unambiguous caption')
        (output / 'oracle_caption_qa.json').write_text(json.dumps(rows, indent=2))
        return {**visible[index],'image_relation':selector,'opengl':graphics}
    finally:
        camera.close()


def collect(output, seed, category, config, grasp_only=False, precision_dock_strategy=None,
            guarded_dock=False, twist_braking_all=False, target_selector='leftmost',grasp_z_offset=None,
            grasp_candidate_index=0,oracle_clutter_clearance=False,require_hardware=False,fruit_count=10,camera_near_m=.005):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Preserve previous attempt: {output}')
    if shutil.disk_usage(output.parent).free < 25.5 * 1024**3:
        raise RuntimeError('Less than25.5GiB free before pilot')
    output.mkdir(parents=True)
    settings = json.loads((ROOT / 'configs/task_defaults.json').read_text())
    settings.update(json.loads(Path(config).read_text()))
    settings.update(stationary_manipulation=True, task_wrist_camera=True,
                    scene_layout_profile='bounded_scatter_v3', bin_label_profile='upright_v2')
    settings['multifruit_precision_dock_strategy'] = precision_dock_strategy
    settings['multifruit_guarded_dock'] = guarded_dock
    settings['multifruit_target_selector'] = target_selector
    settings['curriculum_fruit_count']=fruit_count
    settings['multifruit_grasp_candidate_index']=grasp_candidate_index
    settings.update(collection_expert_source='oracle',oracle_grasp_clutter_clearance=oracle_clutter_clearance)
    if grasp_z_offset is not None:
        if not -.012<=grasp_z_offset<=.012:raise ValueError('Grasp offset restricted to+-12mm')
        settings[category+'_grasp_z_offset_m']=grasp_z_offset
    if twist_braking_all:
        settings.update(payload_precision_twist_slew=True,payload_twist_slew_categories=['apple','banana','orange'])
    (output / 'config.json').write_text(json.dumps(settings, indent=2))
    path, metadata = build('fruit', seed, output / 'scene.xml', wrist_camera=True,
                           layout_profile='bounded_scatter_v3', bin_label_profile='upright_v2',fruit_count=fruit_count)
    from robot_exam.data.rest_initialization import configure_rest_initialization
    metadata=configure_rest_initialization(path,metadata,settings)
    sim = Simulation(path)
    # Independent skill-fixture initialization only; never part of task score.
    joint = sim.model.joint('husky_free').id
    adr = sim.model.jnt_qposadr[joint]
    sim.data.qpos[adr:adr+2] = metadata['dock'][:2]
    mujoco.mj_forward(sim.model, sim.data)
    r = TaskRuntime(sim, output=output, sim_limit=600, wall_limit=900)
    r.config = settings
    result = {'status': 'FAIL', 'qualified_parent': False, 'seed': seed, 'category': category,
              'scope': f'{fruit_count}-fruit parked short-skill fixture; NEVER a5m/full10fruit evaluation',
              'expert_source': 'oracle', 'training_split': 'train_pilot',
              'initialization': f'base dock XY set before physics; all{fruit_count}declared curriculum fruits retained, none removed during manipulation',
              'policy_input': 'front+wrist RGB, measured7D state, language only',
              'navigation_supervision': 'excluded', 'grasp_only_diagnostic': grasp_only}
    source_files=sorted((ROOT/'src/robot_exam').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/build_tasks.py']
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    result['source_sha256_at_start']=source_hashes
    logger = None
    clips = []
    started = time.monotonic()

    def begin_clip(phase, instruction):
        nonlocal logger
        r._ensure_operation_hold()
        logger = EpisodeRecorder(sim, output / phase, instruction, seed, 'oracle',
                                 initial_arm_reference=r.arm.reference,
                                 shared_model_cache=ROOT/'outputs/stage20/multifruit_shared_models_v1',camera_near_m=camera_near_m)
        logger.meta.update(parent_id=f'multifruit_{seed}_{category}', attempt_id=output.name,
                           scene_context_id=f'multifruit_{seed}',
                           phase=phase, target_body=target['body'], category=category,
                           scene_object_count=len(metadata['objects']), base_control='encoder_parking_v1',
                           parent_admission=False, initial_base_pose=r.base.pose().tolist(),
                           parking_anchor=r.parking.anchor.tolist(),
                           parking_wheel_anchor=r.parking.wheel_anchor.tolist(),
                           parking_stable_seconds=r.parking.stable_seconds,
                           parking_ready_at_start=bool(r.parking.ready),
                           scope=result['scope'], instruction_style='multifruit_image_relative_v1')
        logger.meta['fruit_spawn_profile']=metadata.get('fruit_spawn_profile','legacy_v1')
        if metadata.get('rest_initialization'):
            logger.meta['rest_initialization_calibration_sha256']=metadata['rest_initialization']['calibration_sha256']
        logger.meta['grasp_scene_information_source']='oracle_collision_teacher' if oracle_clutter_clearance else 'declared_oracle_geometry'
        r.data_logger = logger

    def finish_clip(success):
        nonlocal logger
        r.data_logger = None
        record = logger.close(success)
        clips.append(record)
        logger = None

    try:
        r.hold(.7)
        r._ensure_operation_hold()
        target = visible_target(sim, metadata, category, output, target_selector,require_hardware)
        body = target['body']
        result['target'] = target
        initial_z = float(sim.data.xpos[sim.model.body(body).id, 2])
        begin_clip('grasp', f'Pick up the {target_selector} {category} in the front camera image and hold it above the table.')
        scale = target['scale']
        rotation = sim.data.xmat[sim.model.body(body).id].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        detection = {'category': category, 'center': sim.data.xpos[sim.model.body(body).id].tolist(),
                     'yaw': yaw, 'length': (.098 if category == 'banana' else .055)*scale,
                     'width': (.034 if category == 'banana' else .055)*scale,
                     'pca_ratio': 3 if category == 'banana' else 1}
        grasp = pick(r, body, detection,candidate_index=grasp_candidate_index)
        from robot_exam.control.contact_grip import enabled_for
        if category in ('banana', 'apple') and not enabled_for(r,category):
            r.gripper.command(float(settings.get(category+'_hold_closure', .70)))
        if not r.stable_grasp(body, seconds=1.5):
            raise TaskFailure('gripper_slip', 'Post-lift1.5s contact/pose witness failed')
        result['grasp'] = {'lift_m': float(sim.data.xpos[sim.model.body(body).id, 2]-initial_z),
                           'bilateral': bool(r.bilateral(body)), 'candidate_count': grasp['candidate_count'],
                           'position':grasp['position'].tolist(),'rotation':grasp['rotation'].tolist(),
                           'candidate_index':grasp_candidate_index}
        finish_clip(True)
        if grasp_only:
            result['status'] = 'GRASP_ONLY_PASS_NOT_QUALIFIED_PARENT'
        else:
            # No arm-only VLA frames while the physical mobile base transports.
            r.transition(State.NAVIGATE, 'unrecorded_traditional_payload_transport', timeout=300)
            r._release_operation_hold()
            r.payload_motion_profile = category
            r.begin_payload(body, grasp['rotation'], world_stabilize=True)
            grid = OccupancyGrid()
            for box in metadata['obstacles']:
                grid.rectangle(*box)
            r.grid = grid
            r.backout(-.85)
            bincenter = np.asarray(metadata['bins'][category]['center'])
            drop = bincenter.copy(); drop[2] = .744
            targets = [('approach', drop+[0, 0, .045], grasp['rotation'], [body]),
                       ('release', drop, grasp['rotation'], [body]),
                       ('retreat', drop+[0, 0, .06], grasp['rotation'], [body])]
            report = evaluate_operational_base_poses(r, [-.25, float(bincenter[1]), 0], targets,
                        offsets=[(dx, dy, da) for dx in (.12, .16, .20)
                                 for dy in (0, -.06, .06) for da in (-.3, 0, .3)])
            (output / 'base_pose_trials.json').write_text(json.dumps(report, indent=2))
            if report['selected'] is None:
                raise TaskFailure('placement_unreachable', 'No feasible full release/exit base pose')
            goal = np.asarray(report['selected']['pose'])
            staging = goal.copy(); staging[:2] -= .4*np.array([np.cos(goal[2]), np.sin(goal[2])])
            r.navigate(staging, grid, timeout=180)
            r.transition(State.NAVIGATE, 'precision_bin_dock', timeout=50)
            r.payload_docking = True
            old_local = {k:r.config.get(k) for k in ('local_dock_strategy', 'local_dock_tracking_version')}
            if precision_dock_strategy:
                r.config.update(local_dock_strategy=precision_dock_strategy,
                                local_dock_tracking_version='axis_bounded_v5')
            try:
                if guarded_dock:
                    from robot_exam.control.payload_dock import dock_held_object
                    result['dock_attempts']=dock_held_object(r,body,goal,grasp['rotation'],initial_z+.055,
                                                precision_dock_strategy,maximum_recoveries=1)
                else:r.dock(goal, timeout=45, tolerance=.03)
            finally:
                r.payload_docking = False
                for k, value in old_local.items():
                    if value is None:
                        r.config.pop(k, None)
                    else:
                        r.config[k] = value
            if not r.stable_grasp(body):
                raise TaskFailure('gripper_slip', 'Payload was lost before place skill')
            r.end_payload()
            r.transition(State.APPROACH, 'parked_place_skill', timeout=30)
            begin_clip('place', f'Place the held {category} into the bin labelled {category.upper()} and withdraw the gripper.')
            r.move(drop+[0, 0, .045], grasp['rotation'], seconds=2.4)
            release(r, drop, grasp['rotation'], retreat_offsets=[(0, 0, .06), (-.03, 0, .06), (.03, 0, .06)])
            r.hold(.5)
            result['placement'] = next(x for x in fruit_score(sim, metadata)['objects'] if x['body'] == body)
            placed = bool(result['placement']['correct_bin'])
            result['target_physical_success']=placed
            finish_clip(placed)
            if not placed:
                raise TaskFailure('wrong_bin_or_unstable', 'Original full-task item containment score failed')
            result['status'] = 'PHYSICAL_PARENT_PASS_PENDING_REPLAY_AND_EXPORT'
            from robot_exam.data.scene_quality import fruit_scene_preservation
            result['scene_preservation']=fruit_scene_preservation(sim,metadata,body)
            result['scene_spawn_profile']=metadata.get('fruit_spawn_profile','legacy_v1')
            if settings.get('require_non_target_scene_preservation',False) and not result['scene_preservation']['success']:
                result['status']='PHYSICAL_TARGET_PASS_SCENE_PRESERVATION_FAIL'
    except Exception as exc:
        result.update(error=str(exc), failure_category=getattr(exc, 'category', 'simulation_or_collection'),
                      failure_stage=r.stage, traceback=traceback.format_exc())
        if logger is not None:
            finish_clip(False)
    finally:
        r.data_logger = None
        result.update(sim_seconds=float(sim.data.time), wall_seconds=time.monotonic()-started,
                      clips=clips, collision_events=r.collision_events,
                      data_warning_counts=sim.data.warning.number.tolist())
        result['source_changed_during_attempt'] = any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha
                                                     for name,sha in source_hashes.items())
        r.end_payload()
        r.save()
        (output / 'result.json').write_text(json.dumps(result, indent=2))
    # Generic full-frequency replay applies stored arm/gripper goals through
    # their normal controller. It does not certify a low-frequency VLA policy.
    result['full_rate_validations'] = [validate_episode(x['path'], replay=True) for x in clips]
    result['admission_pending'] = ['phase-aware low-frequency physical replay', 'LeRobot/OpenPI conversion',
                                    'target-language ambiguity and actual bin visibility review']
    result['total_output_bytes'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=2100001)
    p.add_argument('--category', choices=['apple', 'banana', 'orange'], default='orange')
    p.add_argument('--config', type=Path, default=ROOT/'configs/stage20_fruit_round_route_dev_v7.json')
    p.add_argument('--grasp-only', action='store_true', help='Diagnostic only; cannot qualify a parent')
    p.add_argument('--precision-dock-strategy', choices=['preapproach', 'lattice', 'direct'])
    p.add_argument('--guarded-dock', action='store_true')
    p.add_argument('--twist-braking-all', action='store_true')
    p.add_argument('--target-selector', choices=['leftmost','rightmost'], default='leftmost')
    p.add_argument('--grasp-z-offset',type=float)
    p.add_argument('--grasp-candidate-index',type=int,choices=range(12),default=0)
    p.add_argument('--oracle-clutter-clearance',action='store_true')
    p.add_argument('--require-hardware-renderer',action='store_true')
    p.add_argument('--fruit-count',type=int,choices=[3,5,10],default=10)
    p.add_argument('--camera-near-m',type=float,default=.005,help='Versioned virtual rendering near plane; physics unchanged')
    args = p.parse_args()
    collect(args.output, args.seed, args.category, args.config, args.grasp_only, args.precision_dock_strategy,
            args.guarded_dock,args.twist_braking_all,args.target_selector,args.grasp_z_offset,args.grasp_candidate_index,
            args.oracle_clutter_clearance,args.require_hardware_renderer,args.fruit_count,args.camera_near_m)
