"""Versioned MJCF initial resting poses, established by passive real physics.

This edits a newly generated scene resource before Simulation is created. It
does not expose a running-pose setter, change physical parameters, or constrain
objects. Original files/failed protocols must remain separate.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def configure_rest_initialization(scene,metadata,settings):
    """Opt-in scene version; no change to any existing/default protocol."""
    profile=settings.get('fruit_spawn_profile','legacy_v1')
    if profile=='legacy_v1':return metadata
    if metadata.get('task')!='fruit' or profile!='passive_rest_init_v2':
        raise ValueError('Unsupported fruit spawn profile or non-fruit task')
    value=settings.get('fruit_rest_calibration')
    if isinstance(value,dict):value=value.get(str(metadata['seed']))
    if not value:raise ValueError('Exact-fruit passive rest calibration required before initialization')
    path=Path(value)
    if not path.is_absolute():
        from robot_exam.simulation import ROOT
        path=ROOT/path
    return apply_rest_initialization(scene,metadata,path)


def apply_rest_initialization(scene,metadata,calibration_path):
    scene=Path(scene);calibration_path=Path(calibration_path)
    calibration=json.loads(calibration_path.read_text())
    source_path=Path(calibration['source_scene'])
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=calibration['source_sha256']:
        raise ValueError('Calibration source scene changed')
    if calibration.get('status')!='PASS_ALL_EXACT_FRUITS_STABLE':
        raise ValueError('Passive calibration not stable')
    if metadata.get('fruit_spawn_profile')=='passive_rest_init_v2':
        raise ValueError('Do not apply initialisation twice')
    before=mujoco.MjModel.from_xml_path(str(scene));reference=mujoco.MjModel.from_xml_path(str(source_path))
    table=before.geom('fruit_table_top').id;source_table=reference.geom('fruit_table_top').id
    for attr in ('geom_size','geom_friction','geom_solref','geom_solimp','geom_condim'):
        np.testing.assert_array_equal(getattr(before,attr)[table],getattr(reference,attr)[source_table])
    rows={r['body']:r for r in calibration['rows']};root=ET.parse(scene).getroot()
    original=scene.with_name(scene.stem+'.before_rest.xml')
    if original.exists():raise FileExistsError('Preserve an existing scene-initialization baseline')
    original.write_bytes(scene.read_bytes())
    changes=[]
    for obj in metadata['objects']:
        row=rows[obj['body']]
        if not row['stable'] or obj['category']!=row['category'] or abs(obj['scale']-row['scale'])>1e-12:
            raise ValueError('Rest calibration does not belong to this exact fruit')
        a=before.body(obj['body']).id;b=reference.body(obj['body']).id
        for attr in ('body_mass','body_inertia'):
            np.testing.assert_allclose(getattr(before,attr)[a],getattr(reference,attr)[b],rtol=1e-12,atol=1e-14)
        ga=np.flatnonzero(before.geom_bodyid==a);gb=np.flatnonzero(reference.geom_bodyid==b)
        for attr in ('geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','geom_condim'):
            np.testing.assert_allclose(getattr(before,attr)[ga],getattr(reference,attr)[gb],rtol=1e-12,atol=1e-14)
        yaw=float(obj['yaw']);quaternion=np.empty(4)
        mujoco.mju_mulQuat(quaternion,np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)]),np.asarray(row['rest_quaternion']))
        body=root.find(f".//body[@name='{obj['body']}']")
        position=np.fromstring(body.attrib['pos'],sep=' ');original_position=position.copy()
        position[2]=metadata['table_height']+row['rest_height_m']
        body.attrib.pop('euler',None);body.set('quat',' '.join(format(x,'.17g') for x in quaternion))
        body.set('pos',' '.join(format(x,'.17g') for x in position))
        obj.update(pre_rest_initial=obj['initial'],requested_yaw=yaw,initial=position.tolist(),initial_quaternion=quaternion.tolist())
        matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,quaternion)
        obj['yaw']=float(np.arctan2(matrix[3],matrix[0]))
        changes.append({'body':obj['body'],'old_position':original_position.tolist(),'new_position':position.tolist(),
                        'rest_quaternion':quaternion.tolist()})
    candidate=scene.with_name(scene.stem+'.rest_candidate.xml');ET.ElementTree(root).write(candidate)
    prediction=mujoco.MjModel.from_xml_path(str(candidate))
    for index,key in enumerate(root.findall('keyframe/key')):
        values=before.key_qpos[index].copy()
        for obj in metadata['objects']:
            body=prediction.body(obj['body']).id;adr=int(prediction.jnt_qposadr[prediction.body_jntadr[body]])
            values[adr:adr+7]=prediction.qpos0[adr:adr+7]
        key.set('qpos',' '.join(format(x,'.17g') for x in values))
    ET.ElementTree(root).write(scene)
    after=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(after)
    if after.nkey:mujoco.mj_resetDataKeyframe(after,data,0)
    mujoco.mj_forward(after,data)
    for attr in ('body_mass','body_inertia','geom_size','geom_pos','geom_quat','geom_friction',
                 'geom_solref','geom_solimp','actuator_gainprm','actuator_forcerange','jnt_range','cam_pos','cam_quat','cam_fovy'):
        np.testing.assert_array_equal(getattr(before,attr),getattr(after,attr))
    fruit_ids={after.body(x['body']).id for x in metadata['objects']}
    for contact in data.contact:
        if all(int(after.geom_bodyid[g]) in fruit_ids for g in contact.geom) and contact.dist<-.0003:
            raise ValueError('Resting orientations introduced an initial fruit-fruit penetration; reject this scene')
    for obj in metadata['objects']:
        body=after.body(obj['body']).id;adr=int(after.jnt_qposadr[after.body_jntadr[body]])
        np.testing.assert_array_equal(data.qpos[adr:adr+2],before.qpos0[adr:adr+2])
        np.testing.assert_allclose(data.qpos[adr:adr+7],after.qpos0[adr:adr+7],rtol=0,atol=1e-14)
    metadata.update(fruit_spawn_profile='passive_rest_init_v2',
                    pre_rest_scene_protocol=metadata.get('scene_protocol','fruit_legacy_v1'),
                    scene_protocol=metadata.get('scene_protocol','fruit_legacy_v1')+'_passive_rest_init_v2',
                    rest_initialization={'calibration':str(calibration_path.resolve()),
                                         'calibration_sha256':hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                                         'xy_sizes_mass_inertia_friction_actuator_limits_unchanged':True,
                                         'scope':'MJCF/keyframe initialization before task physics only', 'changes':changes})
    scene.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    return metadata
