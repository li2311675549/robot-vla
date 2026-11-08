#!/usr/bin/env python3
"""Independent passive-contact calibration; not robot manipulation evidence.

Copies each exact fruit's primitive geometry, mass and contact parameters onto
the same tabletop material/box section. Gravity alone finds a resting pose.
No running object pose writes, drives, welds or friction modifications.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from _bootstrap import ROOT


def vector(x):return ' '.join(format(float(v),'.17g') for v in x)


def calibrate(scene,output,seconds=600.):
    scene=Path(scene);output=Path(output)
    if output.exists():raise FileExistsError('Preserve calibration outcomes')
    output.mkdir(parents=True);source=mujoco.MjModel.from_xml_path(str(scene));tree=ET.parse(scene).getroot()
    meta=json.loads(scene.with_suffix('.json').read_text());table=source.geom('fruit_table_top').id
    result={'scope':'independent passive contact calibration, not a robot grasp or task',
            'source_scene':str(scene.resolve()),'source_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
            'seconds_requested_per_body':seconds,'rows':[]}
    for obj in meta['objects']:
        name=obj['body'];body_id=source.body(name).id
        root=ET.Element('mujoco',model='passive_'+name);ET.SubElement(root,'compiler',angle='radian')
        opt=ET.SubElement(root,'option',timestep=str(source.opt.timestep),gravity=vector(source.opt.gravity),
                          integrator='implicitfast',cone='elliptic',impratio=str(source.opt.impratio),
                          iterations=str(source.opt.iterations),tolerance=str(source.opt.tolerance))
        ET.SubElement(opt,'flag',multiccd='enable');world=ET.SubElement(root,'worldbody')
        ET.SubElement(world,'geom',name='calibration_table',type='box',size=vector(source.geom_size[table]),
                      pos='0 0 -.02',friction=vector(source.geom_friction[table]),
                      condim=str(int(source.geom_condim[table])),solref=vector(source.geom_solref[table]),
                      solimp=vector(source.geom_solimp[table]))
        body=copy.deepcopy(tree.find(f".//body[@name='{name}']"))
        body.attrib.pop('euler',None);body.attrib.pop('quat',None)
        body.set('pos',f'0 0 {float(obj["initial"][2])-.725:.17g}')
        source_geoms=np.flatnonzero(source.geom_bodyid==body_id)
        for index,(geom,gid) in enumerate(zip(body.findall('geom'),source_geoms)):
            geom.set('name',f'calibration_g{index}')
            for attr,values in [('friction',source.geom_friction[gid]),('solref',source.geom_solref[gid]),('solimp',source.geom_solimp[gid])]:
                geom.set(attr,vector(values))
            geom.set('condim',str(int(source.geom_condim[gid])))
            geom.set('priority',str(int(source.geom_priority[gid])))
        world.append(body);path=output/(name+'.xml');ET.ElementTree(root).write(path)
        model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        b=model.body(name).id
        np.testing.assert_allclose(model.body_mass[b],source.body_mass[body_id],rtol=1e-12)
        np.testing.assert_allclose(model.body_inertia[b],source.body_inertia[body_id],rtol=1e-10,atol=1e-14)
        for i,gid in enumerate(source_geoms):
            for attr in ['geom_size','geom_friction','geom_solref','geom_solimp']:
                np.testing.assert_allclose(getattr(model,attr)[i+1],getattr(source,attr)[gid],rtol=1e-12,atol=1e-14)
        samples=[];rolled_off=False
        for _ in range(int(np.ceil(seconds))):
            mujoco.mj_step(model,data,nstep=round(1/model.opt.timestep))
            samples.append({'time':float(data.time),'position':data.qpos[:3].tolist(),
                            'quaternion':data.qpos[3:7].tolist(),'speed':float(np.linalg.norm(data.qvel[:3])),
                            'angular_speed':float(np.linalg.norm(data.qvel[3:6]))})
            if data.warning.number.any():raise RuntimeError('MuJoCo warning in independent calibration')
            if data.qpos[2]<-.1:rolled_off=True;break
        tail=samples[-60:]
        stable=not rolled_off and len(tail)==60 and max(x['speed'] for x in tail)<1e-4 and max(x['angular_speed'] for x in tail)<.001
        row={'body':name,'category':obj['category'],'scale':obj['scale'],'stable':stable,'rolled_off':rolled_off,
             'rest_quaternion':data.qpos[3:7].tolist(),'rest_height_m':float(data.qpos[2]),
             'final_xy_in_calibration':data.qpos[:2].tolist(),'last60s_max_speed':max(x['speed'] for x in tail),
             'last60s_max_angular_speed':max(x['angular_speed'] for x in tail),
             'original_mass_inertia_geometry_contact_verified':True,'samples':samples}
        result['rows'].append(row);(output/'calibration.json').write_text(json.dumps(result,indent=2))
        print(name,obj['category'],stable,row['rest_height_m'],row['last60s_max_speed'],flush=True)
    result['status']='PASS_ALL_EXACT_FRUITS_STABLE' if all(x['stable'] for x in result['rows']) else 'PARTIAL_OR_UNSTABLE'
    (output/'calibration.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=600.);a=p.parse_args();calibrate(a.scene,a.output,a.seconds)
