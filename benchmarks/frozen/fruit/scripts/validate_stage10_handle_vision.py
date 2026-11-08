#!/usr/bin/env python3
"""Independent RGB-D handle localization, Oracle used only after prediction to score."""
import argparse,json,math
from pathlib import Path
import mujoco,numpy as np
from _bootstrap import ROOT
from build_tasks import build
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.perception.door_handle import locate_handle


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[1081,1082,1083,1084,1085]);p.add_argument('--grasp',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use new perception evidence folder')
    a.output.mkdir(parents=True);rows=[]
    for seed in a.seeds:
        folder=a.output/str(seed);folder.mkdir();scene,meta=build('door',seed,folder/'scene.xml',door_profile='multistart_v2')
        sim=Simulation(scene);rng=np.random.default_rng([seed,1004])
        base=np.asarray(meta['opening_base'])+rng.uniform([-.03,-.03,-.08],[.03,.03,.08])
        sim.data.qpos[:2]=base[:2];sim.data.qpos[3:7]=[math.cos(base[2]/2),0,0,math.sin(base[2]/2)]
        mujoco.mj_forward(sim.model,sim.data);r=TaskRuntime(sim);r.transition(State.OBSERVE,'local_handle_rgbd');r.hold(.6)
        camera=RGBDCamera(sim);row={'seed':seed,'scope':'initialized near-door perception fixture, not full navigation/manipulation'}
        try:
            rgb,depth=camera.capture();camera.save(folder/'camera',rgb,depth)
            prediction=locate_handle(camera,rgb,depth)
            # Scoring is kept outside locate_handle; detector never receives
            # the model, handle geometry name, hinge or target position.
            truth=sim.data.geom_xpos[sim.model.geom('outer_handle').id].copy()
            error=float(np.linalg.norm(np.asarray(prediction['center'])-truth))
            row.update(prediction=prediction,oracle_scoring_position=truth.tolist(),error_m=error,success=error<.015)
            if a.grasp:
                try:
                    position=np.asarray(prediction['center']);vertical=np.asarray(prediction['axis']);vertical/=np.linalg.norm(vertical)
                    forward=position-camera.world_from_optical()[:3,3];forward-=vertical*(forward@vertical);forward/=np.linalg.norm(forward)
                    rotation=np.column_stack([vertical,np.cross(forward,vertical),forward])
                    r.transition(State.APPROACH,'rgbd_handle_local_grasp',target=position.tolist(),timeout=120)
                    r.gripper.command(0);r.move(position-.09*forward,rotation,seconds=2.2)
                    r.move(position,rotation,seconds=1.6);r.gripper.command(1);r.hold(.9)
                    # Oracle body association is exclusively a physical contact
                    # witness. It is never passed to the pose detector/planner.
                    handle_sides=set();geometry_contacts=[]
                    for number,contact in enumerate(sim.data.contact):
                        for finger,other in (contact.geom,contact.geom[::-1]):
                            finger_name=sim.model.body(int(sim.model.geom_bodyid[finger])).name or ''
                            if not finger_name.startswith('grip_') or sim.model.body(int(sim.model.geom_bodyid[other])).name!='door_leaf':continue
                            force=np.empty(6);mujoco.mj_contactForce(sim.model,sim.data,number,force)
                            capsule=sim.model.geom_type[other]==mujoco.mjtGeom.mjGEOM_CAPSULE
                            if capsule and force[0]>.05:
                                for side in ('left','right'):
                                    if finger_name.startswith('grip_'+side):handle_sides.add(side)
                            geometry_contacts.append({'finger':finger_name,'target_geom':sim.model.geom(int(other)).name or f'geom_{other}',
                                'handle_capsule':bool(capsule),'normal_force_n':float(force[0])})
                    row.update(grasp_success=handle_sides=={'left','right'},contact_geom_pairs=geometry_contacts,grasp_contacts=r.contacts('door_leaf'),
                               actual_tip=r.arm.pose()[0].tolist())
                    camera.save(folder/'grasp_camera',*camera.capture())
                except Exception as exc:row.update(grasp_success=False,grasp_error=str(exc))
        except Exception as exc:row.update(success=False,error=str(exc))
        finally:camera.close()
        rows.append(row);(folder/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        (a.output/'result.json').write_text(json.dumps({'episodes':len(rows),'successes':sum(x['success'] for x in rows),
            'grasp_attempted':a.grasp,'grasp_successes':sum(x.get('grasp_success',False) for x in rows),'results':rows},indent=2)+'\n')
        print(seed,row,flush=True)


if __name__=='__main__':main()
