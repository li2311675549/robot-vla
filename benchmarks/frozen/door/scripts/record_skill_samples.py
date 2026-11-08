#!/usr/bin/env python3
"""Five small Oracle skill fixtures using actual task geometry and physics."""
import argparse,copy,json
from pathlib import Path
import xml.etree.ElementTree as ET
from _bootstrap import ROOT
import mujoco
from robot_exam.simulation import Simulation
from robot_exam.skills.pick_place import PickPlace
from robot_exam.data.episodes import EpisodeRecorder,validate_episode


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/stage34/data_samples');p.add_argument('--skills',nargs='+',default=['apple','banana','orange','book_small','book_large']);p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--oracle-release-feedback',action='store_true');p.add_argument('--scale-min',type=float,default=1.);p.add_argument('--scale-max',type=float,default=1.)
    p.add_argument('--mobile',action='store_true');p.add_argument('--wrist-camera',action='store_true')
    p.add_argument('--parking',action='store_true');p.add_argument('--orange-pinch-offset',type=float,default=0.)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if not -.012<=a.orange_pinch_offset<=0:p.error('Orange pinch offset must be between -12 and 0 mm')
    if a.parking and not a.mobile:p.error('--parking requires the real mobile robot')
    if not 0.8<=a.scale_min<=a.scale_max<=1.2:raise ValueError('Scale interval must lie in [0.8,1.2]')
    fruit=ET.parse(ROOT/'scenes/fruit.xml');fm=json.loads((ROOT/'scenes/fruit.json').read_text());books=ET.parse(ROOT/'scenes/drawer.xml')
    report=[]
    for skill in a.skills:
      for seed in a.seeds:
        rng=__import__('numpy').random.default_rng(seed)
        output=a.output/skill/f'seed_{seed:04d}';output.mkdir(parents=True,exist_ok=True)
        if (output/'episode/episode.json').exists():raise FileExistsError(f'Refusing to overwrite {output}')
        root=ET.parse(ROOT/('scenes/mobile_manipulator.xml' if a.mobile else 'scenes/ur5_gripper.xml')).getroot()
        if a.mobile:
            root.find("worldbody/body[@name='table']").set('pos','.65 0 0')
            root.find("worldbody/site[@name='drop_target']").set('pos','.55 .22 .728')
        if a.wrist_camera:
            from robot_exam.perception.wrist_mount import add_wrist_camera
            add_wrist_camera(root)
        obj=root.find("worldbody/body[@name='object']")
        for g in list(obj.findall('geom')):obj.remove(g)
        if skill.startswith('book'):
            source=books.find(f"worldbody/body[@name='book_{0 if skill=='book_small' else 2}']")
            halfheight=.007 if skill=='book_small' else .013
        else:
            info=next(o for o in fm['objects'] if o['category']==skill)
            source=fruit.find(f"worldbody/body[@name='{info['body']}']")
            halfheight=info['initial'][2]-.725-.001
        size_scale=float(rng.uniform(a.scale_min,a.scale_max)) if a.scale_min!=a.scale_max else a.scale_min
        halfheight*=size_scale
        for index,g in enumerate(source.findall('geom')):
            geom=copy.deepcopy(g);geom.set('name','test_object' if index==0 else f'test_object_{index}')
            for field in ('size','pos','fromto'):
                if geom.get(field):geom.set(field,' '.join(str(float(v)*size_scale) for v in geom.get(field).split()))
            if geom.get('mass'):geom.set('mass',str(float(geom.get('mass'))*size_scale**3))
            obj.append(geom)
        fixture_x=.53+float(rng.uniform(-.025,.025));fixture_y=float(rng.uniform(-.07,.07));fixture_yaw=float(rng.uniform(-.35,.35))
        fixture_z=.725+halfheight+.001
        obj.set('pos',f'{fixture_x} {fixture_y} {fixture_z}');obj.set('euler',f'0 0 {fixture_yaw}')
        key=root.find('keyframe/key');q=key.get('qpos').split();q[-5]=str(.725+halfheight+.001);key.set('qpos',' '.join(q))
        q[-7]=str(fixture_x);q[-6]=str(fixture_y);q[-4]=str(__import__('math').cos(fixture_yaw/2));q[-1]=str(__import__('math').sin(fixture_yaw/2));key.set('qpos',' '.join(q))
        for e in root.findall('.//mesh'):
            if e.get('file'):e.set('file',str((ROOT/'scenes'/e.get('file')).resolve()))
        scene=output/'scene.xml';ET.ElementTree(root).write(scene)
        sim=Simulation(scene)
        sim.step(sim.data.ctrl,500)
        parking=None
        if a.parking:
            from robot_exam.control.parking import WheelParkingHold
            parking=WheelParkingHold(sim);parking.begin()
            for _ in range(1100):sim.step(sim.data.ctrl)
            if not parking.ready:raise RuntimeError('Mobile fixture did not pass parking settle gate')
        actual_fixture=sim.data.xpos[sim.model.body('object').id].copy()
        logger=EpisodeRecorder(sim,output/'episode',f'Pick up the {skill} and place it at the target.',seed,'oracle')
        logger.meta.update({'skill':skill,'fixture_position':actual_fixture.tolist(),
                            'fixture_yaw':fixture_yaw,'fixture_scale':size_scale,
                            'scope':'parked mobile short skill, no navigation' if a.mobile else 'fixed-arm short skill fixture; not mobile or drawer integration',
                            'base_control':'encoder_parking_v1' if parking else 'fixed_fixture' if not a.mobile else 'zero_velocity'})
        logger.meta['orange_pinch_offset_m']=a.orange_pinch_offset
        step=sim.step
        def recorded_step(ctrl,count=1):
            # PickPlace applies one prepared command per physical step.
            if count!=1:raise RuntimeError('Collector requires one-step control calls')
            sim.data.ctrl[:]=ctrl;logger.capture_pre_step('scripted_'+skill,arm_target=expert.qgoal,
                arm_reference=expert.arm.reference,max_speed=.7);return step(ctrl,count)
        sim.step=recorded_step
        try:
            class OracleReleaseExpert(PickPlace):
                def move(self,position,seconds,stage):
                    if skill=='orange' and stage=='descend':
                        position=__import__('numpy').asarray(position).copy();position[2]+=a.orange_pinch_offset
                    if a.oracle_release_feedback and stage=='lower':
                        position=__import__('numpy').asarray(position).copy()
                        position[:2]-=(self.sim.data.xpos[self.object_body]-self.arm.pose()[0])[:2]
                    return super().move(position,seconds,stage)
            expert=OracleReleaseExpert(sim)
            drop=[.55+float(rng.uniform(-.02,.02)),.22+float(rng.uniform(-.02,.02)),.725+halfheight]
            result=expert.run(actual_fixture,drop)
            result['oracle_release_feedback']=a.oracle_release_feedback
        except Exception as e:result={'status':'FAIL','error':str(e)}
        if 'expert' in locals():
            (output/'physical_trace.json').write_text(json.dumps({'rows':expert.rows,'stages':expert.stages}))
        logger.close(result['status']=='PASS')
        if parking:
            (output/'parking.json').write_text(json.dumps({'anchor':parking.anchor.tolist(),'ready':parking.ready,'samples':parking.samples},indent=2))
            parking.release()
        (output/'skill_result.json').write_text(json.dumps(result,indent=2))
        validation=validate_episode(output/'episode',replay=True)
        report.append({'skill':skill,'seed':seed,'physical_result':result,'validation':validation})
        print(skill,seed,result['status'],validation,flush=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
