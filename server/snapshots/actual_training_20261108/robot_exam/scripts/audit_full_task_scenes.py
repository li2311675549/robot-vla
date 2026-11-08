#!/usr/bin/env python3
"""Render and check the complete task scenes, distinct from single-fruit VLA fixtures."""
import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from _bootstrap import ROOT
from build_tasks import build
from robot_exam.simulation import Simulation


def audit(task,seed,output,layout_profile='legacy_v1'):
    model,meta=build(task,seed,output/'scene.xml',book_profile='distinct_v2',door_profile='aligned_handles_v3',wrist_camera=True,
                     layout_profile=layout_profile if task!='door' else 'legacy_v1',
                     bin_label_profile='upright_v2' if layout_profile!='legacy_v1' else 'legacy_v1')
    sim=Simulation(model);sim.step(sim.data.ctrl,350)
    m,d=sim.model,sim.data;checks={};details={}
    free=[m.body(int(m.jnt_bodyid[j])).name for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
    checks['original_ur5_six_joints']=all(m.joint(n).id>=0 for n in ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'))
    checks['no_mocap_or_weld']=m.nmocap==0 and not np.any(m.eq_type==mujoco.mjtEq.mjEQ_WELD)
    checks['body_and_wrist_cameras']=m.camera('d435i_rgbd').id>=0 and m.camera('wrist_rgb').id>=0
    if task=='door':
        adr=m.jnt_qposadr[m.joint('door_hinge').id]
        distance=math.dist(meta['start'][:2],meta['start_distance_reference'])
        checks.update(initially_closed=abs(float(d.qpos[adr]))<.035,single_passive_hinged_leaf=not any(int(j)==m.joint('door_hinge').id for j in m.actuator_trnid[:,0]),
                      start_5m=abs(distance-5)<1e-8,paired_latch_edge_handle=meta['handle_layout']=='paired_latch_edge')
        details.update(start_distance_m=distance,clear_doorway_m=meta['doorway_clear_width'],handle_radius_m=meta['handle_r'],door_profile=meta['door_profile'])
        focus=[.3,0,1.];distance_view=5.8;full_focus=[-.6,0,.8];full_distance=11.
    else:
        objects=meta['objects'];name='fruit_table' if task=='fruit' else 'book_table'
        geom=m.geom(name+'_top').id;front=float(d.geom_xpos[geom,0]-m.geom_size[geom,0]);distance=front-meta['start'][0]
        checks['start_5m']=abs(distance-5)<1e-8
        checks['all_targets_are_dynamic']=all(o['body'] in free for o in objects)
        details.update(start_distance_to_table_front_m=distance,table_width_depth_m=(2*m.geom_size[geom,:2]).tolist(),target_count=len(objects))
        if task=='fruit':
            categories={s:sum(o['category']==s for o in objects) for s in ('apple','banana','orange')}
            checks.update(exactly10_fruits=len(objects)==10,all_three_classes=all(categories.values()),
                          within_class_scale_differences=all(len({round(o['scale'],9) for o in objects if o['category']==s})>1 for s in categories),
                          three_collidable_bins=all(m.geom('bin_'+s+'_floor').contype>0 for s in categories),
                          three_label_materials_present=all(m.geom('bin_label_'+s).matid>=0 for s in categories))
            details.update(class_counts=categories,poses=[{'body':o['body'],'category':o['category'],'position':o['initial'],'yaw':o['yaw'],'scale':o['scale']} for o in objects],
                           randomization='Permuted class assignment; two rows with seeded position jitter, orientation and scale variations. Not an arbitrary clutter distribution.',
                           label_readability='Material presence is not a text visibility test. Separate actual-camera preview found inverted lettering in legacy_v1; upright_v2 fixes only the texture.')
            focus=[.5,.65,.6];distance_view=3.4
        else:
            checks.update(exactly3_books=len(objects)==3,distinct_planar_areas=len({round(o['area'],9) for o in objects})==3,
                          three_passive_drawers=all(m.joint('drawer_slide_'+str(i)).type==mujoco.mjtJoint.mjJNT_SLIDE for i in range(3)))
            details.update(book_sizes_m=[o['size'] for o in sorted(objects,key=lambda o:o['rank'])],
                           poses=[{'body':o['body'],'position':o['initial'],'yaw':o['yaw'],'rank':o['rank']} for o in objects],
                           drawer_numbering='Legacy first layer is the lowest drawer; this convention is explicit and must not be silently changed.',
                           randomization='Permuted size slots with seeded bounded position/yaw variations; not full360-degree clutter.')
            focus=[.5,-.15,.65];distance_view=3.5
        full_focus=[-1.9,.25,.6];full_distance=8.5
    severe=[]
    names={o['body'] for o in meta.get('objects',[])}
    for c in d.contact:
        bodies=[m.body(int(m.geom_bodyid[g])).name for g in c.geom]
        if c.dist<-.003 and any(b in names for b in bodies):severe.append({'bodies':bodies,'distance':float(c.dist)})
    checks['no_severe_target_initial_penetration']=not severe
    if meta.get('layout_randomization'):details['randomization']=meta['layout_randomization']
    details['scene_protocol']=meta.get('scene_protocol','fruit_seeded_v1')
    details['severe_target_contacts']=severe;details['data_warning']=[int(w.number) for w in d.warning]
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        renderer.update_scene(d)
        from OpenGL import GL
        details['renderer']=GL.glGetString(GL.GL_RENDERER).decode()
        for filename,lookat,dist in [('work_area.png',focus,distance_view),('five_metre_start.png',full_focus,full_distance)]:
            camera=mujoco.MjvCamera();camera.lookat[:]=lookat;camera.distance=dist;camera.azimuth=145;camera.elevation=-55
            renderer.update_scene(d,camera=camera);iio.imwrite(output/filename,renderer.render())
    result=dict(task=task,seed=seed,checks=checks,details=details,structural_checks_passed=all(checks.values()),
                scope='Scene structure and initial settling only; not a completed task or policy success test.')
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=200001)
    p.add_argument('--layout-profile',choices=('legacy_v1','bounded_scatter_v3'),default='legacy_v1');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior scene audits')
    a.output.mkdir(parents=True)
    results=[]
    for task in ('door','fruit','drawer'):
        folder=a.output/task;folder.mkdir();results.append(audit(task,a.seed,folder,a.layout_profile))
    (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# 完整任务场景核对','','这里不是驻车单水果VLA训练场景，而是原始三项完整任务。','']
    for row in results:
        task=row['task'];lines += [f'## {task}','','![操作区]('+task+'/work_area.png)','','![5m起点]('+task+'/five_metre_start.png)', '',f"结构检查：{row['structural_checks_passed']}；详见[{task}/audit.json]({task}/audit.json)。",'']
    lines += ['数量要求为水果10个、书3本，没有额外干扰物要求。当前随机化为有边界分布，不能称为任意杂乱场景；结构检查不替代完整动作与十回合成功率。']
    (a.output/'场景核对.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:x[k] for k in ('task','structural_checks_passed','checks')} for x in results],indent=2))


if __name__=='__main__':main()
