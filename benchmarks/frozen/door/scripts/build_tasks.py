#!/usr/bin/env python3
"""Seeded task MJCF generation, reusing the validated stage-two robot verbatim."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from _bootstrap import ROOT
from build_stage2 import add, scene_root, husky, compose_arm, camera_module
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from robot_exam.tasks.runtime import TaskRuntime


def robot(root, start):
    body = husky(root)
    body.set("pos", f"{start[0]} {start[1]} .13428")
    body.set("euler", f"0 0 {start[2]}")
    mount = add(body, "body", name="arm_pedestal", pos="0 0 .24375")
    add(mount, "geom", type="cylinder", size=".09 .01625", mass=".8", rgba=".25 .25 .27 1")
    compose_arm(root, body, .26)
    camera_module(root, body, True)
    return body


def table(root, name, center, halfsize, height=.725):
    body = add(root.find("worldbody"), "body", name=name, pos=f"{center[0]} {center[1]} 0")
    add(body, "geom", name=name+"_top", type="box", pos=f"0 0 {height-.02}", size=f"{halfsize[0]} {halfsize[1]} .02", rgba=".48 .32 .18 1")
    for x in (-halfsize[0]+.025, halfsize[0]-.025):
        for y in (-halfsize[1]-.11, halfsize[1]+.11):
            add(body, "geom", name=f"table_leg_{name}_{x}_{y}", type="box", pos=f"{x} {y} {(height-.04)/2}", size=f".02 .02 {(height-.04)/2}", rgba=".25 .22 .18 1")
    return body


def label_asset(root, category, color):
    path = ROOT / "assets/task_labels" / (category+".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        im = Image.new("RGB", (256, 96), tuple(int(c*255) for c in color))
        draw = ImageDraw.Draw(im)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 30) if Path(font_path).exists() else ImageFont.load_default()
        draw.text((9, 28), category.upper(), fill="white", font=font)
        im.save(path)
    add(root.find("asset"), "texture", name="label_"+category, type="2d", file=str(path))
    add(root.find("asset"), "material", name="label_"+category, texture="label_"+category)


def bin_box(root, category, center, color):
    label_asset(root, category, color)
    body = add(root.find("worldbody"), "body", name="bin_"+category, pos=f"{center[0]} {center[1]} .55")
    add(body, "geom", name="bin_"+category+"_floor", type="box", pos="0 0 .006", size=".095 .09 .006", rgba=".25 .28 .3 1")
    for x in (-.10, .10):
        add(body, "geom", type="box", pos=f"{x} 0 .07", size=".005 .095 .07", rgba=" ".join(map(str, [*color, 1])))
    for y in (-.095, .095):
        add(body, "geom", type="box", pos=f"0 {y} .07", size=".10 .005 .07", rgba=" ".join(map(str, [*color, 1])))
    add(body, "geom", name="bin_label_"+category, type="plane", pos=".106 0 .25", euler="0 -1.57079632679 1.57079632679", size=".085 .031 .001", material="label_"+category, contype="0", conaffinity="0")


def fruit_scene(seed):
    rng = np.random.default_rng(seed)
    root = scene_root("fruit_classification")
    robot(root, [-4.64, 0, 0])
    # Finite silicone contact patches resist both torsion and rolling. Pad
    # priority otherwise silently overrides the fruit's six-dimensional contact.
    for cls in ("grip_pad_box1", "grip_pad_box2"):
        root.find(f".//default[@class='{cls}']/geom").set("condim", "6")
    table(root, "fruit_table", [.49, 0], [.13, .29])
    table(root, "bin_table", [.55, 1.35], [.18, .85], .55)
    categories = ["apple", "banana", "orange", "apple", "banana", "orange", "apple", "banana", "orange", "apple"]
    rng.shuffle(categories)
    metadata = {"task": "fruit", "seed": seed, "start": [-4.64, 0, 0],
                "distance_definition": "base origin to front table plane x=.36 (5m)",
                "table_height": .725, "objects": [], "bins": {},
                "obstacles": [[.36, .62, -.40, .40], [.37, .73, .39, 2.31]],
                "bin_stations": [[0,.85,0],[0,1.35,0],[0,1.85,0]],
                "staging": [-.40, 0, 0], "dock": [0, 0, 0]}
    for category, y, color in (("apple", .85, [.1,.65,.75]), ("banana", 1.35, [.60,.15,.75]), ("orange", 1.85, [.12,.25,.85])):
        bin_box(root, category, [.55, y], color)
        metadata["bins"][category] = {"center": [.55, y, .55], "inside_halfsize": [.095, .09], "rim_z": .69, "label_rgb": color}
    for i, category in enumerate(categories):
        x, y = .435 + (i//5)*.11 + rng.uniform(-.006,.006), (i%5-2)*.105+rng.uniform(-.004,.004)
        yaw = rng.uniform(-.45,.45)
        scale = rng.uniform(.88,1.08)
        halfheight = .027*scale if category != "banana" else .017*scale
        body = add(root.find("worldbody"), "body", name=f"fruit_{i}", pos=f"{x} {y} {.725+halfheight+.001}", euler=f"0 0 {yaw}")
        add(body, "freejoint", name=f"fruit_{i}_free")
        # Geom masses sum to the stated physical fruit mass; no object actuator.
        common = {"friction": "1 .01 .002", "condim": "6"}
        if category == "apple":
            add(body, "geom", name=f"fruit_{i}_g0", type="ellipsoid", size=f"{.026*scale} {.025*scale} {halfheight}", mass=".10", rgba=".85 .045 .025 1", **common)
            add(body, "geom", name=f"fruit_{i}_g1", type="ellipsoid", pos=f"{.008*scale} 0 {.003*scale}", size=f"{.020*scale} {.023*scale} {.024*scale}", mass=".025", rgba=".85 .045 .025 1", **common)
            add(body, "geom", type="capsule", fromto=f"0 0 {halfheight-.003} 0 0 {halfheight+.008}", size=".002", mass=".001", rgba=".22 .12 .025 1")
        elif category == "orange":
            add(body, "geom", name=f"fruit_{i}_g0", type="ellipsoid", size=f"{.026*scale} {.027*scale} {halfheight}", mass=".12", rgba=".95 .32 .025 1", **common)
        else:
            for k, (ax, ay, bx, by) in enumerate(((-.032,.002,-.012,-.005),(-.012,-.005,.012,-.005),(.012,-.005,.032,.002))):
                add(body, "geom", name=f"fruit_{i}_g{k}", type="capsule", fromto=f"{ax*scale} {ay*scale} 0 {bx*scale} {by*scale} 0", size=f"{halfheight}", mass=".03", rgba=".92 .80 .045 1", **common)
        metadata["objects"].append({"body":f"fruit_{i}", "category":category, "initial":[x,y,.725+halfheight+.001], "yaw":yaw, "scale":scale})
    return root, metadata


def door_scene(seed,door_profile='fixed_v1'):
    rng = np.random.default_rng(seed)
    root = scene_root("pull_enter_close_door")
    start=[-5.09,0.,0.]
    aligned=door_profile=='aligned_handles_v3'
    if door_profile in ('multistart_v2','aligned_handles_v3'):
        initial_rng=np.random.default_rng([seed,802])
        lateral=float(initial_rng.uniform(-.25,.25))
        start=[-.09-math.sqrt(25-lateral*lateral),lateral,float(initial_rng.uniform(-.18,.18))]
    elif door_profile!='fixed_v1':raise ValueError(f'Unknown door profile: {door_profile}')
    robot(root,start)
    world = root.find("worldbody")
    wall_halfheight=1.075 if aligned else .80
    for y in (-1.9, 1.9):
        add(world, "geom", name=f"wall_front_{y}", type="box", pos=f"0 {y} {wall_halfheight}", size=f".06 1.10 {wall_halfheight}", rgba=".7 .7 .67 1")
    for y in (-3.,3.):
        add(world,"geom",name=f"wall_side_{y}",type="box",pos=f"1.9 {y} {wall_halfheight}",size=f"1.9 .06 {wall_halfheight}",rgba=".7 .7 .67 1")
    add(world,"geom",name="wall_back",type="box",pos=f"3.8 0 {wall_halfheight}",size=f".06 3 {wall_halfheight}",rgba=".7 .7 .67 1")
    for y in (-.81,.81):
        add(world,"geom",name=f"door_frame_{y}",type="box",pos=f"0 {y} {1.03 if aligned else .8}",size=f".07 .025 {1.03 if aligned else .8}",rgba=".3 .22 .15 1")
    add(world,"geom",name="door_frame_top",type="box",pos=f"0 0 {2.085 if aligned else 1.62}",size=".07 .835 .025",rgba=".3 .22 .15 1")
    leaf = add(world, "body", name="door_leaf", pos="-.09 -.8 0")
    add(leaf,"joint",name="door_hinge",type="hinge",axis="0 0 1",range="0 1.65",damping=".75",frictionloss=".30")
    add(leaf,"geom",name="door_panel",type="box",pos="0 .80 1.02" if aligned else "0 .80 .76",size=".022 .78 1.0" if aligned else ".022 .78 .76",mass="18" if aligned else "9",rgba=".47 .28 .12 1",friction=".7 .01 .001")
    z = (.95 if aligned else .79)+rng.uniform(-.008,.008)
    radius=1.40 if aligned else 1.22
    inner_radius=radius if aligned else .22
    inner_height=z if aligned else .65
    for name, x, y in (("outer_handle",-.065,radius),("inner_handle",.065,inner_radius)):
        h=inner_height if name=='inner_handle' else z
        half=.07 if aligned else .055
        add(leaf,"geom",name=name,type="capsule",fromto=f"{x} {y} {h-half} {x} {y} {h+half}",size=".012",mass=".07",rgba=".15 .8 .85 1",friction="1 .01 .001",condim="4")
        if aligned:
            # A pull bar is supported at both ends, leaving its middle free.
            # The legacy central T-standoff occupied the gripper pinch region.
            for label,offset in (('lower',-.06),('upper',.06)):
                add(leaf,"geom",name=name+'_mount_'+label,type="capsule",fromto=f"0 {y} {h+offset} {x} {y} {h+offset}",size=".008",mass=".03",rgba=".6 .6 .6 1")
        else:
            add(leaf,"geom",type="capsule",fromto=f"0 {y} {h} {x} {y} {h}",size=".008",mass=".03",rgba=".6 .6 .6 1")
        add(leaf,"site",name=name,pos=f"{x} {y} {h}",size=".003")
    metadata={"task":"door","seed":seed,"door_profile":door_profile,"scene_protocol":'door_'+door_profile,"start":start,"hinge":[-.09,-.8,0],"door_width":1.56,"doorway_clear_width":1.57,
                  "start_distance_reference":[-.09,0.],"start_distance_m":float(math.hypot(start[0]+.09,start[1])),
                  "handle_height":z,"inner_handle_height":inner_height,"inner_handle_radius":inner_radius,"handle_r":radius,"open_angle":1.56,
                  "obstacles":[[-.07,.07,-3,-.78],[-.07,.07,.78,3]],
                  "opening_base":[-.905,.42,.59],"closing_base":[.50,-.30,-1.5707963267948966]}
    if aligned:
        metadata.update(handle_layout='paired_latch_edge',door_height=2.0,door_panel_mass_kg=18.,
                        handle_edge_inset_m=.18,handle_approach_offset_m=.065,
                        handle_bar_length_m=.14,handle_mount_separation_m=.12,
                        closing_yaw_bias_rad=-.30,
                        opening_base=[-.905,.60,math.atan2(.815,radius)],
                        closing_base=[.65,0.,math.pi],
                        geometry_note='Wide 1.57m robot passage retained; both faces share the same latch-edge handle axis and height. No hinge-side auxiliary pull handle.')
    return root,metadata


def drawer_scene(seed,book_profile='booklet'):
    rng = np.random.default_rng(seed)
    root = scene_root("three_books_three_drawers")
    robot(root, [-4.62,-.65,0])
    if book_profile=='booklet':
        sizes=[[.105,.048,.014],[.135,.058,.020],[.165,.068,.026]]
        table_half=[.12,.29];slot_spacing=.17;yaw_center=0.;yaw_span=.5
        masses=[.08,.12,.16];obstacles=[[.38,.62,-1.01,-.29],[.55,.93,.28,.72]]
    elif book_profile=='a6_a5_a4':
        sizes=[[.148,.105,.018],[.210,.148,.022],[.297,.210,.028]]
        table_half=[.22,.55];slot_spacing=.32;yaw_center=math.pi/2;yaw_span=.20
        masses=[.20,.45,.75];obstacles=[[.27,.71,-1.20,-.10],[.55,.93,.28,.72]]
    elif book_profile=='distinct_v2':
        sizes=[[.120,.080,.015],[.180,.120,.020],[.240,.160,.025]]
        table_half=[.22,.48];slot_spacing=.30;yaw_center=math.pi/2;yaw_span=.18
        masses=[.15,.30,.50];obstacles=[[.28,.72,-1.24,-.06],[.55,.93,.28,.72]]
        # New protocol: exactly five metres from the table front plane.
        base=root.find(".//body[@name='husky']");base.set('pos','-4.72 -.65 .13428')
    else:raise ValueError(f'Unknown book profile: {book_profile}')
    table(root,"book_table",[.50,-.65],table_half)
    world = root.find("worldbody")
    objects=[]
    order=rng.permutation(3)
    for slot,rank in enumerate(order):
        length,width,thickness=sizes[rank]
        x=.49+rng.uniform(-.010,.010);y=-.65+(slot-1)*slot_spacing+rng.uniform(-.008,.008);yaw=yaw_center+rng.uniform(-yaw_span,yaw_span)
        body=add(world,"body",name=f"book_{rank}",pos=f"{x} {y} {.725+thickness/2+.001}",euler=f"0 0 {yaw}")
        add(body,"freejoint",name=f"book_{rank}_free")
        add(body,"geom",name=f"book_{rank}_cover",type="box",size=f"{length/2} {width/2} {thickness/2}",mass=str(masses[rank]),rgba=[".8 .06 .08 1",".08 .65 .12 1",".08 .15 .85 1"][rank],friction="1 .01 .001",condim="4")
        add(body,"geom",name=f"book_{rank}_pages",type="box",pos="0 0 0",size=f"{length/2-.002} {width/2-.002} {max(.002,thickness/2-.002)}",mass=".001",rgba=".9 .88 .75 1",contype="0",conaffinity="0")
        objects.append({"body":f"book_{rank}","size":sizes[rank],"area":length*width,"rank":int(rank),"initial":[x,y,.725+thickness/2+.001],"yaw":yaw})
    cabinet = add(world,"body",name="cabinet",pos=".58 .50 0")
    for y in (-.212,.212):
        add(cabinet,"geom",name=f"cabinet_shell_side_{y}",type="box",pos=f".16 {y} .5",size=".17 .01 .43",rgba=".35 .38 .42 1")
    add(cabinet,"geom",name="cabinet_shell_back",type="box",pos=".33 0 .5",size=".01 .21 .43",rgba=".35 .38 .42 1")
    for layer,z in enumerate((.34,.53,.72)):
        drawer=add(cabinet,"body",name=f"drawer_{layer}",pos=f"0 0 {z}")
        add(drawer,"joint",name=f"drawer_slide_{layer}",type="slide",axis="-1 0 0",range="0 .27",damping="2",frictionloss=".5")
        add(drawer,"geom",name=f"drawer_{layer}_bottom",type="box",pos=".14 0 0",size=".14 .19 .008",mass=".6",rgba=".55 .42 .24 1",friction=".9 .01 .001")
        add(drawer,"geom",name=f"drawer_{layer}_front",type="box",pos="0 0 .07",size=".009 .19 .07",mass=".25",rgba=".5 .32 .16 1")
        for y in (-.19,.19):
            add(drawer,"geom",type="box",pos=f".14 {y} .06",size=".14 .006 .06",mass=".08",rgba=".55 .42 .24 1")
        add(drawer,"geom",type="box",pos=".275 0 .06",size=".006 .19 .06",mass=".08",rgba=".55 .42 .24 1")
        add(drawer,"geom",name=f"drawer_{layer}_handle",type="capsule",fromto="-.055 -.06 .165 -.055 .06 .165",size=".012",mass=".05",rgba=".1 .8 .8 1",friction="1 .01 .001",condim="4")
        for y in (-.06,.06):
            add(drawer,"geom",type="capsule",fromto=f"-.008 {y} .10 -.055 {y} .10",size=".007",mass=".015",rgba=".6 .6 .6 1")
            add(drawer,"geom",type="capsule",fromto=f"-.055 {y} .10 -.055 {y} .165",size=".007",mass=".015",rgba=".6 .6 .6 1")
        add(drawer,"site",name=f"drawer_handle_{layer}",pos="-.055 0 .165",size=".003")
    return root,{"task":"drawer","seed":seed,"book_profile":book_profile,"scene_protocol":'drawer_'+book_profile,"size_sort_rule":"planar area length*width","start":[-4.72 if book_profile=='distinct_v2' else -4.62,-.65,0],"table_height":.725,"objects":objects,
                 "observation_workspace":[.25,.75,-1.20,-.10] if book_profile=='distinct_v2' else [.34,.64,-1.,-.30],
                 "obstacles":obstacles,
                 "book_dock":[0,-.65,0],"cabinet_dock":[-.20,.5,0],"drawer_bottoms":[.34,.53,.72],"drawer_handle_offset":.165,"drawer_travel":.27}


def build(task, seed=0, output=None, book_profile='booklet',door_profile='fixed_v1',wrist_camera=False):
    root, metadata = ({"fruit":fruit_scene,"door":lambda s:door_scene(s,door_profile),"drawer":lambda s:drawer_scene(s,book_profile)}[task])(seed)
    if wrist_camera:
        from robot_exam.perception.wrist_mount import add_wrist_camera
        metadata['wrist_camera']=add_wrist_camera(root)
        metadata['scene_protocol']=metadata.get('scene_protocol','fruit_seeded_v1')+'_wrist_rgb_v1'
    output = Path(output).resolve() if output else ROOT / "scenes" / (task+".xml")
    output.parent.mkdir(parents=True,exist_ok=True)
    # Normalize paths independently of the destination directory.
    for element in root.findall(".//mesh")+root.findall(".//texture"):
        filename=element.get("file")
        if filename:
            absolute=Path(filename) if Path(filename).is_absolute() else (ROOT/"scenes"/filename).resolve()
            element.set("file",str(absolute))
    pre=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    q=pre.qpos0.copy();ctrl=np.zeros(pre.nu)
    for name,value in zip(("shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"),TaskRuntime.STOW):
        q[pre.jnt_qposadr[pre.joint(name).id]]=value
        ctrl[pre.actuator(name+"_position").id]=value
    key=add(root,"keyframe")
    add(key,"key",name="home",qpos=" ".join(map(str,q)),ctrl=" ".join(map(str,ctrl)))
    for element in root.findall(".//mesh")+root.findall(".//texture"):
        if element.get("file"):
            element.set("file",os.path.relpath(element.get("file"),output.parent))
    ET.indent(root)
    ET.ElementTree(root).write(output,encoding="unicode",xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(output))
    assert model.nu==11 and model.nmocap==0 and not np.any(model.eq_type==mujoco.mjtEq.mjEQ_WELD)
    metadata["model"]={"nq":model.nq,"nv":model.nv,"nu":model.nu,"path":os.path.relpath(output,ROOT)}
    if metadata.get('handle_layout')=='paired_latch_edge':
        metadata['door_total_mass_kg']=float(model.body_mass[model.body('door_leaf').id])
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2))
    return output,metadata


def main():
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["all","fruit","door","drawer"],default="all");p.add_argument("--seed",type=int,default=0);p.add_argument('--book-profile',choices=['booklet','a6_a5_a4','distinct_v2'],default='booklet');p.add_argument('--door-profile',choices=['fixed_v1','multistart_v2','aligned_handles_v3'],default='fixed_v1');a=p.parse_args()
    for task in (["fruit","door","drawer"] if a.task=="all" else [a.task]):
        path,metadata=build(task,a.seed,book_profile=a.book_profile,door_profile=a.door_profile);print(task,path,metadata["model"])


if __name__=="__main__":main()
