#!/usr/bin/env python3
"""Generate source-derived primitive Husky and composed original-UR5 models.

Keeps all stage-one/upstream files unchanged. No new downloads are needed.
"""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

from _bootstrap import ROOT
import mujoco


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def write(root, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    print(f"BUILD {path.relative_to(ROOT)} nq={model.nq} nv={model.nv} nu={model.nu}")


def scene_root(name):
    root = ET.Element("mujoco", model=name)
    add(root, "compiler", angle="radian", autolimits="true")
    option = add(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic", impratio="10")
    add(option, "flag", multiccd="enable")
    add(root, "size", njmax="3000", nconmax="600")
    visual = add(root, "visual")
    add(visual, "global", offwidth="640", offheight="480")
    add(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.8 0.8 0.8")
    asset = add(root, "asset")
    add(asset, "texture", name="floor_tex", type="2d", builtin="checker", rgb1=".2 .24 .28", rgb2=".28 .32 .36", width="128", height="128")
    add(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="12 12", texuniform="true")
    world = add(root, "worldbody")
    add(world, "light", pos="0 -2 4", diffuse="0.9 0.9 0.9")
    add(world, "light", pos="2 2 3")
    add(world, "geom", name="floor", type="plane", size="8 8 .1", material="floor_mat", friction=".9 .005 .0001")
    add(root, "contact")
    add(root, "actuator")
    return root


def husky(root):
    source = ET.parse(ROOT / "third_party/husky/husky_description/urdf/husky.urdf.xacro").getroot()
    ns = "{http://ros.org/wiki/xacro}"
    props = {e.get("name"): float(e.get("value")) for e in source
             if e.tag.endswith("property") and e.get("name") in
             ("base_x_size", "base_y_size", "base_z_size", "wheelbase", "track", "wheel_vertical_offset", "wheel_length", "wheel_radius")}
    wheel_source = ET.parse(ROOT / "third_party/husky/husky_description/urdf/wheel.urdf.xacro").getroot()
    inertial = source.find("./link[@name='inertial_link']/inertial")
    wheel_inertial = wheel_source.find(".//inertial")
    body = add(root.find("worldbody"), "body", name="husky", pos=f"0 0 {props['wheel_radius'] - props['wheel_vertical_offset'] + .002}")
    add(body, "freejoint", name="husky_free")
    tensor = inertial.find("inertia")
    add(body, "inertial", mass=inertial.find("mass").get("value"), pos=inertial.find("origin").get("xyz"),
        fullinertia=" ".join(tensor.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
    bx, by, bz = (props[k] for k in ("base_x_size", "base_y_size", "base_z_size"))
    add(body, "geom", name="chassis_lower", type="box", pos=f"0 0 {bz/4}", size=f"{bx/2} {by/2} {bz/4}", rgba=".75 .62 .12 1", mass="0")
    add(body, "geom", name="chassis_upper", type="box", pos=f"0 0 {bz*3/4-.01}", size=f"{bx*.4} {by/2} {bz/4-.01}", rgba=".20 .21 .23 1", mass="0")
    for axle, sx in (("front", 1), ("rear", -1)):
        for side, sy in (("left", 1), ("right", -1)):
            name = f"{axle}_{side}_wheel"
            wheel = add(body, "body", name=name + "_link", pos=f"{sx*props['wheelbase']/2} {sy*props['track']/2} {props['wheel_vertical_offset']}")
            tensor = wheel_inertial.find("inertia")
            add(wheel, "inertial", mass=wheel_inertial.find("mass").get("value"), pos="0 0 0",
                diaginertia=" ".join(tensor.get(k) for k in ("ixx", "iyy", "izz")))
            add(wheel, "joint", name=name, type="hinge", axis="0 1 0", limited="false", damping=".1", armature=".01")
            add(wheel, "geom", name=name + "_tire", type="cylinder", euler="1.57079632679 0 0", size=f"{props['wheel_radius']} {props['wheel_length']/2}",
                rgba=".06 .06 .07 1", friction=".9 .005 .0001", condim="6", mass="0")
            add(wheel, "geom", type="box", size=".01 .06 .11", rgba=".55 .55 .55 1", contype="0", conaffinity="0", mass="0")
            add(root.find("actuator"), "velocity", name=name + "_velocity", joint=name, kv="60", ctrlrange="-12 12", forcerange="-60 60")
    return body


def compose_arm(root, parent, mount_z):
    arm = ET.parse(ROOT / "assets/ur5/ur5.xml").getroot()
    root.append(copy.deepcopy(arm.find("default")))
    for mesh in arm.find("asset"):
        mesh.set("file", "../third_party/universal_robot/ur_description/meshes/ur5/collision/" + Path(mesh.get("file")).name)
        root.find("asset").append(mesh)
    arm_body = copy.deepcopy(arm.find("worldbody/body"))
    arm_body.set("pos", f"0 0 {mount_z}")
    parent.append(arm_body)
    # Original fixed base alias clashes with gripper base. Rename only alias.
    alias = arm_body.find("body[@name='base']")
    if alias is not None:
        alias.set("name", "ur_base_frame")
    for tag in ("contact", "actuator"):
        for element in arm.find(tag):
            root.find(tag).append(copy.deepcopy(element))
    gripper = ET.parse(ROOT / "assets/robotiq_2f85/2f85.xml").getroot()
    for mesh in gripper.findall("asset/mesh"):
        mesh.set("name", Path(mesh.get("file")).stem)
    names = {v for e in gripper.iter() for k, v in e.attrib.items() if k in ("name", "class", "childclass")}
    for e in gripper.iter():
        for key, value in list(e.attrib.items()):
            if key == "file":
                e.set(key, "../assets/robotiq_2f85/assets/" + Path(value).name)
            elif value in names:
                e.set(key, "grip_" + value)
    # Soft silicone pads transmit a finite torsional moment over their patch.
    # The upstream priority=1 pads otherwise force condim=3 even when a round
    # object requests torsional contact, allowing it to roll out of the grasp.
    for cls in ("grip_pad_box1", "grip_pad_box2"):
        pad = gripper.find(f".//default[@class='{cls}']/geom")
        pad.set("condim", "4")
        pad.set("friction", ".7 .01 .001")
    # Prefixing each implicit mesh name avoids UR5 'base' mesh collision.
    root.find("asset").extend(list(gripper.find("asset")))
    root.find("default").extend(list(gripper.find("default")))
    mount = arm_body.find(".//body[@name='tool0']")
    add(mount, "site", name="ur5_flange", size=".005", rgba="0 0 1 1")
    mount.append(copy.deepcopy(gripper.find("worldbody/body")))
    for tag in ("contact", "actuator"):
        root.find(tag).extend(list(gripper.find(tag)))
    for tag in ("tendon", "equality"):
        root.append(copy.deepcopy(gripper.find(tag)))


def task_geometry(root, mobile, variant="box"):
    world = root.find("worldbody")
    # Same work geometry relative to the final parking pose for both benches.
    x = 1.65 if mobile else .65
    table = add(world, "body", name="table", pos=f"{x} 0 0")
    add(table, "geom", name="table_top", type="box", pos="0 0 .70", size=".38 .48 .025", rgba=".55 .38 .22 1")
    for px in (-.30, .30):
        for py in (-.40, .40):
            add(table, "geom", type="box", pos=f"{px} {py} .34", size=".025 .025 .34", rgba=".3 .25 .2 1")
    dimensions = {"box": ("box", ".022 .022 .025", .025),
                  "small": ("box", ".015 .018 .020", .020),
                  "wide": ("box", ".030 .026 .030", .030),
                  "cylinder": ("cylinder", ".022 .025", .025)}
    shape, size, halfheight = dimensions[variant]
    obj = add(world, "body", name="object", pos=f"{x-.10} 0 {.725+halfheight+.002}")
    add(obj, "freejoint", name="object_free")
    add(obj, "geom", name="test_object", type=shape, size=size, mass=".10", rgba=".9 .12 .08 1", friction="1 .01 .001", condim="4")
    add(world, "site", name="drop_target", pos=f"{x-.10} .22 .728", type="cylinder", size=".07 .002", rgba=".1 .7 .2 .7")


def camera_module(root, parent, mobile):
    # Lightweight geometric envelope, not the 44MiB Menagerie visual mesh.
    # Optical axes: MuJoCo +X right, +Y up, -Z forward; RGB-D is ideal/registered.
    z = .95 if mobile else 1.08428
    camera = add(parent, "body", name="d435i_virtual", pos=f"-.25 -.25 {z}")
    add(camera, "geom", name="d435i_envelope", type="box", size=".045 .0125 .0125", mass=".072", rgba=".7 .7 .72 1")
    add(camera, "camera", name="d435i_rgbd", pos=".015 0 0", xyaxes="0 -1 0 .385 0 .923", fovy="58")
    add(camera, "site", name="d435i_mount", size=".003")
    if mobile:
        arch = add(parent, "body", name="camera_arch")
        for y in (-.25, .25):
            add(arch, "geom", name=f"camera_arch_{'left' if y>0 else 'right'}", type="capsule", fromto=f"-.25 {y} .25 -.25 {y} .95", size=".012", mass=".2", rgba=".25 .25 .27 1")
        add(arch, "geom", type="capsule", fromto="-.25 -.25 .95 -.25 .25 .95", size=".012", mass=".15", rgba=".25 .25 .27 1")


def add_home(root, mobile, with_arm, object_height=.752):
    # Positions supplied only at reset via model keyframe, never during actions.
    qpos = ([0, 0, .13428, 1, 0, 0, 0] + [0]*4) if mobile else []
    ctrl = [0]*4 if mobile else []
    if with_arm:
        home = [0, -1.9, 1.57, -1.24, -1.57, 0]
        qpos += home + [0]*8
        ctrl += home + [0]
        qpos += [1.55 if mobile else .55, 0, object_height, 1, 0, 0, 0]
    keys = add(root, "keyframe")
    add(keys, "key", name="home", qpos=" ".join(map(str, qpos)), ctrl=" ".join(map(str, ctrl)))


def main():
    base = scene_root("husky_a200_source_derived")
    husky(base)
    add_home(base, True, False)
    write(base, ROOT / "scenes/husky.xml")
    for mobile, name in ((False, "ur5_gripper"), (True, "mobile_manipulator")):
        root = scene_root(name)
        parent = husky(root) if mobile else root.find("worldbody")
        if mobile:
            mount = add(parent, "body", name="arm_pedestal", pos="0 0 .24375")
            add(mount, "geom", type="cylinder", size=".09 .01625", mass=".8", rgba=".25 .25 .27 1")
        else:
            add(parent, "geom", name="bench_pedestal", type="cylinder", pos="0 0 .1975", size=".12 .1975", rgba=".25 .25 .27 1")
        compose_arm(root, parent, .26 if mobile else .395)
        camera_module(root, parent, mobile)
        task_geometry(root, mobile)
        add_home(root, mobile, True)
        write(root, ROOT / f"scenes/{name}.xml")
        if not mobile:
            for variant, height in (("small", .747), ("wide", .757), ("cylinder", .752)):
                alternate = copy.deepcopy(root)
                obj = alternate.find("worldbody/body[@name='object']")
                alternate.find("worldbody").remove(obj)
                # Replace only static object shape and initialization, preserving table.
                obj.set("pos", f".55 0 {height}")
                geom = obj.find("geom")
                geom.set("type", "cylinder" if variant == "cylinder" else "box")
                geom.set("size", {"small": ".015 .018 .020", "wide": ".030 .026 .030", "cylinder": ".022 .025"}[variant])
                alternate.find("worldbody").append(obj)
                alternate.remove(alternate.find("keyframe"))
                add_home(alternate, False, True, height)
                write(alternate, ROOT / f"scenes/ur5_gripper_{variant}.xml")


if __name__ == "__main__":
    main()
