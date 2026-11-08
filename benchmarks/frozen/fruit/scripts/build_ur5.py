#!/usr/bin/env python3
"""Original UR5: upstream xacro -> collision URDF -> MuJoCo MJCF.

No ROS installation needed. Upstream files are kept unchanged; package lookup is
resolved in an outputs/build copy. Uses original UR5 kinematics/inertias/limits.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

from _bootstrap import ROOT
import mujoco
import xacro
from robot_exam.simulation import UR5_JOINTS


def write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def main():
    source = ROOT / "third_party/universal_robot/ur_description"
    build = ROOT / "outputs/build/ur_description"
    build.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix not in (".xacro", ".yaml"):
            continue
        dest = build / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(path.read_text().replace("$(find ur_description)", str(build)))
    document = xacro.process_file(str(build / "urdf/ur5.xacro"))
    urdf = ET.fromstring(document.toxml())
    # MuJoCo doesn't read COLLADA. Collision meshes remain genuine UR5 meshes
    # and serve as both physical and visual geometry for this stage.
    for link in urdf.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    meshdir = source / "meshes/ur5/collision"
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", str(meshdir / Path(mesh.get("filename")).name))
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="true", fusestatic="false", balanceinertia="true", meshdir=str(meshdir))
    urdf_path = build / "ur5_collision.urdf"
    write_xml(urdf, urdf_path)
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    assert model.njnt == 6
    output = ROOT / "assets/ur5"
    output.mkdir(parents=True, exist_ok=True)
    generated = output / "ur5.xml"
    mujoco.mj_saveLastXML(str(generated), model)
    mjcf = ET.parse(generated).getroot()
    mjcf.set("model", "original_ur5_ros_industrial")
    compiler = mjcf.find("compiler")
    compiler.set("meshdir", "../../third_party/universal_robot/ur_description/meshes/ur5/collision")
    for mesh in mjcf.findall(".//mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    ET.SubElement(mjcf, "option", timestep="0.002", integrator="implicitfast")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", damping="2", armature="0.05")
    ET.SubElement(default, "geom", rgba="0.42 0.62 0.75 1", friction="0.8 0.01 0.001")
    contact = ET.SubElement(mjcf, "contact")
    # The base is welded to world: MuJoCo's usual parent collision filtering
    # does not suppress its designed overlap with the first rotating link.
    ET.SubElement(contact, "exclude", body1="base_link_inertia", body2="shoulder_link")
    actuators = ET.SubElement(mjcf, "actuator")
    urdf_joints = {j.get("name"): j for j in urdf.findall("joint")}
    for i, name in enumerate(UR5_JOINTS):
        limit = urdf_joints[name].find("limit")
        effort = limit.get("effort")
        ET.SubElement(actuators, "position", name=name + "_position", joint=name,
                      kp="2000" if i < 3 else "300", kv="120" if i < 3 else "20",
                      ctrlrange=f"{limit.get('lower')} {limit.get('upper')}",
                      forcerange=f"-{effort} {effort}")
    home = "0 -1.57079632679 1.57079632679 -1.57079632679 -1.57079632679 0"
    keys = ET.SubElement(mjcf, "keyframe")
    ET.SubElement(keys, "key", name="home", qpos=home, ctrl=home)
    write_xml(mjcf, generated)
    scene = ET.fromstring('''<mujoco model="original_ur5_smoke">
      <include file="ur5.xml"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody>
        <light pos="1 -1 3"/>
        <light pos="-1 1 2"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.24 0.28 1"/>
      </worldbody>
    </mujoco>''')
    write_xml(scene, output / "scene.xml")
    check = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    assert check.nq == check.nv == check.nu == 6
    print(f"PASS original UR5: nq={check.nq}, nu={check.nu}, {generated}")


if __name__ == "__main__":
    main()
