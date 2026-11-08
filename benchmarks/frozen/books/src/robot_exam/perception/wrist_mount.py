"""Versioned small virtual wrist RGB camera, with mass and collision housing."""
import xml.etree.ElementTree as ET
import numpy as np


def add_wrist_camera(root):
    tool=root.find(".//body[@name='tool0']")
    if tool is None:raise ValueError('Original UR5 tool0 body required')
    if root.find(".//camera[@name='wrist_rgb']") is not None:raise ValueError('Duplicate wrist camera')
    mount=ET.SubElement(tool,'body',name='wrist_rgb_mount',pos='0 -.065 .07')
    ET.SubElement(mount,'geom',name='wrist_rgb_housing',type='box',size='.012 .016 .009',
                  mass='.025',rgba='.16 .18 .20 1')
    ET.SubElement(tool,'geom',name='wrist_rgb_bracket',type='capsule',
                  fromto='0 -.025 .025 0 -.065 .07',size='.004',mass='.008',rgba='.3 .3 .32 1')
    forward=np.array([0,.065,.13]);forward/=np.linalg.norm(forward)
    right=np.array([1.,0.,0.]);up=np.cross(-forward,right)
    axes=' '.join(map(str,np.r_[right,up]))
    ET.SubElement(mount,'camera',name='wrist_rgb',xyaxes=axes,fovy='85')
    return {'sensor_schema':'front_wrist_rgb_v1','wrist_camera':'wrist_rgb',
            'mount_version':'tool_y_side_v2','parent':'tool0','position_tool_m':[0,-.065,.07],'housing_mass_kg':.025,
            'bracket_mass_kg':.008,'scope':'ideal wrist RGB camera, not a complete D435i sensor model'}
