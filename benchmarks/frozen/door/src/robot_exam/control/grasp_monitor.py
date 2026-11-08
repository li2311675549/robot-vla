"""Grasp measurements in the moving tool frame, without altering dynamics."""
import numpy as np


def relative_grasp_pose(tool_position, tool_rotation, object_position, object_rotation):
    """Return T_tool_object; a common world motion must not look like slip."""
    rotation = np.asarray(tool_rotation).reshape(3, 3)
    position = rotation.T @ (np.asarray(object_position) - np.asarray(tool_position))
    orientation = rotation.T @ np.asarray(object_rotation).reshape(3, 3)
    return position, orientation
