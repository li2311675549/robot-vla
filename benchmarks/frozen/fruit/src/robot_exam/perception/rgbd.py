"""Ideal registered virtual D435i RGB-D, with explicit camera calibration.

No stereo/IR/rolling shutter/noise/IMU emulation. A color fixture detector
returns geometry from RGB/depth, never object body transforms or segmentation IDs.
"""
import json
import math
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np


class RGBDCamera:
    def __init__(self, sim, width=640, height=480, camera_name='d435i_rgbd'):
        self.sim = sim
        self.width, self.height = width, height
        self.id = sim.model.camera(camera_name).id
        self.name = camera_name
        self.renderer = mujoco.Renderer(sim.model, height=height, width=width)
        fy = height/(2*math.tan(math.radians(sim.model.cam_fovy[self.id])/2))
        self.K = np.array([[fy, 0, (width-1)/2], [0, fy, (height-1)/2], [0, 0, 1.]])

    def close(self):
        self.renderer.close()

    def capture(self):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.sim.data, camera=self.id)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.sim.data, camera=self.id)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        if not np.isfinite(depth).all() or depth.min() <= 0:
            raise RuntimeError("Invalid metric depth")
        return rgb, depth

    def capture_rgb(self):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.sim.data,camera=self.id)
        return self.renderer.render().copy()

    def world_from_optical(self):
        # Optical frame uses X right, Y down, Z forward.
        transform = np.eye(4)
        transform[:3, :3] = self.sim.data.cam_xmat[self.id].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = self.sim.data.cam_xpos[self.id]
        return transform

    def unproject(self, uv, depth):
        uv = np.asarray(uv)
        d = np.asarray(depth)
        optical = np.column_stack([(uv[:, 0]-self.K[0, 2])*d/self.K[0, 0],
                                   (uv[:, 1]-self.K[1, 2])*d/self.K[1, 1], d])
        transform = self.world_from_optical()
        return optical @ transform[:3, :3].T + transform[:3, 3]

    def detect_red_fixture(self, rgb, depth):
        # This intentionally limited detector is a declared engineering baseline.
        red = (rgb[:, :, 0] > 90) & (rgb[:, :, 0] > 1.8*rgb[:, :, 1]) & (rgb[:, :, 0] > 1.8*rgb[:, :, 2])
        red &= (depth > .1) & (depth < 2.5)
        rows, cols = np.nonzero(red)
        if len(rows) < 20:
            raise RuntimeError(f"Red fixture not visible: {len(rows)} pixels")
        points = self.unproject(np.column_stack([cols, rows]), depth[rows, cols])
        # Known support plane and upright box/cylinder, not object world truth.
        # Top-surface footprint gives XY; physical tabletop is z=.725m.
        top_z = float(np.percentile(points[:, 2], 95))
        top = points[points[:, 2] > top_z-.003]
        center_xy = (np.percentile(top[:, :2], 5, axis=0) + np.percentile(top[:, :2], 95, axis=0))/2
        center_z = (.725 + top_z)/2
        return {"center": [*center_xy.tolist(), center_z], "top_z": top_z,
                "pixels": len(rows), "top_pixels": len(top),
                "assumptions": "red upright fixture on known horizontal table z=.725; top visible"}

    def save(self, output, rgb, depth):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output / "rgb.png", rgb)
        np.save(output / "depth_m.npy", depth)
        near, far = np.percentile(depth, [2, 95])
        preview = np.clip((depth-near)/max(float(far-near), 1e-6), 0, 1)
        iio.imwrite(output / "depth_preview.png", (preview*255).astype(np.uint8))
        transform = self.world_from_optical()
        body = self.sim.model.body("husky").id if "husky" in [self.sim.model.body(i).name for i in range(self.sim.model.nbody)] else 0
        world_base = np.eye(4)
        world_base[:3, :3] = self.sim.data.xmat[body].reshape(3, 3)
        world_base[:3, 3] = self.sim.data.xpos[body]
        parameters = {"width": self.width, "height": self.height, "K": self.K.tolist(),
                      "T_world_optical": transform.tolist(),
                      "T_base_optical": (np.linalg.inv(world_base) @ transform).tolist(),
                      "depth_units": "metres; optical Z", "frame": "X right, Y down, Z forward",
                      "simulated": ["ideal registered RGB", "metric depth", "fixed mounting transform"],
                      "not_simulated": ["stereo disparity", "IR projector", "noise", "rolling shutter", "IMU", "hardware range/dropouts"]}
        (output / "camera.json").write_text(json.dumps(parameters, indent=2))
        return parameters
