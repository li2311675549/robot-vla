"""Small, explicit simulation boundary. No pose-setting during execution."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENES = {
    "minimal": ROOT / "scenes/minimal.xml",
    "ur5": ROOT / "assets/ur5/scene.xml",
    "robotiq": ROOT / "assets/robotiq_2f85/smoke.xml",
}
UR5_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


class Simulation:
    def __init__(self, scene="minimal"):
        path = SCENES.get(str(scene), Path(scene))
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.step_controllers = []
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.check()

    def joint_positions(self):
        return {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i):
                float(self.data.qpos[self.model.jnt_qposadr[i]])
                for i in range(self.model.njnt)
                if self.model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)}

    def step(self, ctrl, count=1):
        ctrl = np.asarray(ctrl, dtype=float)
        if ctrl.shape != (self.model.nu,) or not np.isfinite(ctrl).all():
            raise ValueError("Control must have nu finite entries")
        limited = self.model.actuator_ctrllimited.astype(bool)
        ranges = self.model.actuator_ctrlrange
        if np.any(ctrl[limited] < ranges[limited, 0]) or np.any(ctrl[limited] > ranges[limited, 1]):
            raise ValueError("Control outside actuator limits")
        if count < 1:
            raise ValueError("Step count must be positive")
        self.data.ctrl[:] = ctrl
        for _ in range(count):
            self.prepare_controls()
            mujoco.mj_step(self.model, self.data)
            self.check()
            for controller in tuple(self.step_controllers):controller.after_step()

    def add_step_controller(self, controller):
        if controller not in self.step_controllers:self.step_controllers.append(controller)

    def remove_step_controller(self, controller):
        if controller in self.step_controllers:self.step_controllers.remove(controller)

    def prepare_controls(self):
        """Idempotent pre-step controls; recorders call this before auditing ctrl."""
        for controller in tuple(self.step_controllers):controller.before_step()

    def check(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulation state")
        warnings = {mujoco.mjtWarning(i).name: int(w.number)
                    for i, w in enumerate(self.data.warning) if w.number}
        if warnings:
            raise RuntimeError(f"MuJoCo warnings: {warnings}")


def camera_for(scene):
    if scene == "minimal":
        return "overview"
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.45] if scene == "ur5" else [0, 0, 0.12]
    camera.distance = 2.2 if scene == "ur5" else 0.55
    camera.azimuth = 135
    camera.elevation = -25
    return camera


def targets(sim, scene):
    target = sim.data.ctrl.copy()
    if scene == "minimal":
        target[0] = 0.7
    elif scene == "ur5":
        target[0] += 0.45
        target[5] += 0.30
    elif scene == "robotiq":
        target[:] = 180
    return target
