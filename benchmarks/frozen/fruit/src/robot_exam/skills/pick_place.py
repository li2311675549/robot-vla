"""Physical pick/place primitive; object state is used only for logging/scoring.

Caller supplies target from a calibrated camera (or an explicitly labelled
known fixture in Gate 2). No live qpos assignment or object attachment.
"""
import numpy as np
from robot_exam.control.arm import ArmController, GripperController


class PickPlace:
    def __init__(self, sim, callback=None):
        self.sim = sim
        self.arm = ArmController(sim)
        self.gripper = GripperController(sim)
        self.callback = callback
        self.rows = []
        self.stages = []
        self.rotation = np.diag([1., -1., -1.])
        self.qgoal = self.arm.reference.copy()
        self.object_body = sim.model.body("object").id

    def tick(self, stage):
        self.arm.command(self.qgoal)
        self.sim.step(self.sim.data.ctrl)
        if round(self.sim.data.time/self.sim.model.opt.timestep) % 25 == 0:
            position, rotation = self.arm.pose()
            contacts = self.gripper.contacts()
            self.rows.append({"time": self.sim.data.time, "stage": stage,
                              "tip": position.tolist(), "object": self.sim.data.xpos[self.object_body].tolist(),
                              "joint_error": float(np.max(np.abs(self.sim.data.qpos[self.arm.qadr]-self.arm.reference))),
                              "contacts": contacts, "qpos": self.sim.data.qpos[self.arm.qadr].tolist(),
                              "ctrl": self.sim.data.ctrl.tolist()})
        if self.callback:
            self.callback(self.sim, stage)

    def hold(self, seconds, stage):
        for _ in range(round(seconds/self.sim.model.opt.timestep)):
            self.tick(stage)

    def move(self, position, seconds, stage):
        start, _ = self.arm.pose()
        steps = round(seconds/self.sim.model.opt.timestep)
        for i in range(steps):
            if i % 10 == 0:
                fraction = min(1., (i+10)/steps)
                fraction = fraction*fraction*(3-2*fraction)
                goal = start + fraction*(np.asarray(position)-start)
                self.qgoal, info = self.arm.solve(goal, self.rotation, seed=self.qgoal)
            self.tick(stage)
        self.hold(.4, stage + "_settle")
        actual, rotation = self.arm.pose()
        error = float(np.linalg.norm(actual-position))
        self.stages.append({"stage": stage, "target": list(position), "actual": actual.tolist(), "position_error_m": error})
        if error > .015:
            raise RuntimeError(f"Physical arm tracking failed at {stage}: {error:.4f}m")

    def run(self, pick_position, drop_position):
        pick = np.asarray(pick_position, dtype=float)
        drop = np.asarray(drop_position, dtype=float)
        self.gripper.command(0)
        self.hold(.7, "settle")
        self.move(pick + [0, 0, .085], 2.5, "approach")
        self.move(pick, 2., "descend")
        self.gripper.command(1)
        self.hold(1.2, "close")
        if not self.gripper.bilateral():
            raise RuntimeError("No bilateral object contacts after closure")
        self.move(pick + [0, 0, .08], 2., "lift")
        self.hold(1.5, "hold")
        held = [row for row in self.rows if row["stage"] == "hold"]
        bilateral_fraction = sum(self.gripper.bilateral(row["contacts"]) for row in held)/len(held)
        min_hold_height = min(row["object"][2]-pick[2] for row in held)
        lift_height = float(self.sim.data.xpos[self.object_body, 2] - pick[2])
        if lift_height < .055 or min_hold_height < .05 or bilateral_fraction < .95:
            raise RuntimeError(f"Physical lift failed: {lift_height:.4f}m")
        self.move(drop + [0, 0, .08], 2.5, "transfer")
        self.move(drop + [0, 0, .005], 2., "lower")
        self.gripper.command(0)
        self.hold(1., "release")
        self.move(drop + [0, 0, .08], 1.5, "retreat")
        self.hold(1., "released_settle")
        object_dof = self.sim.model.jnt_dofadr[self.sim.model.joint("object_free").id]
        # Round objects can rock on the support after release. Wait on measured
        # physical velocity, with a finite timeout; no pose correction is applied.
        additional_settle = 0.
        while np.linalg.norm(self.sim.data.qvel[object_dof:object_dof+3]) > .005 and additional_settle < 4:
            self.hold(.2, "released_settle")
            additional_settle += .2
        final = self.sim.data.xpos[self.object_body].copy()
        error = float(np.linalg.norm(final[:2]-drop[:2]))
        final_speed = float(np.linalg.norm(self.sim.data.qvel[object_dof:object_dof+3]))
        success = error < .04 and abs(final[2] - drop[2]) < .01 and final_speed < .01 and not self.gripper.contacts()
        return {"status": "PASS" if success else "FAIL", "pick": pick.tolist(), "drop": drop.tolist(),
                "lift_height_m": lift_height, "final_object": final.tolist(), "placement_error_m": error,
                "min_hold_height_m": float(min_hold_height), "bilateral_contact_fraction_during_hold": bilateral_fraction,
                "final_object_speed_m_s": final_speed,
                "additional_release_settle_seconds": additional_settle,
                "stages": self.stages, "warnings": self.sim.data.warning.number.tolist(),
                "observation_mode": "known fixture target (Gate 2); object truth only for contact logging and scoring"}
