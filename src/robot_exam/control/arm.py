"""Original UR5 numerical IK using MuJoCo Jacobians and isolated planning data.

Only the scratch MjData is assigned joint configurations for kinematics. The
live simulation is moved exclusively by bounded actuator commands/mj_step.
"""
import mujoco
import numpy as np
import time

from robot_exam.simulation import UR5_JOINTS


def orientation_error(target, current):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (target @ current.T).ravel())
    if quat[0] < 0:
        quat *= -1
    error = np.empty(3)
    mujoco.mju_quat2Vel(error, quat, 1.0)
    return error


class ArmController:
    def __init__(self, sim):
        self.sim = sim
        self.joints = [sim.model.joint(name).id for name in UR5_JOINTS]
        self.qadr = sim.model.jnt_qposadr[self.joints]
        self.dofs = sim.model.jnt_dofadr[self.joints]
        self.actuators = [sim.model.actuator(name + "_position").id for name in UR5_JOINTS]
        self.site = sim.model.site("grip_pinch").id
        self.scratch = mujoco.MjData(sim.model)
        self.jacp = np.zeros((3, sim.model.nv))
        self.jacr = np.zeros((3, sim.model.nv))
        self.reference = sim.data.ctrl[self.actuators].copy()
        self.last_successful = []
        self.last_solve_report = {}
        self.check_self_collisions=False

    def pose(self):
        return (self.sim.data.site_xpos[self.site].copy(),
                self.sim.data.site_xmat[self.site].reshape(3, 3).copy())

    def solve(self, position, rotation, seed=None, max_iterations=300):
        started=time.perf_counter()
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Invalid end-effector target")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < .999:
            raise ValueError("Target rotation must be in SO(3)")
        self.scratch.qpos[:] = self.sim.data.qpos
        q = self.sim.data.qpos[self.qadr].copy() if seed is None else np.asarray(seed).copy()
        limits = self.sim.model.jnt_range[self.joints]
        for iteration in range(max_iterations):
            self.scratch.qpos[self.qadr] = q
            mujoco.mj_forward(self.sim.model, self.scratch)
            dp = position - self.scratch.site_xpos[self.site]
            dr = orientation_error(rotation, self.scratch.site_xmat[self.site].reshape(3, 3))
            if np.linalg.norm(dp) < .0005 and np.linalg.norm(dr) < .003:
                result={"position_error_m": float(np.linalg.norm(dp)), "orientation_error_rad": float(np.linalg.norm(dr)), "iterations": iteration,
                        "solver":"damped_least_squares","approximate_convergence":False,"elapsed_s":time.perf_counter()-started}
                self.last_successful=[q.copy(),*self.last_successful[:3]]
                self.last_solve_report={"status":"converged",**result}
                return q, result
            mujoco.mj_jacSite(self.sim.model, self.scratch, self.jacp, self.jacr, self.site)
            jacobian = np.vstack([self.jacp[:, self.dofs], .35*self.jacr[:, self.dofs]])
            error = np.r_[dp, .35*dr]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .002**2*np.eye(6), error)
            delta *= min(1, .15/max(np.linalg.norm(delta), 1e-12))
            q = np.clip(q + delta, limits[:, 0]+.005, limits[:, 1]-.005)
        # MuJoCo Jacobian finite precision can leave a few millimetres of
        # residual even when the command is well inside the Cartesian tracking
        # tolerance.  Return this explicitly bounded approximate solution;
        # the live move() still performs its independent physical endpoint
        # check and task success is unchanged.
        if np.linalg.norm(dp)<.003 and np.linalg.norm(dr)<.01:
            result={"position_error_m":float(np.linalg.norm(dp)),
                       "orientation_error_rad":float(np.linalg.norm(dr)),
                       "iterations":max_iterations,"approximate_convergence":True,
                       "elapsed_s":time.perf_counter()-started}
            self.last_successful=[q.copy(),*self.last_successful[:3]]
            self.last_solve_report={"status":"approximate","seed":None if seed is None else np.asarray(seed).tolist(),**result}
            return q, result
        report={"status":"failed","target_position":position.tolist(),"seed":None if seed is None else np.asarray(seed).tolist(),
                "position_error_m":float(np.linalg.norm(dp)),"orientation_error_rad":float(np.linalg.norm(dr)),
                "iterations":max_iterations,"joint_position_rad":self.scratch.qpos[self.qadr].tolist(),
                "joint_limits_rad":limits.tolist(),"solver":"damped_least_squares","elapsed_s":time.perf_counter()-started,
                "failure_reason":"iteration_limit_or_local_singularity"}
        self.last_solve_report=report
        raise RuntimeError(f"IK failed: position={report['position_error_m']:.6f}m rotation={report['orientation_error_rad']:.6f}rad target={position}")

    def _collision_report(self,q,allowed_bodies=()):
        model=self.sim.model;allowed=set(allowed_bodies);husky=None
        try:husky=model.body('husky').id
        except KeyError:pass
        robot=set()
        if husky is not None:
            for body in range(model.nbody):
                parent=body
                while parent>0 and parent!=husky:parent=int(model.body_parentid[parent])
                if parent==husky:robot.add(body)
        if not robot:return []
        current=self.sim.data.qpos[self.qadr].copy();risk={}
        # Check the joint-space approach as well as the endpoint, using only
        # scratch configurations. Intended contact with the grasp target is
        # explicitly permitted; table/wall/cabinet penetrations are not.
        for alpha in np.linspace(0.,1.,9):
            self.scratch.qpos[:]=self.sim.data.qpos
            self.scratch.qpos[self.qadr]=current+alpha*(np.asarray(q)-current)
            mujoco.mj_forward(model,self.scratch)
            for contact in self.scratch.contact:
                g1,g2=map(int,contact.geom);b1,b2=map(int,model.geom_bodyid[[g1,g2]])
                if b1 in robot and b2 in robot:
                    if self.check_self_collisions and alpha>=1.-1e-12 and contact.dist<-.0003:
                        bodies=[model.body(b).name or '' for b in (b1,b2)]
                        if not all(name.startswith('grip_') for name in bodies):
                            key=('self:'+('/'.join(bodies)), '/'.join(bodies))
                            risk[key]=max(risk.get(key,0.),-float(contact.dist))
                    continue
                if (b1 in robot)==(b2 in robot):continue
                other=g2 if b1 in robot else g1;other_body=b2 if b1 in robot else b1
                body_name=model.body(other_body).name or ''
                if body_name in allowed:continue
                name=model.geom(other).name or ''
                blocking=(name.startswith(('wall','door_frame','obstacle','cabinet_shell','table_leg')) or name.endswith('_top'))
                depth=-float(contact.dist)
                if blocking and depth>.002:
                    key=(name,body_name)
                    risk[key]=max(risk.get(key,0.),depth)
        return [{'other_geom':k[0],'other_body':k[1],'maximum_penetration_m':v} for k,v in risk.items()]

    def solve_multi(self,position,rotation,seed=None,seeds=None,max_iterations=500,allowed_bodies=()):
        """Try bounded analytic initial branches and return the best limit-safe IK.

        No Cartesian tolerance is changed: every candidate must pass ``solve``'s
        existing convergence criteria. Reports retain failures for diagnostics.
        """
        current=self.sim.data.qpos[self.qadr].copy()
        candidates=[]
        for value in ([seed] if seed is not None else [])+[current,self.reference,*self.last_successful,
                np.array([0.,-1.5,2.,-2.,-1.57,0.]),np.array([0.,-1.5,-2.,1.,-1.57,0.]),
                np.array([0.,-.7,1.4,-2.2,-1.57,0.])]+list(seeds or []):
            q=np.asarray(value,dtype=float)
            if q.shape==(6,) and np.isfinite(q).all() and not any(np.linalg.norm(q-x)<1e-5 for x in candidates):candidates.append(q.copy())
        limits=self.sim.model.jnt_range[self.joints]
        trials=[];solutions=[];start=time.perf_counter()
        for q0 in candidates:
            try:
                q,metrics=self.solve(position,rotation,seed=q0,max_iterations=max_iterations)
                margin=np.minimum(q-limits[:,0],limits[:,1]-q)
                collisions=self._collision_report(q,allowed_bodies)
                score=float(np.linalg.norm(q-current)+.04*np.sum(1/np.maximum(margin,.02))+
                            (1000. if metrics.get('approximate_convergence',False) else 0.)+100*len(collisions))
                trials.append({"seed":q0.tolist(),"status":"converged","collision_free":not collisions,
                               "collisions":collisions,**metrics,"joint_margin_rad":margin.tolist(),"score":score})
                solutions.append((score,q,metrics,not collisions))
            except RuntimeError:
                trials.append({"seed":q0.tolist(),**self.last_solve_report})
        safe_solutions=[x for x in solutions if x[3]]
        if safe_solutions:solutions=safe_solutions
        elif solutions:
            report={"status":"all_branches_colliding","target_position":np.asarray(position).tolist(),
                    "target_rotation":np.asarray(rotation).tolist(),"measured_joint_position_rad":current.tolist(),
                    "reference_joint_position_rad":self.reference.tolist(),"joint_limits_rad":limits.tolist(),
                    "elapsed_s":time.perf_counter()-start,"trials":trials}
            self.last_solve_report=report
            raise RuntimeError("All converged IK branches intersect configured obstacles")
        if not solutions:
            report={"status":"failed_all_branches","target_position":np.asarray(position).tolist(),
                    "target_rotation":np.asarray(rotation).tolist(),"measured_joint_position_rad":current.tolist(),
                    "reference_joint_position_rad":self.reference.tolist(),"joint_limits_rad":limits.tolist(),
                    "elapsed_s":time.perf_counter()-start,"trials":trials}
            self.last_solve_report=report
            raise RuntimeError("All IK initial branches failed: "+" | ".join(
                f"{t.get('failure_reason','failed')} pos={t.get('position_error_m',float('nan')):.6f}m rot={t.get('orientation_error_rad',float('nan')):.6f}rad"
                for t in trials))
        solutions.sort(key=lambda x:x[0])
        _,q,metrics,_=solutions[0]
        self.last_solve_report={"status":"multi_branch_converged","target_position":np.asarray(position).tolist(),
                                "measured_joint_position_rad":current.tolist(),"reference_joint_position_rad":self.reference.tolist(),
                                "selected_joint_position_rad":q.tolist(),"selected_metrics":metrics,
                                "elapsed_s":time.perf_counter()-start,"trials":trials}
        return q,metrics

    def command(self, q, max_speed=.7):
        q = np.asarray(q)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint targets")
        limits = self.sim.model.jnt_range[self.joints]
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError("Joint target exceeds UR5 limits")
        step = max_speed*self.sim.model.opt.timestep
        self.reference += np.clip(q-self.reference, -step, step)
        # Standard gravity feed-forward through the same torque-limited position
        # actuators. No qfrc_applied, base/object forces or equality attachments.
        command = self.reference.copy()
        for i, (dof, actuator) in enumerate(zip(self.dofs, self.actuators)):
            kp = self.sim.model.actuator_gainprm[actuator, 0]
            bias = self.sim.data.qfrc_bias[dof]/kp
            command[i] += np.clip(bias, -.08, .08)
        self.sim.data.ctrl[self.actuators] = np.clip(command, limits[:, 0], limits[:, 1])


class GripperController:
    def __init__(self, sim):
        self.sim = sim
        self.actuator = sim.model.actuator("grip_fingers_actuator").id

    def command(self, closure):
        """Normalized tendon position command; force limited in model."""
        if not np.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError("Gripper closure must be in [0,1]")
        self.sim.data.ctrl[self.actuator] = 255*closure

    def contacts(self):
        """Contact witnesses with the free object, not a grasp success oracle."""
        model, data = self.sim.model, self.sim.data
        obj = model.geom("test_object").id
        target_body=int(model.geom_bodyid[obj])
        result = []
        for i in range(data.ncon):
            contact = data.contact[i]
            bodies=[int(model.geom_bodyid[g]) for g in contact.geom]
            if target_body not in bodies:
                continue
            other = int(contact.geom[0] if bodies[1] == target_body else contact.geom[1])
            body = model.body(int(model.geom_bodyid[other])).name
            if body.startswith("grip_"):
                force = np.empty(6)
                mujoco.mj_contactForce(model, data, i, force)
                result.append({"body": body, "normal_force_n": float(force[0])})
        return result

    def bilateral(self, contacts=None):
        contacts = self.contacts() if contacts is None else contacts
        return all(any(c["body"].startswith("grip_" + side) and c["normal_force_n"] > .05 for c in contacts)
                   for side in ("left", "right"))
