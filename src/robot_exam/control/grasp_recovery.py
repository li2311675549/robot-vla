"""Conservative stopped-grasp revalidation; never opens a still-stable grip."""
import numpy as np


def stop_and_revalidate_grasp(runtime,body,minimum_height_m,seconds=1.2):
    r=runtime;model,data=r.sim.model,r.sim.data;body_id=model.body(body).id
    joint=int(model.body_jntadr[body_id]);dof=int(model.jnt_dofadr[joint])
    r.base.command(0,0);r.hold(.5)
    def relative():
        tip,rotation=r.arm.pose()
        return rotation.T@(data.xpos[body_id]-tip),rotation.T@data.xmat[body_id].reshape(3,3)
    initial,initial_rotation=relative();samples=[]
    for _ in range(max(1,round(seconds/.02))):
        r.hold(.02);point,rotation=relative()
        angle=float(np.arccos(np.clip((np.trace(initial_rotation.T@rotation)-1)/2,-1,1)))
        samples.append(dict(time=float(data.time),bilateral=bool(r.bilateral(body)),
                            position_change_m=float(np.linalg.norm(point-initial)),rotation_change_rad=angle,
                            gripper_distance_m=float(np.linalg.norm(point)),height_m=float(data.xpos[body_id,2])))
    base_velocity=data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
    report=dict(body=body,seconds=seconds,minimum_height_m=minimum_height_m,
                bilateral_fraction=float(np.mean([x['bilateral'] for x in samples])),
                max_relative_translation_m=max(x['position_change_m'] for x in samples),
                max_relative_rotation_rad=max(x['rotation_change_rad'] for x in samples),
                max_gripper_distance_m=max(x['gripper_distance_m'] for x in samples),
                minimum_measured_height_m=min(x['height_m'] for x in samples),
                final_object_angular_speed_rad_s=float(np.linalg.norm(data.qvel[dof+3:dof+6])),
                final_base_linear_speed_m_s=float(np.linalg.norm(base_velocity[:2])),
                final_base_yaw_rate_rad_s=float(abs(base_velocity[5])),
                release_or_object_pose_write=False,samples=samples)
    report['success']=bool(report['bilateral_fraction']>=.9 and report['max_relative_translation_m']<.01
        and report['max_relative_rotation_rad']<.10 and report['max_gripper_distance_m']<.075
        and report['minimum_measured_height_m']>=minimum_height_m
        and report['final_object_angular_speed_rad_s']<.20
        and report['final_base_linear_speed_m_s']<.02 and report['final_base_yaw_rate_rad_s']<.03)
    return report
