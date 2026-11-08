"""Explicit Oracle push/side-pinch skill, using real robot/object contact only."""
import numpy as np
from robot_exam.tasks.runtime import State,TaskFailure
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.control.joint_rrt import plan_joint_rrt


def pick_pushed_book(r,obj,grid):
    body=r.sim.model.body(obj['body']).id
    center=r.sim.data.xpos[body].copy()
    safe_y=float(np.clip(center[1],float(r.config.get('book_safe_bay_lower_y',-.80)),
                        float(r.config.get('book_safe_bay_upper_y',-.65))))
    r.transition(State.NAVIGATE,'oracle_book_safe_operation_bay',target=[0,safe_y,0],timeout=120)
    r.stow();r.backout(-.45);r.navigate([-.45,safe_y,0],grid);r.dock([0,safe_y,0],timeout=45)
    r.transition(State.MANIPULATE,'oracle_push_book_edge',target=obj['body'],timeout=180)
    r.gripper.command(1);r.hold(.5)
    center=r.sim.data.xpos[body].copy();initial=center.copy()
    R=r.sim.data.xmat[body].reshape(3,3);halfx=float(np.sum(np.abs(R[0,:2])*np.asarray(obj['size'][:2])/2))
    far=center[0]+halfx+.035
    r.move([far,center[1],.84],seconds=3);r.move([far,center[1],.731],seconds=2)
    push_trace=[]
    if r.config.get('book_feedback_push',False):
        # Oracle geometry expert: stop on the actual near edge, not on a
        # commanded hand displacement. The old open-loop push could move a
        # small book's centre of mass past the 0.28 m table support boundary.
        for step in range(80):
            center_now=r.sim.data.xpos[body].copy();Rnow=r.sim.data.xmat[body].reshape(3,3)
            hx=float(np.sum(np.abs(Rnow[0,:2])*np.asarray(obj['size'][:2])/2))
            near=float(center_now[0]-hx)
            push_trace.append({'time':float(r.sim.data.time),'near_edge_x_m':near,
                               'center':center_now.tolist(),'pinch_x_m':float(r.arm.pose()[0][0])})
            if near<=.252:break
            if center_now[0]<.282 or center_now[2]<.725:
                raise TaskFailure('book_fell','Feedback push lost support before grasp clearance')
            r.move([far-.005*(step+1),center[1],.731],seconds=.12)
        else:raise TaskFailure('push_timeout','Bounded edge feedback did not reach safe overhang')
    else:r.move([.230+2*halfx,center[1],.731],seconds=4)
    r.move(r.arm.pose()[0]+[0,0,.12],seconds=2);r.hold(.4)
    center=r.sim.data.xpos[body].copy()
    if center[2]<.725:raise TaskFailure('book_fell','Book lost table support during push')
    R=r.sim.data.xmat[body].reshape(3,3);halfx=float(np.sum(np.abs(R[0,:2])*np.asarray(obj['size'][:2])/2))
    grasp=center.copy();grasp[0]-=halfx-.015;pre=grasp-[.075,0,0];rotation=side_rotation()
    if grasp[0]>.27:raise TaskFailure('edge_clearance','Pushed edge still lacks lower-finger clearance')
    r.stow();r.backout(-.30);r.gripper.command(0)
    r.move([pre[0],pre[1],.90],rotation,seconds=3)
    path=None;failures=[]
    for seed in (r.qgoal,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
        try:
            q,ik=r.arm.solve(pre,rotation,seed=seed,max_iterations=500)
            path,planning=plan_joint_rrt(r.arm,q,allowed_bodies=[obj['body']]);break
        except RuntimeError as exc:failures.append(str(exc))
    if path is None:raise TaskFailure('book_joint_path',' | '.join(failures))
    for q in path:r.joint_move(q,seconds=2)
    r.move(grasp,rotation,seconds=1.5);r.gripper.command(1);r.hold(.8)
    if not r.bilateral(obj['body']):raise TaskFailure('grasp_contact','No physical bilateral book contact')
    r.move(grasp+[0,0,.10],rotation,seconds=2)
    lift=float(r.sim.data.xpos[body,2]-center[2])
    if lift<=.055 or not r.stable_grasp(obj['body'],seconds=.6):
        raise TaskFailure('gripper_slip','Book side grip did not retain a physical lift')
    return {'rotation':rotation,'lift_m':lift,'candidate_count':len(failures)+1,
            'scope':'Oracle physical push-and-side-pinch','push_displacement_m':(center-initial).tolist(),
            'joint_path_planning':planning,'ik':ik,'push_feedback_trace':push_trace}
