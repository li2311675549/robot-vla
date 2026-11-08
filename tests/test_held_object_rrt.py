"""Held-object sweep collisions must be rejected without altering physics."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from robot_exam.control.joint_rrt import plan_joint_rrt


def fixture():
    dummy=''.join(f'<body pos="{i+4} 0 0"><joint range="-.001 .001" limited="true"/>'
                  '<geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/></body>'
                  for i in range(5))
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
    <worldbody><geom name="obstacle" type="box" size=".04 .04 .04" pos="0 1.25 .3"/>
    <body name="tool0" pos="0 0 .3"><joint name="hinge" axis="0 0 1" range="-3.2 3.2" limited="true"/>
    <geom name="hand" type="sphere" size=".02" pos="1 0 0" mass="1"/>
    <site name="pinch" pos="1 0 0"/></body>'''+dummy+'''
    <body name="payload" pos="1.25 0 .3"><freejoint/><geom type="box" size=".05 .10 .05" mass=".2"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    arm=SimpleNamespace(sim=SimpleNamespace(model=model,data=data),qadr=np.arange(6),
                        joints=np.arange(6),site=model.site('pinch').id)
    arm.pose=lambda:(data.site_xpos[arm.site].copy(),data.site_xmat[arm.site].reshape(3,3).copy())
    return arm


@pytest.mark.parametrize('goal,error',[(np.pi/2,'RRT goal'),(np.pi,'bounded search')])
def test_held_payload_goal_and_intermediate_sweep_are_checked(goal,error):
    arm=fixture();m,d=arm.sim.model,arm.sim.data
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    before=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,before,spec)
    # The arm alone clears the obstacle. Its held object does not.
    target=np.array([goal,0.,0.,0.,0.,0.])
    path,_=plan_joint_rrt(arm,target,max_nodes=4,wall_seconds=1.)
    np.testing.assert_allclose(path[-1],target)
    with pytest.raises(RuntimeError,match=error):
        plan_joint_rrt(arm,target,max_nodes=4,wall_seconds=1.,held_body='payload')
    after=np.empty_like(before);mujoco.mj_getState(m,d,after,spec)
    np.testing.assert_array_equal(before,after)
