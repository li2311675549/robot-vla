"""Explicit 7D absolute joint-goal actions and one real camera for OpenPI."""
from dataclasses import dataclass
import numpy as np

ACTION_SEMANTICS='ur5_absolute_joint_goal_v1'
JOINT_NAMES=('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3','gripper')


def validate_actions(actions,limits=None):
    a=np.asarray(actions,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=7 or not 1<=len(a)<=50 or not np.isfinite(a).all():
        raise ValueError('Expected 1..50 finite 7D actions')
    if np.any((a[:,6]<0)|(a[:,6]>1)):raise ValueError('Gripper closure outside [0,1]')
    if limits is not None and (np.any(a[:,:6]<limits[:,0]) or np.any(a[:,:6]>limits[:,1])):
        raise ValueError('UR5 joint goal exceeds limits')
    return a


def delta_actions(actions,state,inverse=False):
    """OpenPI-style chunk delta: each arm goal minus chunk-start state.

    Gripper is always absolute. Deltas are not cumulative between chunk rows.
    """
    a=np.asarray(actions,dtype=float).copy();state=np.asarray(state,dtype=float)
    if a.shape[-1]!=7 or state.shape!=(7,):raise ValueError('Expected 7D state/action')
    a[...,:6]+=state[:6] if inverse else -state[:6]
    return a


def parse_rgb(value):
    image=np.asarray(value)
    if image.ndim!=3:raise ValueError('Expected RGB image')
    if image.shape[0]==3 and image.shape[-1]!=3:image=np.moveaxis(image,0,-1)
    if image.shape[-1]!=3 or not np.isfinite(image).all():raise ValueError('Invalid RGB shape/values')
    if np.issubdtype(image.dtype,np.floating):
        if image.min()<0 or image.max()>1:raise ValueError('Float RGB must be in [0,1]')
        image=np.round(image*255)
    if image.min()<0 or image.max()>255:raise ValueError('RGB outside byte range')
    return image.astype(np.uint8)


@dataclass(frozen=True)
class OriginalUR5Inputs:
    require_wrist: bool = False

    def __call__(self,data):
        state=np.asarray(data['state'],dtype=np.float32)
        if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError('Expected measured 7D state')
        rgb=parse_rgb(data['front_rgb'])
        if self.require_wrist and 'wrist_rgb' not in data:raise ValueError('Real wrist RGB required by this model configuration')
        wrist=parse_rgb(data['wrist_rgb']) if 'wrist_rgb' in data else np.zeros_like(rgb)
        result={'state':state,'image':{'base_0_rgb':rgb,
                'left_wrist_0_rgb':wrist,'right_wrist_0_rgb':np.zeros_like(rgb)},
                'image_mask':{'base_0_rgb':np.True_,'left_wrist_0_rgb':np.bool_('wrist_rgb' in data),'right_wrist_0_rgb':np.False_},
                'prompt':str(data['prompt'])}
        if 'actions' in data:result['actions']=validate_actions(data['actions']).astype(np.float32)
        return result


@dataclass(frozen=True)
class OriginalUR5Outputs:
    def __call__(self,data):
        actions=np.asarray(data['actions'])
        if actions.ndim!=2 or actions.shape[1]<7:raise ValueError('Expected action chunk with at least 7 channels')
        return {'actions':validate_actions(actions[:,:7]).astype(np.float32)}


@dataclass(frozen=True)
class StateActionNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self,value):return (np.asarray(value)-self.mean)/np.maximum(self.std,1e-6)
    def denormalize(self,value):return np.asarray(value)*np.maximum(self.std,1e-6)+self.mean


class ActionChunkExecutor:
    """5 Hz absolute goals -> bounded 500 Hz references -> normal actuators.

    Base navigation stays outside this arm-only interface. Gravity compensation
    is applied exactly once by ArmController.command, never by the policy.
    """
    def __init__(self,sim,arm,gripper,policy_hz=5.,max_speed=.7,max_acceleration=1.4,base_hold=None):
        self.sim=sim;self.arm=arm;self.gripper=gripper
        self.period=1/float(policy_hz);self.dt=float(sim.model.opt.timestep)
        if not 0<self.dt<=self.period or max_speed<=0 or max_acceleration<=0:raise ValueError('Invalid execution rates')
        self.max_speed=float(max_speed);self.max_acceleration=float(max_acceleration)
        self.limits=sim.model.jnt_range[arm.joints].copy()
        self.reference=arm.reference.copy();self.velocity=np.zeros(6)
        self.chunk=None;self.elapsed=0.;self.trace=[]
        self.base_hold=base_hold

    def submit(self,actions,observation_time):
        if self.base_hold is not None and (not self.base_hold.active or not self.base_hold.ready):
            raise RuntimeError('Physical base parking must be stable before a VLA chunk')
        age=float(self.sim.data.time)-float(observation_time)
        if not np.isfinite(age) or age< -1e-6 or age>self.period*2:raise ValueError('Stale/future policy observation')
        # Reject the complete chunk before changing any actuator or buffer.
        a=validate_actions(actions,self.limits).copy()
        self.chunk=a;self.elapsed=0.

    def step(self):
        if self.chunk is None:raise RuntimeError('No action chunk buffered')
        index=min(self.elapsed/self.period,len(self.chunk)-1)
        lo=int(index);hi=min(lo+1,len(self.chunk)-1);blend=index-lo
        goal=(1-blend)*self.chunk[lo]+blend*self.chunk[hi]
        error=goal[:6]-self.reference
        desired=np.clip(error/self.dt,-self.max_speed,self.max_speed)
        desired=np.clip(desired,-np.sqrt(2*self.max_acceleration*np.abs(error)),
                        np.sqrt(2*self.max_acceleration*np.abs(error)))
        self.velocity+=np.clip(desired-self.velocity,-self.max_acceleration*self.dt,self.max_acceleration*self.dt)
        candidate=self.reference+self.velocity*self.dt
        if np.any(candidate<self.limits[:,0]) or np.any(candidate>self.limits[:,1]):
            self.chunk=None;raise ValueError('Limited reference would exceed joint range')
        self.reference=candidate
        self.arm.command(self.reference,max_speed=self.max_speed)
        self.gripper.command(float(goal[6]))
        self.trace.append({'time':float(self.sim.data.time),'reference':self.reference.tolist(),
                           'velocity':self.velocity.tolist(),'goal':goal.tolist()})
        try:self.sim.step(self.sim.data.ctrl)
        except RuntimeError:
            self.chunk=None;self.velocity[:]=0
            self.arm.reference=self.sim.data.qpos[self.arm.qadr].copy()
            self.arm.command(self.arm.reference)
            raise
        self.elapsed+=self.dt
        done=self.elapsed>=len(self.chunk)*self.period-1e-9
        if done:self.chunk=None
        return done

    def execute(self,actions,observation_time):
        self.submit(actions,observation_time)
        while not self.step():pass

    def execute_prefix(self,actions,observation_time,count=1):
        """Execute a short prefix while retaining the predicted interpolation.

        Dataset deployment replay submits the full horizon and replans at5Hz.
        Slicing to one row first removes the next row used for interpolation.
        Unexecuted predictions are discarded; reference/velocity remain
        continuous for the next observation. Legacy execute is unchanged.
        """
        if type(count) is not int or not 1<=count<=len(actions):
            raise ValueError('Invalid executed prefix length')
        self.submit(actions,observation_time)
        try:
            while self.elapsed<count*self.period-1e-9:self.step()
        finally:
            self.chunk=None
