"""Bounded SE(2) dock paths: no lateral motion or physics-state modification."""
import heapq
import itertools
import math
import time

import numpy as np
from robot_exam.control.base import wrap_angle


def segment_samples(a, b):
    a, b = np.asarray(a), np.asarray(b)
    angle = wrap_angle(b[2]-a[2])
    count = max(2, int(np.linalg.norm(b[:2]-a[:2])/.015)+2, int(abs(angle)/.05)+2)
    for t in np.linspace(0, 1, count):
        yield np.r_[a[:2]+t*(b[:2]-a[:2]), wrap_angle(a[2]+t*angle)]


def connect(start, end, valid):
    """Rotate, drive forwards/backwards along the line, then rotate; exact endpoint."""
    start, end = np.asarray(start), np.asarray(end)
    delta = end[:2]-start[:2]
    if np.linalg.norm(delta)<1e-7:
        path = [start, end]
    else:
        bearing = math.atan2(delta[1], delta[0])
        options = [bearing, wrap_angle(bearing+math.pi)]
        heading = min(options, key=lambda h: abs(wrap_angle(h-start[2]))+abs(wrap_angle(end[2]-h)))
        path = [start, np.r_[start[:2], heading], np.r_[end[:2], heading], end]
    if not all(valid(p) for a,b in zip(path[:-1],path[1:]) for p in segment_samples(a,b)):
        return None
    return [np.asarray(p).tolist() for p in path]


def plan_dock(start, goal, strategy='preapproach', valid=lambda pose: True, max_nodes=12000):
    """Return collision-sampled primitives; the caller must verify real wheel tracking."""
    clock = time.monotonic()
    start,goal=np.asarray(start,dtype=float),np.asarray(goal,dtype=float)
    if start.shape!=(3,) or goal.shape!=(3,) or not np.isfinite([start,goal]).all():
        raise ValueError('Finite SE(2) poses required')
    direction=np.array([math.cos(goal[2]),math.sin(goal[2])])
    lateral=abs(float((start[:2]-goal[:2])@[-direction[1],direction[0]]))
    # Offset depends on cross-track error, not one fixed retreat for every scene.
    distances=sorted(set([max(.22,min(.55,4*lateral+.16)),.30,.45,.60]))
    candidates=[]
    for distance in distances:
        for sign in (-1,1):
            pre=np.r_[goal[:2]+sign*distance*direction,goal[2]]
            first=connect(start,pre,valid);last=connect(pre,goal,valid)
            if first is not None and last is not None:
                path=first+last[1:]
                length=sum(np.linalg.norm(np.subtract(b[:2],a[:2])) for a,b in zip(path[:-1],path[1:]))
                turns=sum(abs(wrap_angle(b[2]-a[2])) for a,b in zip(path[:-1],path[1:]))
                candidates.append((length+.22*turns,path))
    expanded=0
    if strategy=='preapproach':
        if not candidates:raise RuntimeError('No collision-free preapproach connection')
        path=min(candidates,key=lambda item:item[0])[1]
    elif strategy=='lattice':
        # Search short forward/reverse and in-place yaw primitives; exact
        # goal-axis connectors are tested near the goal. No holonomic shortcut.
        serial=itertools.count();resolution=.055;yaw_step=math.pi/12
        def key(p):return (round(p[0]/resolution),round(p[1]/resolution),round(wrap_angle(p[2])/yaw_step))
        def heuristic(p):return float(np.linalg.norm(p[:2]-goal[:2]))+.12*abs(wrap_angle(goal[2]-p[2]))
        frontier=[(heuristic(start),next(serial),0.,start,[start.tolist()])];costs={key(start):0.}
        path=None
        while frontier and expanded<max_nodes:
            _,_,cost,pose,history=heapq.heappop(frontier);expanded+=1
            if cost>costs.get(key(pose),float('inf'))+1e-9:continue
            if np.linalg.norm(pose[:2]-goal[:2])<.15:
                for distance in (.22,.35):
                    for sign in (-1,1):
                        pre=np.r_[goal[:2]+sign*distance*direction,goal[2]]
                        a=connect(pose,pre,valid);b=connect(pre,goal,valid)
                        if a is not None and b is not None:path=history+a[1:]+b[1:];break
                    if path is not None:break
                if path is not None:break
            for linear,yaw in ((.08,0),(-.08,0),(0,yaw_step),(0,-yaw_step)):
                nxt=pose.copy();nxt[:2]+=linear*np.array([math.cos(pose[2]),math.sin(pose[2])]);nxt[2]=wrap_angle(nxt[2]+yaw)
                if np.linalg.norm(nxt[:2]-start[:2])>1.3:continue
                new_cost=cost+abs(linear)*(1.15 if linear<0 else 1)+.22*abs(yaw)+.015
                if new_cost>=costs.get(key(nxt),float('inf')):continue
                if not all(valid(p) for p in segment_samples(pose,nxt)):continue
                costs[key(nxt)]=new_cost
                heapq.heappush(frontier,(new_cost+heuristic(nxt),next(serial),new_cost,nxt,history+[nxt.tolist()]))
        if path is None:raise RuntimeError(f'Local lattice exhausted {expanded} states')
    else:raise ValueError('Unknown local dock strategy')
    return {'strategy':strategy,'poses':path,'expanded_nodes':expanded,
            'planning_wall_seconds':time.monotonic()-clock,
            'kinematic_model':'skid-steer forward/reverse/in-place yaw, no lateral primitive'}
