"""RGB-D color components, geometric/PCA estimates; no simulator object access."""
import numpy as np
import mujoco


def components(mask, depth, minimum=8):
    seen=np.zeros(mask.shape,dtype=bool)
    height,width=mask.shape
    result=[]
    for row,col in zip(*np.nonzero(mask)):
        if seen[row,col]:continue
        queue=[(row,col)];seen[row,col]=True;group=[]
        while queue:
            y,x=queue.pop();group.append((y,x))
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=yy<height and 0<=xx<width and mask[yy,xx] and not seen[yy,xx] and abs(float(depth[yy,xx]-depth[y,x]))<.035:
                    seen[yy,xx]=True;queue.append((yy,xx))
        if len(group)>=minimum:result.append(np.asarray(group))
    return result


def masks(rgb, kind="fruit"):
    r,g,b=np.moveaxis(rgb.astype(float),-1,0)
    if kind=="fruit":
        return {"apple":(r>65)&(g<r*.23)&(b<r*.22),
                "banana":(r>65)&(g>r*.60)&(g<r*1.2)&(b<r*.3),
                "orange":(r>65)&(g>r*.24)&(g<r*.58)&(b<r*.25)}
    if kind=="book":
        return {"red":(r>60)&(r>g*2)&(r>b*2),"green":(g>50)&(g>r*2)&(g>b*2),"blue":(b>60)&(b>r*2)&(b>g*1.5)}
    return {"apple":(g>40)&(b>40)&(g>r*2)&(b>r*2)&(g>b*.65),
            "banana":(r>45)&(b>50)&(r>g*2)&(b>g*2),
            "orange":(b>55)&(b>r*3.2)&(b>g*1.8)}


def split_round_component(pixels):
    """Separate touching round silhouettes using observed distance peaks.

    Erosion can leave an entire large lobe as one seed and a tiny fragment of
    its neighbour as another. Radius-normalized Voronoi regions instead use
    the interior distance maxima as centres. Only image pixels enter here;
    body IDs, true positions and the number of remaining objects are absent.
    """
    if len(pixels)<20:return [pixels]
    origin=pixels.min(axis=0)-1;shape=pixels.max(axis=0)-origin+2
    mask=np.zeros(tuple(shape),dtype=bool);local=pixels-origin;mask[local[:,0],local[:,1]]=True
    background=np.argwhere(~mask);distance=np.empty(len(local))
    # Exact pixel distance in bounded blocks; avoids a new SciPy dependency
    # and an unbounded foreground-by-background temporary array.
    for begin in range(0,len(local),128):
        delta=local[begin:begin+128,None,:]-background[None,:,:]
        distance[begin:begin+128]=np.sqrt(np.min(np.sum(delta*delta,axis=2),axis=1))
    peaks=[]
    for index in np.argsort(-distance,kind='stable'):
        radius=float(distance[index]);center=local[index]
        if radius<max(3.,.70*float(distance.max())):break
        if any(np.linalg.norm(center-other)<.80*(radius+r) for other,r in peaks):continue
        peaks.append((center.copy(),radius))
        if len(peaks)>4:return [pixels]
    if len(peaks)<2:return [pixels]
    costs=np.stack([np.sum((local-center)**2,axis=1)/(radius*radius) for center,radius in peaks],axis=1)
    labels=np.argmin(costs,axis=1)
    groups=[pixels[labels==label] for label in range(len(peaks))]
    return groups if all(len(g)>=max(10,len(pixels)*.12) for g in groups) else [pixels]


def observe_objects(camera, rgb, depth, table_height=.725, workspace=(.34,.62,-.32,.32), kind="fruit",split_round_instances=False):
    results=[]
    for category,mask in masks(rgb,kind).items():
        pixel_groups=components(mask & (depth>.1)&(depth<2.5),depth,minimum=10)
        if split_round_instances and kind=='fruit' and category in ('apple','orange'):
            split=[]
            for pixels in pixel_groups:
                y,x=pixels.T;world=camera.unproject(np.column_stack([x,y]),depth[y,x])
                if len(world)>20 and np.max(np.ptp(world[:,:2],axis=0))>.075:
                    split.extend(split_round_component(pixels))
                else:split.append(pixels)
            pixel_groups=split
        for pixels in pixel_groups:
            y,x=pixels.T
            points=camera.unproject(np.column_stack([x,y]),depth[y,x])
            keep=(points[:,2]>table_height+.002)&(points[:,0]>workspace[0])&(points[:,0]<workspace[1])&(points[:,1]>workspace[2])&(points[:,1]<workspace[3])
            points=points[keep]
            if len(points)<10:continue
            top_z=float(np.percentile(points[:,2],98))
            top=points[points[:,2]>top_z-(.003 if kind=="book" else .005)]
            if len(top)<6:continue
            xy=(np.percentile(top[:,:2],3,axis=0)+np.percentile(top[:,:2],97,axis=0))/2
            covariance=np.cov(top[:,:2].T)
            vals,vecs=np.linalg.eigh(covariance)
            major=vecs[:,np.argmax(vals)]
            yaw=float(np.arctan2(major[1],major[0]))
            axes=np.column_stack([major,[-major[1],major[0]]])
            local=top[:,:2]@axes
            size=np.percentile(local,98,axis=0)-np.percentile(local,2,axis=0)
            results.append({"category":category,"center":[float(xy[0]),float(xy[1]),(table_height+top_z)/2],
                            "top_z":top_z,"yaw":yaw,"length":float(max(size)),"width":float(min(size)),
                            "thickness":float(max(.008,top_z-table_height)),
                            "area":float(np.prod(size)),"pixels":len(points),"pca_ratio":float(vals[-1]/max(vals[0],1e-9))})
    return sorted(results,key=lambda x:(x["center"][0],abs(x["center"][1])))


def observe_bins(camera,rgb,depth):
    results={}
    for category,mask in masks(rgb,"bin").items():
        y,x=np.nonzero(mask & (depth>.1)&(depth<2.5))
        if len(y)<10:continue
        points=camera.unproject(np.column_stack([x,y]),depth[y,x])
        points=points[(points[:,2]>.765)&(points[:,2]<.84)]
        if len(points)<10:continue
        center=np.median(points[:,:2],axis=0)-[.106,0]
        results[category]={"center":[*center.tolist(),.55],"pixels":len(points),"source":"visible colored/text label; known label-to-bin transform"}
    return results


def grasp_candidates(arm, detection, rotation=None,solve_ik=True):
    center=np.asarray(detection["center"]).copy()
    allowed=[detection['target_body']] if detection.get('target_body') else []
    # Pinch site is at the distal edge of the silicone pads. For rounded
    # fruit, grip below its centre so the finite pad patch spans the equator;
    # keep the fingertips above the calibrated tabletop. Thin books retain
    # their original approach height.
    if detection.get('category')=='apple':
        center[2]-=min(.010,max(0.,(center[2]-.725)*.6))
    if detection.get('category') in ('apple','banana','orange'):
        # Explicit planner offset from the observed center, not a fabricated
        # perception measurement or any change to the object's live pose.
        center[2]+=float(detection.get('grasp_z_offset_m',0.))
    book=detection.get('category') in ('red','green','blue')
    if book and detection.get('thickness',.04)<.078 and detection.get('width',0)>.078:
        # Approach horizontally from either principal edge. Choosing only
        # the long-axis edge can send the wrist across the tabletop when the
        # robot is physically beside the short edge.
        yaw=detection.get('yaw',0);major=np.array([np.cos(yaw),np.sin(yaw),0.])
        minor=np.array([-major[1],major[0],0.])
        axes=[(major,detection.get('length',0),'long'),
              (minor,detection.get('width',0),'short')]
        try:base=arm.sim.data.xpos[arm.sim.model.body('arm_pedestal').id]
        except KeyError:base=arm.pose()[0]
        closing=np.array([0.,0.,1.]);thickness=detection['thickness'];results=[]
        for candidate_axis,extent,axis_name in axes:
            if extent<=.04:continue
            axis=candidate_axis.copy()
            # PCA axes are unoriented; point the approach from the robot
            # toward the object, then target the near physical edge.
            if np.dot(axis[:2],center[:2]-base[:2])<0:axis=-axis
            lateral=np.cross(closing,axis)
            R=np.column_stack([lateral,closing,axis]) if rotation is None else rotation
            edge=max(.012,min(.025,extent*.15))
            p=center-axis*(extent/2-edge);p[2]=.725+thickness/2
            for shift in (0.,-.004,.004):
                point=p+lateral*shift;pre=point-axis*.075
                if solve_ik:
                    try:
                        q,metrics=arm.solve_multi(point,R,max_iterations=400,allowed_bodies=allowed)
                        pre_q,_=arm.solve_multi(pre,R,seed=q,max_iterations=400,allowed_bodies=allowed)
                    except RuntimeError:continue
                    arm.scratch.qpos[arm.qadr]=pre_q;mujoco.mj_forward(arm.sim.model,arm.scratch)
                else:q=None;metrics={'position_error_m':0.};pre_q=None
                risk=0.
                if pre_q is not None:
                    for contact in arm.scratch.contact:
                        names=[arm.sim.model.geom(int(g)).name or '' for g in contact.geom]
                        if any(n.startswith(('wall','door_frame','cabinet_shell','table_leg')) or n.endswith('_top') for n in names):
                            risk+=max(0.,-float(contact.dist))
                score=(np.linalg.norm(pre-arm.pose()[0]) if pre_q is None else np.linalg.norm(pre_q-arm.reference))+10*thickness+50*metrics['position_error_m']+100*risk
                results.append({'position':point,'pregrasp':pre,'rotation':R,'q':q,'score':float(score),
                                'closing_width':thickness,'grasp_type':'thin_book_edge','approach_axis':axis_name,
                                'static_pregrasp_penetration_m':risk})
        return sorted(results,key=lambda x:x['score'])
    elongated=detection.get('category')=='banana' or (detection.get('pca_ratio',1)>4 and detection.get('length',0)>.08)
    yaw=detection.get("yaw",0) if elongated else 0
    results=[]
    if detection.get('category')=='banana':
        axis=np.array([np.cos(yaw),np.sin(yaw),0.]);normal=np.array([-axis[1],axis[0],0.])
        offsets=([0,0,0],axis*.010,-axis*.010,normal*.006,-normal*.006)
    else:offsets=([0,0,0],[.003,0,0],[-.003,0,0])
    for angle in (yaw,yaw+np.pi,yaw+np.pi/2,yaw-np.pi/2):
        c,s=np.cos(angle),np.sin(angle)
        R=np.array([[c,s,0],[s,-c,0],[0,0,-1.]]) if rotation is None else rotation
        closing_width=detection.get("width",.04) if angle in (yaw,yaw+np.pi) else detection.get("length",.04)
        if closing_width>.078:continue
        for offset in offsets:
            p=center+offset;pre_position=p+[0,0,.085]
            if solve_ik:
                try:
                    q,metrics=arm.solve_multi(p,R,max_iterations=400,allowed_bodies=allowed)
                    pre_q,pm=arm.solve_multi(pre_position,R,seed=q,max_iterations=400,allowed_bodies=allowed)
                except RuntimeError:continue
            else:q=None;pre_q=None;metrics={'position_error_m':0.};pm={}
            # Only known static geometry and robot self geometry are used for
            # this pregrasp check; hidden object poses do not score candidates.
            if q is not None:arm.scratch.qpos[arm.qadr]=pre_q
            if q is not None:mujoco.mj_forward(arm.sim.model,arm.scratch)
            risk=0.
            if pre_q is not None:
                for contact in arm.scratch.contact:
                    names=[arm.sim.model.geom(int(g)).name or '' for g in contact.geom]
                    if any(n.startswith(('wall','door_frame','cabinet_shell','table_leg')) or n.endswith('_top') for n in names):
                        risk+=max(0.,-float(contact.dist))
            score=(np.linalg.norm(pre_position-arm.pose()[0]) if pre_q is None else np.linalg.norm(pre_q-arm.reference))+10*closing_width+50*metrics["position_error_m"]+100*risk+(3 if elongated and angle not in (yaw,yaw+np.pi) else 0)
            clutter_penetration=0.
            if q is not None and detection.get('oracle_avoid_other_fruit_contacts',False):
                # Declared Oracle teacher only; these body IDs/poses/contact
                # queries never enter RGB/state learning-policy observations.
                model=arm.sim.model;tool=model.body('tool0').id;tool_bodies=set()
                for body_id in range(model.nbody):
                    parent=body_id
                    while parent>0 and parent!=tool:parent=int(model.body_parentid[parent])
                    if parent==tool:tool_bodies.add(body_id)
                for blend in (0.,.5,1.):
                    arm.scratch.qpos[:]=arm.sim.data.qpos
                    arm.scratch.qpos[arm.qadr]=(1-blend)*pre_q+blend*q
                    mujoco.mj_forward(model,arm.scratch)
                    for contact in arm.scratch.contact:
                        a,b=map(int,model.geom_bodyid[np.asarray(contact.geom,dtype=int)])
                        if (a in tool_bodies)==(b in tool_bodies):continue
                        other=b if a in tool_bodies else a;name=model.body(other).name or ''
                        if name.startswith('fruit_') and name!=detection.get('target_body'):
                            clutter_penetration=max(clutter_penetration,max(0.,-float(contact.dist)))
                score+=1000*clutter_penetration
            results.append({"position":p,"pregrasp":pre_position,"rotation":R,"q":q,"score":float(score),"closing_width":closing_width,'grasp_type':'topdown','static_pregrasp_penetration_m':risk})
            if detection.get('oracle_avoid_other_fruit_contacts',False):
                results[-1]['oracle_other_fruit_penetration_m']=clutter_penetration
    return sorted(results,key=lambda x:x["score"])
