"""Cyan vertical handle RGB-D baseline; no simulator object transforms/IDs."""
import numpy as np
from robot_exam.perception.objects import components


def locate_handle(camera,rgb,depth):
    r,g,b=np.moveaxis(rgb.astype(float),-1,0)
    mask=(g>55)&(b>55)&(g>1.7*r)&(b>1.7*r)&(g>.65*b)&(depth>.15)&(depth<2.5)
    candidates=[]
    for pixels in components(mask,depth,minimum=8):
        rows,cols=pixels.T;points=camera.unproject(np.column_stack([cols,rows]),depth[rows,cols])
        zlo,zhi=np.percentile(points[:,2],[2,98]);height=float(zhi-zlo)
        if not .045<height<.20 or not .40<np.median(points[:,2])<1.35:continue
        central=points[(points[:,2]>zlo+.25*height)&(points[:,2]<zhi-.25*height)]
        if len(central)<6:continue
        xy=central[:,:2];A=np.column_stack([2*xy,np.ones(len(xy))]);values=np.sum(xy**2,axis=1)
        fit=np.linalg.lstsq(A,values,rcond=None)[0];center=fit[:2]
        radius=float(np.sqrt(max(0,fit[2]+center@center)))
        # A cylinder prior rejects tiny components and poorly conditioned
        # background fits; it supplies no hidden handle position.
        if not .006<radius<.025:continue
        covariance=np.cov((points-points.mean(axis=0)).T)
        _,axes=np.linalg.eigh(covariance);axis=axes[:,-1]
        if axis[2]<0:axis=-axis
        candidates.append({'center':[*center.tolist(),float((zlo+zhi)/2)],
            'axis':axis.tolist(),'radius_m':radius,'pixels':len(points),
            'scope':'RGB/depth colour+vertical-cylinder fit; camera calibration only'})
    if not candidates:raise RuntimeError('No visible cyan vertical handle cylinder')
    return max(candidates,key=lambda value:value['pixels'])
