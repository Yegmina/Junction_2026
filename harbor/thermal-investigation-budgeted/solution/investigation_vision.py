"""Public radial-profile support for prioritizing inspection, not a probability."""
import numpy as np
from vision import proposals


def radial_support(pixels,u,v,radius,phase):
    h,w=pixels.shape;half=max(6,min(50,int(radius*1.7)))
    x0=max(0,int(u)-half);x1=min(w,int(u)+half+1);y0=max(0,int(v)-half);y1=min(h,int(v)+half+1)
    yy,xx=np.mgrid[y0:y1,x0:x1];x=(xx-u)/half;y=(yy-v)/half;mask=x*x+y*y<1
    values=pixels[y0:y1,x0:x1][mask].astype(float)
    plane=np.stack([np.ones(mask.sum()),x[mask],y[mask]],axis=-1)
    residual=values-plane@np.linalg.lstsq(plane,values,rcond=None)[0];baseline=float(residual@residual)
    best=0.;sign=1 if phase in ('warming','late_warming') else -1
    for sigma in [.2,.35,.5,.7,1.]:
        shape=np.exp(-((xx-u)**2+(yy-v)**2)/(2*(sigma*radius)**2))[mask]
        shape-=plane@np.linalg.lstsq(plane,shape,rcond=None)[0]
        amplitude=float(residual@shape/max(1e-12,shape@shape))
        if amplitude*sign<=0:continue
        best=max(best,1-float(np.sum((residual-amplitude*shape)**2))/max(1e-9,baseline))
    # A clipped thermal core does not retain a Gaussian profile. Keep that
    # uncertainty explicit rather than treating this shape fit as calibrated.
    core=(xx-u)**2+(yy-v)**2<max(2.,radius*.4)**2
    center=pixels[y0:y1,x0:x1][core]
    if center.size and np.mean((center<=1)|(center>=254))>.1:best=max(best,.75)
    return float(min(1.,max(0.,best)))


def propose(observation,thermal,rgb,phase):
    return [(u,v,r,strength,radial_support(thermal,u,v,r,phase))
            for u,v,r,strength in proposals(observation,thermal,rgb,phase)]
