"""Public-image proposals: register RGB, fit background, find compact residuals.

No scene files, target positions, object counts or verifier state are used.
RGB predicts broad terrain appearance; thermal residuals propose signatures.
"""
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates,label,find_objects
from camera_geometry import rotation


def registered_rgb(observation,rgb,shape):
    h,w=shape;v,u=np.mgrid[:h,:w];pose=observation['pose']
    body=rotation(pose.get('roll_rad',0),pose.get('pitch_rad',0),pose.get('yaw_rad',0))
    def camera(calibration):
        mount=calibration['camera_mount']
        orientation=body@rotation(mount['roll_rad'],mount['pitch_rad'],mount['yaw_rad'])
        position=np.array([pose['x_m'],pose['y_m'],pose['z_m']])+body@np.array([mount['x_m'],mount['y_m'],mount['z_m']])
        return orientation,position
    rt,pt=camera(observation);rr,pr=camera(observation['rgb_calibration']);k=observation['intrinsics']
    rays=rt@np.array([np.ones(u.size),-(u.ravel()+.5-w/2)/k['fx'],-(v.ravel()+.5-h/2)/k['fy']])
    ground=pt[:,None]-rays*(pt[2]/rays[2]);coordinates=rr.T@(ground-pr[:,None])
    kr=observation['rgb_calibration']['intrinsics'];rh,rw=rgb.shape[:2]
    ur=rw/2-.5-kr['fx']*coordinates[1]/coordinates[0]
    vr=rh/2-.5-kr['fy']*coordinates[2]/coordinates[0]
    valid=(rays[2]<-1e-6)&(coordinates[0]>0)&(ur>=0)&(ur<=rw-1)&(vr>=0)&(vr<=rh-1)
    values=np.stack([map_coordinates(rgb[:,:,c].astype(float),[vr,ur],order=1,mode='nearest') for c in range(3)],axis=1)
    return values.reshape(h,w,3)/255,valid.reshape(h,w)


def proposals(observation,thermal,rgb,phase):
    thermal=np.asarray(thermal,dtype=float);h,w=thermal.shape
    color,valid=registered_rgb(observation,np.asarray(rgb),thermal.shape)
    r,g,b=np.moveaxis(color,-1,0);y,x=np.mgrid[:h,:w]
    features=np.stack([np.ones_like(r),r,g,b,r*r,g*g,b*b,r*g,r*b,g*b,x/w,y/h],axis=-1)
    fit=valid&(thermal>8)&(thermal<247);sample=fit.copy();sample[:]=False;sample[::4,::4]=fit[::4,::4]
    design=features[sample];target=thermal[sample]
    if len(target)<100:return []
    keep=np.ones(len(target),dtype=bool)
    for _ in range(4):
        coefficients=np.linalg.lstsq(design[keep],target[keep],rcond=1e-5)[0]
        error=target-design@coefficients;center=np.median(error[keep]);scale=max(1.,1.4826*np.median(abs(error[keep]-center)))
        keep=abs(error-center)<3*scale
        if keep.sum()<100:return []
    predicted=features@coefficients
    diameter=observation['intrinsics']['fx']*.24/max(1.,observation['pose']['z_m']-.12)
    sign=1 if phase in ('warming','late_warming') else -1
    residual=gaussian_filter(sign*(thermal-predicted),1.1)
    residual-=gaussian_filter(residual,max(5.,3*diameter))
    central=residual[valid];center=float(np.median(central));scale=max(.7,float(1.4826*np.median(abs(central-center))))
    signal=(residual-center)/scale
    components,_=label((signal>4.5)&valid);found=[];expected=np.pi*(diameter/2)**2
    for index,slices in enumerate(find_objects(components),1):
        if slices is None:continue
        yy,xx=slices;mask=components[yy,xx]==index;area=int(mask.sum());height=yy.stop-yy.start;width=xx.stop-xx.start
        if not max(8,.12*expected)<=area<=8*expected or min(width,height)<3 or max(width,height)>3*min(width,height) or area/(width*height)<.3:continue
        vs,us=np.nonzero(mask);weights=signal[yy,xx][mask]
        u=float(np.average(us+xx.start,weights=weights));v=float(np.average(vs+yy.start,weights=weights))
        found.append((u,v,max(width,height)/2+2,float(np.quantile(weights,.9))))
    return sorted(found,key=lambda p:p[3],reverse=True)[:12]
