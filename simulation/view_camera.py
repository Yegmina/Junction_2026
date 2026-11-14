"""Orbit math for the native overview camera; never controls the aircraft."""
import math
import numpy as np

DEFAULT = dict(mode='follow', distance_m=7., elevation_deg=25., azimuth_deg=-48., pan_x_m=0., pan_y_m=0.)
PRESETS = {
    'follow': DEFAULT,
    'wide': dict(mode='overview', distance_m=48., elevation_deg=48., azimuth_deg=-48., pan_x_m=0., pan_y_m=0.),
    'top': dict(mode='follow', distance_m=15., elevation_deg=84., azimuth_deg=-90., pan_x_m=0., pan_y_m=0.),
}


def settings(previous, updates):
    if set(updates) - (set(DEFAULT) | {'preset'}):
        raise ValueError('Unknown overview camera setting')
    value = dict(previous)
    if 'preset' in updates:
        if updates['preset'] not in PRESETS:
            raise ValueError('Unknown overview camera preset')
        value = dict(PRESETS[updates['preset']])
    value.update({k: v for k,v in updates.items() if k != 'preset'})
    if value['mode'] not in ('follow','overview'):
        raise ValueError('Unknown overview camera mode')
    for key, low, high in [('distance_m',3,85),('elevation_deg',8,88),
                           ('azimuth_deg',-3600,3600),('pan_x_m',-50,50),('pan_y_m',-50,50)]:
        number = value[key]
        if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or not low<=number<=high:
            raise ValueError('Overview camera '+key+' is outside its limits')
        value[key] = float(number)
    value['azimuth_deg'] = (value['azimuth_deg']+180)%360-180
    return value


def rotation(roll,pitch,yaw):
    a,b,c=roll,pitch,yaw;ca,sa,cb,sb,cc,sc=math.cos(a),math.sin(a),math.cos(b),math.sin(b),math.cos(c),math.sin(c)
    return np.array([[cc*cb,cc*sb*sa-sc*ca,cc*sb*ca+sc*sa],[sc*cb,sc*sb*sa+cc*ca,sc*sb*ca-cc*sa],[-sb,cb*sa,cb*ca]])


def quaternion(roll,pitch,yaw):
    cr,sr,cp,sp,cy,sy=math.cos(roll/2),math.sin(roll/2),math.cos(pitch/2),math.sin(pitch/2),math.cos(yaw/2),math.sin(yaw/2)
    return np.array([sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy])


def multiply(a,b):
    av,bv=a[:3],b[:3]
    return np.r_[a[3]*bv+b[3]*av+np.cross(av,bv),a[3]*b[3]-np.dot(av,bv)]


def camera_pose(view,drone,mount=(22.,-24.,21.,0.,.57,2.31)):
    """Return the model transform that places its offset sensor at the orbit pose."""
    following=view['mode']=='follow'
    target=np.array([drone['x_m'] if following else 0.,drone['y_m'] if following else 0.,max(.2,drone['z_m']) if following else 0.],dtype=float)
    target+=np.array([view['pan_x_m'],view['pan_y_m'],.05 if following else 0.])
    az,el=math.radians(view['azimuth_deg']),math.radians(view['elevation_deg'])
    position=target+view['distance_m']*np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
    yaw=math.atan2(target[1]-position[1],target[0]-position[0])
    desired=rotation(0,el,yaw);local=rotation(*mount[3:])
    model_position=position-(desired@local.T)@np.asarray(mount[:3])
    inverse=quaternion(*mount[3:]);inverse[:3]*=-1
    q=multiply(quaternion(0,el,yaw),inverse)
    return dict(position=dict(zip(('x','y','z'),map(float,model_position))),
                orientation=dict(zip(('x','y','z','w'),map(float,q))),
                sensor_pose=dict(x_m=float(position[0]),y_m=float(position[1]),z_m=float(position[2]),roll_rad=0.,pitch_rad=el,yaw_rad=yaw))
