"""Public pinhole camera calibration helpers; no scene or target information."""
import math
import numpy as np


def rotation(roll,pitch,yaw):
    ca,sa,cb,sb,cc,sc=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    return np.array([[cc*cb,cc*sb*sa-sc*ca,cc*sb*ca+sc*sa],[sc*cb,sc*sb*sa+cc*ca,sc*sb*ca-cc*sa],[-sb,cb*sa,cb*ca]])


def pixel_to_ground(observation,u,v):
    """Project normalized 0..1000 thermal pixels to local ENU ground metres."""
    p=observation['pose'];w=observation['width'];h=observation['height'];k=observation['intrinsics']
    mount=observation.get('camera_mount',dict(x_m=0,y_m=0,z_m=-.12,roll_rad=0,pitch_rad=math.pi/2,yaw_rad=0))
    body=rotation(p.get('roll_rad',0),p.get('pitch_rad',0),p.get('yaw_rad',0))
    position=np.array([p['x_m'],p['y_m'],p['z_m']])+body@np.array([mount['x_m'],mount['y_m'],mount['z_m']])
    ray=body@rotation(mount['roll_rad'],mount['pitch_rad'],mount['yaw_rad'])@np.array([1,-(u*w/1000+.5-w/2)/k['fx'],-(v*h/1000+.5-h/2)/k['fy']])
    if ray[2]>=-1e-6:raise ValueError('Camera ray does not intersect the ground')
    hit=position-ray*(position[2]/ray[2]);return float(hit[0]),float(hit[1])


def ground_to_pixel(observation,x,y):
    """Inverse projection in the same public normalized pixel convention."""
    p=observation['pose'];w=observation['width'];h=observation['height'];k=observation['intrinsics']
    mount=observation.get('camera_mount',dict(x_m=0,y_m=0,z_m=-.12,roll_rad=0,pitch_rad=math.pi/2,yaw_rad=0))
    body=rotation(p.get('roll_rad',0),p.get('pitch_rad',0),p.get('yaw_rad',0))
    position=np.array([p['x_m'],p['y_m'],p['z_m']])+body@np.array([mount['x_m'],mount['y_m'],mount['z_m']])
    camera=(body@rotation(mount['roll_rad'],mount['pitch_rad'],mount['yaw_rad'])).T@(np.array([x,y,0])-position)
    if camera[0]<=0:raise ValueError('Ground point is behind the camera')
    return float((w/2-.5-k['fx']*camera[1]/camera[0])/w*1000),float((h/2-.5-k['fy']*camera[2]/camera[0])/h*1000)
