"""Perspective thermal sensor over a composed DuMux surface, using Gazebo pose."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter

def rotation(roll,pitch,yaw):
    a,b,c=roll,pitch,yaw;ca,sa,cb,sb,cc,sc=math.cos(a),math.sin(a),math.cos(b),math.sin(b),math.cos(c),math.sin(c)
    return np.array([[cc*cb,cc*sb*sa-sc*ca,cc*sb*ca+sc*sa],[sc*cb,sc*sb*sa+cc*ca,sc*sb*ca-cc*sa],[-sb,cb*sa,cb*ca]])

class PhysicsCamera:
    def __init__(self,folder,manifest='scene.json'):
        self.folder=Path(folder);self.metadata=json.loads((self.folder/manifest).read_text());path=Path(self.metadata.get('thermal_field_path',self.folder/'temperature_k.npy'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.metadata['thermal_field_sha256']:raise ValueError('Thermal field checksum mismatch')
        self.field=np.load(path,mmap_mode='r',allow_pickle=False);self.seed=20261114;self.ray_cache={}
    def ground_points(self,pose,width,height,hfov=1.25):
        fx=width/(2*math.tan(hfov/2));key=(width,height,hfov)
        if key not in self.ray_cache:
            u,v=np.meshgrid((np.arange(width)+.5-width/2)/fx,(np.arange(height)+.5-height/2)/fx)
            # Gazebo camera convention: +X forward, +Y left, +Z up.
            self.ray_cache[key]=np.stack([np.ones_like(u),-u,-v],axis=-1)
        body=rotation(pose.get('roll_rad',0),pose.get('pitch_rad',0),pose.get('yaw_rad',0));mount=rotation(0,math.pi/2,0)
        camera_pos=np.array([pose['x_m'],pose['y_m'],pose['z_m']])+body@np.array([0,0,-.12])
        rays=self.ray_cache[key]@(body@mount).T
        valid=rays[:,:,2]<-1e-5;distance=-camera_pos[2]/np.minimum(rays[:,:,2],-1e-5)
        xs=camera_pos[0]+rays[:,:,0]*distance;ys=camera_pos[1]+rays[:,:,1]*distance
        b=self.metadata['world_bounds'];valid&=(xs>=b['x_min'])&(xs<=b['x_max'])&(ys>=b['y_min'])&(ys<=b['y_max'])&(distance>0)
        return xs,ys,valid
    def render(self,pose,width=640,height=512,sim_time_s=0,hfov=1.25):
        xs,ys,valid=self.ground_points(pose,width,height,hfov)
        b=self.metadata['world_bounds'];ny,nx=self.field.shape
        x=np.clip((xs-b['x_min'])/(b['x_max']-b['x_min'])*(nx-1),0,nx-1);y=np.clip((b['y_max']-ys)/(b['y_max']-b['y_min'])*(ny-1),0,ny-1)
        x0=x.astype(int);y0=y.astype(int);x1=np.minimum(x0+1,nx-1);y1=np.minimum(y0+1,ny-1);wx=x-x0;wy=y-y0
        kelvin=(1-wx)*(1-wy)*self.field[y0,x0]+wx*(1-wy)*self.field[y0,x1]+(1-wx)*wy*self.field[y1,x0]+wx*wy*self.field[y1,x1]
        rng=np.random.default_rng(int(sim_time_s*1000)%2**32+self.seed);kelvin=kelvin+rng.normal(0,.04,kelvin.shape)
        values=kelvin[valid]
        if not values.size:low,high=285.,286.
        else:
            low,high=np.percentile(values,[1,99]);center=(low+high)/2;span=max(.8,float(high-low));low=center-span/2;high=center+span/2
        pixels=np.clip((kelvin-low)/(high-low)*230+12,0,255).astype('uint8');pixels[~valid]=3
        image=Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(.65))
        display=dict(method='white_hot_percentile_AGC_NETD_0.04K_PSF_0.65px',input_unit='kelvin',output_unit='display_gray',low_k=float(low),high_k=float(high),low_gray=12,high_gray=242,clipped_fraction=float(((pixels==0)|(pixels==255)).mean()))
        return image,dict(source_kind='dumux_field_crop',source='Perspective physics camera driven by native Gazebo odometry',field_sha256=self.metadata['thermal_field_sha256'],physics_time_s=self.metadata['physics_time_s'],physics_model=self.metadata['physics'],thermal_model='Composed DuMux present-minus-absent surface response plus empirical recorded microtexture',display_mapping=display,research_accepted=False)
