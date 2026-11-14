"""Loopback client for native Gazebo simulation; contains no drone SDK."""
from __future__ import annotations
import json,math,os
from pathlib import Path
import urllib.parse,urllib.request

class GazeboUnavailable(RuntimeError):pass

class GazeboBridge:
    def __init__(self,base_url=None,timeout=20):
        self.base_url=(base_url or os.environ.get('AEROROZUM_GAZEBO_URL','http://127.0.0.1:18930')).rstrip('/')
        parsed=urllib.parse.urlsplit(self.base_url)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password:raise ValueError('Gazebo adapter must use loopback HTTP')
        self.timeout=float(timeout)
    def _request(self,path,payload=None,binary=False,timeout=None):
        request=urllib.request.Request(self.base_url+path,data=None if payload is None else json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json'},method='GET' if payload is None else 'POST')
        try:
            with urllib.request.urlopen(request,timeout=timeout or self.timeout) as response:
                body=response.read();return body if binary else json.loads(body)
        except (OSError,ValueError) as error:raise GazeboUnavailable('Native Gazebo adapter unavailable: '+str(error)) from error
    def status(self):
        try:return self._request('/status')
        except GazeboUnavailable as error:return {'ready':False,'simulation_only':True,'backend':'gazebo_native_multirotor','pose':None,'error':str(error)}
    def pose(self):
        status=self.status()
        if not status.get('ready'):raise GazeboUnavailable(status.get('error','Native odometry not ready'))
        return status['pose']
    def takeoff(self,altitude_m=5.):return self._request('/command',{'command':'takeoff','altitude_m':float(altitude_m)})
    def set_target(self,x_m,y_m,z_m,yaw_rad=0.):
        values=[float(x_m),float(y_m),float(z_m),float(yaw_rad)]
        if not all(math.isfinite(v) for v in values):raise ValueError('Finite simulation coordinates required')
        if abs(values[0])>24 or abs(values[1])>24 or not .15<=values[2]<=15:raise ValueError('Target outside simulated mission bounds')
        return self._request('/target',dict(zip(('x_m','y_m','z_m','yaw_rad'),values)))
    def land(self):return self._request('/command',{'command':'land'})
    def hover(self):return self._request('/command',{'command':'hover'})
    def set_thermal_phase(self,phase):return self._request('/command',{'command':'set_thermal_phase','phase':phase})
    def pause(self):return self._request('/command',{'command':'pause'})
    def resume(self):return self._request('/command',{'command':'resume'})
    def tick(self,seconds=.5):
        seconds=float(seconds)
        if not math.isfinite(seconds) or not 0<=seconds<=10:raise ValueError('tick must be0..10 actual simulation seconds')
        return self._request('/tick',{'seconds':seconds},timeout=max(self.timeout,seconds*5+5))
    def frame(self,kind='thermal'):
        if kind not in ('thermal','rgb','overview'):raise ValueError('Unknown simulated camera')
        return self._request('/frame-info/'+kind)
    def read_frame(self,kind='thermal'):
        if kind not in ('thermal','rgb','overview'):raise ValueError('Unknown simulated camera')
        return self._request('/frame/'+kind,binary=True)
    def capture(self,destination,kind='thermal'):
        return self._copy_capture(self.frame(kind),destination)
    def capture_pair(self,folder,stem):
        pair=self._request('/frame-pair')
        if pair['thermal'].get('pair_id')!=pair['rgb'].get('pair_id'):raise GazeboUnavailable('Paired camera identities differ')
        if pair['thermal'].get('sim_time_s')!=pair['rgb'].get('sim_time_s') or pair['thermal'].get('pose')!=pair['rgb'].get('pose'):raise GazeboUnavailable('Paired camera pose or timestamp differs')
        return {kind:self._copy_capture(info,Path(folder)/(stem+('' if kind=='thermal' else '-rgb')+'.png')) for kind,info in pair.items()}
    def _copy_capture(self,info,destination):
        import hashlib
        source=Path(info['path']).resolve();allowed=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/simulation/media/captures').resolve()
        if not source.is_relative_to(allowed):raise GazeboUnavailable('Capture path is outside native media directory')
        data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=info['sha256']:raise GazeboUnavailable('Native capture checksum differs')
        path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        intrinsics=dict(info.get('intrinsics',{}))
        for key in ('fx','fy','cx','cy'):
            if key in intrinsics:intrinsics[key+'_px']=intrinsics[key]
        mapping=info.get('display_mapping',{'native_encoding':'RGB8','display':'Identity RGB PNG'})
        return {**info,'intrinsics':intrinsics,'source_path':str(source),'path':str(path),'image_sha256':info['sha256'],'atomic_capture':True,'pose_frame':'gazebo_world_ENU','display_mapping':mapping}
