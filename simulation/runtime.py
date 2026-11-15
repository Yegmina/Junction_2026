#!/usr/bin/python3
"""Native Gazebo transport adapter. All motion comes from integrated physics."""
import os
os.environ.update(GZ_PARTITION='aerorozum-mission-local',GZ_IP='127.0.0.1',GZ_DISCOVERY_MSG_PORT='18935',GZ_DISCOVERY_SRV_PORT='18936')
import collections,hashlib,io,json,math,threading,time,uuid,subprocess
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
from PIL import Image as PillowImage
from gz.transport13 import Node
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.world_stats_pb2 import WorldStatistics
from physics_camera import PhysicsCamera
from view_camera import DEFAULT as DEFAULT_VIEW,settings as view_settings,camera_pose

SSD=Path(os.environ.get('AEROROZUM_SIM_ROOT','/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/simulation'))

def rpy(q):
    return (math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y)),math.asin(np.clip(2*(q.w*q.y-q.z*q.x),-1,1)),math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))

class Runtime:
    def __init__(self):
        self.lock=threading.RLock();self.node=Node();self.pose=None;self.velocity={};self.sim_time=0.;self.paused=False;self.last_stats=0.;self.mode='landed';self.target=None;self.frames={};self.history=collections.deque(maxlen=2400);self.last_odom=0.;self.frame_id=0
        self.twist=self.node.advertise('/aerorozum_sim/gazebo/command/twist',Twist);self.enable=self.node.advertise('/aerorozum_sim/enable',Boolean)
        self.node.subscribe(Odometry,'/aerorozum_sim/odometry',self.odometry)
        self.node.subscribe(WorldStatistics,'/world/aerorozum_mission/stats',self.statistics)
        self.pending_frames={};self.frame_lock=threading.Lock();self.last_disk_frame={}
        self.view=dict(DEFAULT_VIEW);self.view_confirmed=None;self.view_error=None
        self.physics_camera=PhysicsCamera(SSD/'sensor-scene')
        self.run_id=uuid.uuid4().hex[:10]
        for kind in ['rgb','overview']:self.node.subscribe(Image,'/aerorozum_sim/'+kind,lambda msg,k=kind:self.enqueue_image(k,msg))
        threading.Thread(target=self.camera_worker,daemon=True).start()
        self.control_diagnostic={};self.control_thread=threading.Thread(target=self.control,daemon=True);self.control_thread.start()
        threading.Thread(target=self.view_worker,daemon=True).start()
    def set_view(self,data):
        with self.lock:self.view=view_settings(self.view,data)
        return self.view_status()
    def view_status(self):
        with self.lock:return dict(settings=dict(self.view),confirmed=self.view_confirmed,error=self.view_error)
    def view_worker(self):
        import xml.etree.ElementTree as E
        root=E.parse(SSD/'worlds/aerorozum.sdf').getroot()
        mount=tuple(map(float,root.findtext("world/model[@name='overview_camera']/link/sensor/pose").split()))
        previous=None
        while True:
            try:
                with self.lock:drone=None if self.pose is None else dict(self.pose);view=dict(self.view);paused=self.paused
                if drone is not None and not paused:
                    pose=camera_pose(view,drone,mount);signature=tuple(round(v,2) for v in [*pose['position'].values(),*pose['orientation'].values()])
                    if signature!=previous:
                        position=' '.join(f'{k}: {v:.8f}' for k,v in pose['position'].items());orientation=' '.join(f'{k}: {v:.9f}' for k,v in pose['orientation'].items())
                        request='name: "overview_camera" position { '+position+' } orientation { '+orientation+' }'
                        # Run outside the flight publisher's Node and Python GIL.
                        result=subprocess.run(['/usr/bin/gz','service','-s','/world/aerorozum_mission/set_pose','--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean','--timeout','600','--req',request],capture_output=True,text=True,timeout=2)
                        if result.returncode==0 and 'data: true' in result.stdout:
                            with self.lock:self.view_confirmed=dict(pose=pose['sensor_pose'],at_sim_time_s=self.sim_time);self.view_error=None
                            previous=signature
                        else:
                            with self.lock:self.view_error='Native overview camera is waiting for pose acknowledgement'
            except Exception as error:
                with self.lock:self.view_error=str(error)[:200]
            time.sleep(.25)
    def statistics(self,msg):
        with self.lock:self.paused=msg.paused;self.sim_time=msg.sim_time.sec+msg.sim_time.nsec*1e-9;self.last_stats=time.monotonic()
    def odometry(self,msg):
        p=msg.pose.position;a,b,c=rpy(msg.pose.orientation);at=msg.header.stamp.sec+msg.header.stamp.nsec*1e-9
        pose=dict(x_m=p.x,y_m=p.y,z_m=p.z,roll_rad=a,pitch_rad=b,yaw_rad=c)
        with self.lock:
            self.pose=pose;self.velocity=dict(x_m_s=msg.twist.linear.x,y_m_s=msg.twist.linear.y,z_m_s=msg.twist.linear.z);self.last_odom=time.monotonic();self.history.append((at,pose));self.sim_time=max(self.sim_time,at)
    def enqueue_image(self,kind,msg):
        # Transport callbacks must return promptly: decoding/compression and
        # removable-drive writes otherwise delay queued flight odometry.
        with self.frame_lock:self.pending_frames[kind]=msg
    def camera_worker(self):
        while True:
            with self.frame_lock:pending=self.pending_frames;self.pending_frames={}
            for kind,msg in pending.items():self.image(kind,msg)
            time.sleep(.02)
    def image(self,kind,msg):
        try:
            physics_camera=self.physics_camera
            at=msg.header.stamp.sec+msg.header.stamp.nsec*1e-9;w,h=msg.width,msg.height;step=msg.step
            with self.lock:
                if not self.history:return
                pose_at,pose=min(self.history,key=lambda item:abs(item[0]-at));pose=dict(pose);self.frame_id+=1;number=self.frame_id
            if abs(pose_at-at)>.15:return
            channels=step//w;pixels=np.frombuffer(msg.data,dtype=np.uint8).reshape(h,step)[:,:w*channels].reshape(h,w,channels)[:,:,:3].copy()
            fov=1.32 if kind=='overview' else 1.25;fx=w/(2*math.tan(fov/2));pair_id=f'gazebo-{self.run_id}-{number}'
            common=dict(sim_time_s=at,capture_sim_time_s=at,pose=pose,pose_sim_time_s=pose_at,pose_time_offset_s=pose_at-at,width=w,height=h,intrinsics=dict(fx=fx,fy=fx,cx=w/2-.5,cy=h/2-.5,hfov_rad=fov),camera_mount=dict(x_m=0,y_m=0,z_m=-.12,roll_rad=0,pitch_rad=math.pi/2,yaw_rad=0),pair_id=pair_id,scene_version=physics_camera.metadata['version'],physics_time_s=physics_camera.metadata['physics_time_s'])
            images=[(kind,PillowImage.fromarray(pixels),dict(source_kind='gazebo_camera',source='Native Gazebo RGB camera; photographic ground material',display_mapping=dict(native_encoding='RGB8',display='Identity RGB PNG')))]
            if kind=='rgb':
                thermal,physical=physics_camera.render(pose,w,h,at,fov);images.append(('thermal',thermal,physical))
            completed={}
            for name,im,extra in images:
                data=io.BytesIO();im.save(data,format='PNG',compress_level=1);encoded=data.getvalue()
                meta={**common,**extra,'kind':name,'event_id':pair_id+'-'+name,'camera_topic':'/aerorozum_sim/'+name,'sha256':hashlib.sha256(encoded).hexdigest()}
                if name=='rgb':meta['camera_mount']=dict(x_m=.03,y_m=0,z_m=-.13,roll_rad=0,pitch_rad=math.pi/2,yaw_rad=0)
                if name=='thermal':meta['camera_topic']='/aerorozum_sim/physics_thermal';meta['paired_rgb_topic']='/aerorozum_sim/rgb';meta['derived_from_native_rgb_timestamp']=True
                if name=='overview':
                    with self.lock:confirmed=self.view_confirmed
                    meta.update(drone_pose=pose,pose=confirmed['pose'] if confirmed else dict(x_m=22.,y_m=-24.,z_m=21.,roll_rad=0.,pitch_rad=.57,yaw_rad=2.31),pose_sim_time_s=confirmed['at_sim_time_s'] if confirmed else None,pose_time_offset_s=None,camera_pose_source='native_set_pose_acknowledgement' if confirmed else 'initial_sdf_camera_pose',camera_mount=dict(x_m=0,y_m=0,z_m=0,roll_rad=0,pitch_rad=0,yaw_rad=0),view_only=True)
                completed[name]=(encoded,meta)
            # RGB and thermal become visible as one atomic pair.
            with self.lock:self.frames.update(completed)
            for name,(encoded,meta) in completed.items():
                if time.monotonic()-self.last_disk_frame.get(name,0)>1:
                    path=SSD/'media'/(name+'_latest.png');temporary=path.with_suffix('.tmp');temporary.write_bytes(encoded);temporary.replace(path);self.last_disk_frame[name]=time.monotonic()
        except Exception as error:print(json.dumps({'camera_error':kind,'error':str(error)}),flush=True)
    def status(self):
        with self.lock:
            distance=None if not self.pose or not self.target else math.dist([self.pose[k] for k in ['x_m','y_m','z_m']],[self.target[k] for k in ['x_m','y_m','z_m']])
            return dict(ready=self.pose is not None and (self.paused or time.monotonic()-self.last_odom<2),simulation_only=True,backend='gazebo_native_multirotor',sim_time_s=self.sim_time,pose=self.pose,velocity=self.velocity,requested_target=self.target,mode=self.mode,paused=self.paused,arrived=distance is not None and distance<.35,frames={k:v[1] for k,v in self.frames.items()},odometry_age_s=None if not self.last_odom else time.monotonic()-self.last_odom,control=self.control_diagnostic,control_thread_alive=self.control_thread.is_alive(),world='aerorozum_mission',model='aerorozum_drone')
    def control(self):
        while True:
            with self.lock:
                pose=self.pose;target=self.target;mode=self.mode;velocity=dict(self.velocity)
                if mode=='landing' and pose and pose['z_m']<.30 and abs(self.velocity.get('z_m_s',0))<.25:self.mode='landed';self.target=None;mode='landed'
            enabled=mode!='landed';self.enable.publish(Boolean(data=enabled));command=Twist()
            if enabled and pose and target:
                world=np.array([np.clip(.7*(target[k]-pose[k])-.9*velocity.get(k+'_s',0),-.9 if k=='z_m' else -1.8,.9 if k=='z_m' else 1.8) for k in ['x_m','y_m','z_m']]);yaw=pose['yaw_rad'];c,s=math.cos(yaw),math.sin(yaw)
                command.linear.x=c*world[0]+s*world[1];command.linear.y=-s*world[0]+c*world[1];command.linear.z=world[2];command.angular.z=float(np.clip(math.atan2(math.sin(target['yaw_rad']-yaw),math.cos(target['yaw_rad']-yaw))*1.5,-1,1))
            published=self.twist.publish(command)
            self.control_diagnostic=dict(at=time.monotonic(),enabled=enabled,pose_z=None if not pose else pose['z_m'],target_z=None if not target else target['z_m'],command_z=command.linear.z,published=bool(published))
            time.sleep(.05)
    def target_position(self,data):
        values={k:float(data.get(k,0)) for k in ['x_m','y_m','z_m','yaw_rad']}
        if not all(math.isfinite(v) for v in values.values()) or abs(values['x_m'])>24 or abs(values['y_m'])>24 or not .15<=values['z_m']<=15:raise ValueError('Target outside simulation bounds')
        with self.lock:
            if self.pose is None:raise ValueError('Native odometry unavailable')
            self.target=values;self.mode='flying'
        return self.status()

    def snapshot(self,kind):
        with self.lock:
            if kind not in self.frames:raise ValueError('Native '+kind+' camera not ready')
            data,meta=self.frames[kind];meta=dict(meta)
        folder=SSD/'media/captures';folder.mkdir(exist_ok=True);path=folder/(kind+'-'+meta['event_id']+'.png');path.write_bytes(data);meta['path']=str(path);return meta

    def snapshot_pair(self):
        with self.lock:
            if not all(k in self.frames for k in ('thermal','rgb')):raise ValueError('Paired cameras are not ready')
            captured={k:(self.frames[k][0],dict(self.frames[k][1])) for k in ('thermal','rgb')}
        if captured['thermal'][1]['pair_id']!=captured['rgb'][1]['pair_id']:raise ValueError('Camera pair mismatch')
        result={};folder=SSD/'media/captures';folder.mkdir(exist_ok=True)
        for kind,(data,meta) in captured.items():
            path=folder/(kind+'-'+meta['event_id']+'.png');path.write_bytes(data);result[kind]={**meta,'path':str(path)}
        return result

    def set_paused(self,pause):
        for attempt in range(2):
            if self.last_stats and (self.paused or time.monotonic()-self.last_stats<2) and self.paused==pause:break
            # A synchronous transport request on the publisher's Node can hold
            # its transport lock while the native aircraft keeps moving. A
            # separate process leaves odometry and control publication live.
            try:
                result=subprocess.run(['/usr/bin/gz','service','-s','/world/aerorozum_mission/control','--reqtype','gz.msgs.WorldControl','--reptype','gz.msgs.Boolean','--timeout','5000','--req','pause: '+('true' if pause else 'false')],capture_output=True,text=True,timeout=6.5,check=False)
                if result.returncode:print(json.dumps({'pause_service_exit':result.returncode,'detail':(result.stdout+result.stderr)[-800:]}),flush=True)
            except subprocess.TimeoutExpired:pass
            deadline=time.monotonic()+1.5
            while time.monotonic()<deadline:
                if self.last_stats and time.monotonic()-self.last_stats<2 and self.paused==pause:break
                time.sleep(.01)
            if self.paused==pause and self.last_stats:break
        if not self.last_stats or self.paused!=pause:raise ValueError('Native WorldStats did not confirm pause state')
        if not pause:
            deadline=time.monotonic()+1
            while time.monotonic()-self.last_odom>.5 and time.monotonic()<deadline:time.sleep(.01)
        return self.status()
    def command(self,data):
        command=data.get('command')
        if command=='set_thermal_phase':
            phase=data.get('phase')
            if phase not in ('warming','cooling'):raise ValueError('Unknown thermal phase')
            camera=PhysicsCamera(SSD/'sensor-scene',phase+'-scene.json');self.physics_camera=camera
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                with self.lock:frame=self.frames.get('thermal')
                if frame and frame[1].get('field_sha256')==camera.metadata['thermal_field_sha256']:return self.status()
                time.sleep(.05)
            raise ValueError('New paired thermal phase not ready')
        if command in ('pause','resume'):return self.set_paused(command=='pause')
        if not self.pose:raise ValueError('Native odometry unavailable')
        if command=='takeoff':return self.target_position({**self.pose,'z_m':data.get('altitude_m',5)})
        if command=='hover':return self.target_position(self.pose)
        if command=='land':
            self.target_position({**self.pose,'z_m':.17});self.mode='landing';return self.status()
        raise ValueError('Unknown simulation command')
    def tick(self,seconds):
        seconds=float(seconds)
        if not math.isfinite(seconds) or not 0<=seconds<=10:raise ValueError('tick must be0..10 seconds')
        paused=self.paused
        if paused:self.set_paused(False)
        start=self.sim_time;deadline=time.monotonic()+seconds*5+10
        while self.sim_time<start+seconds and time.monotonic()<deadline:time.sleep(.01)
        if paused:self.set_paused(True)
        if self.sim_time<start+seconds:raise ValueError('Native simulation did not advance')
        return self.status()


RUNTIME=None

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,value,status=200,mime='application/json'):
        payload=value if isinstance(value,bytes) else json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        path=urlsplit(self.path).path
        try:
            if path=='/status':return self.send(RUNTIME.status())
            if path=='/view':return self.send(RUNTIME.view_status())
            if path=='/frame-pair':return self.send(RUNTIME.snapshot_pair())
            if path.startswith('/frame-info/'):return self.send(RUNTIME.snapshot(path.rsplit('/',1)[1]))
            if path.startswith('/frame/'):
                kind=path.rsplit('/',1)[1]
                with RUNTIME.lock:
                    if kind not in RUNTIME.frames:raise ValueError('Native camera not ready')
                    data=RUNTIME.frames[kind][0]
                return self.send(data,mime='image/png')
            self.send({'error':'Unknown route'},404)
        except Exception as error:self.send({'error':str(error)},503)
    def do_POST(self):
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size>16384:raise ValueError('Command is too large')
            data=json.loads(self.rfile.read(size));path=urlsplit(self.path).path
            if path=='/target':result=RUNTIME.target_position(data)
            elif path=='/view':result=RUNTIME.set_view(data)
            elif path=='/command':result=RUNTIME.command(data)
            elif path=='/tick':result=RUNTIME.tick(data['seconds'])
            else:return self.send({'error':'Unknown route'},404)
            self.send(result)
        except Exception as error:self.send({'error':str(error)},400)

if __name__=='__main__':
    RUNTIME=Runtime();print('Native Gazebo adapter listening127.0.0.1:18930',flush=True);ThreadingHTTPServer(('127.0.0.1',18930),Handler).serve_forever()
