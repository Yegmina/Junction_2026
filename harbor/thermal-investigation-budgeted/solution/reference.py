"""Inspect within each region while reserving flight for the remaining mission."""
import json,math,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
from camera_geometry import pixel_to_ground
from investigation_vision import propose


def call(command,*args):
    process=subprocess.run(['drone',command,*map(str,args)],capture_output=True,text=True)
    if process.returncode:raise RuntimeError(process.stdout+process.stderr)
    return json.loads(process.stdout)


def pose(state):
    p=state['pose'];return p['x_m'],p['y_m'],p['z_m']


def leg(a,b):return max(math.hypot(a[0]-b[0],a[1]-b[1])/1.2,abs(a[2]-b[2])/.9)+1.5


def home_cost(p):
    z=min(5,p[2]);return leg(p,(0,0,z))+z/.8+2


def remaining_cost(p,regions):
    total=0.;pending=list(regions)
    while pending:
        pending.sort(key=lambda r:math.dist((r[1][0],r[1][1]),p[:2]));_,target,_=pending.pop(0)
        total+=leg(p,target)+1;p=target
    return total+home_cost(p)


def at(target):
    for _ in range(600):
        state=call('status')
        if math.dist(pose(state),target)<.45:return state
        if state['remaining_s']<home_cost(pose(state)):return state
        time.sleep(.3)
    raise RuntimeError('Navigation did not settle')


initial=call('status');phase=initial['thermal_condition']['phase'];regions=initial['regions'];tracks=[]
pending=[(name,((a+c)/2,(b+d)/2,10),(a,b,c,d)) for name,(a,b,c,d) in regions.items()]


def capture():
    for attempt in range(3):
        try:observation=call('observe');break
        except RuntimeError:
            if attempt==2:raise
            time.sleep(.5)
    files={item['kind']:item['local_file'] for item in observation['images']}
    with Image.open(files['thermal']) as image:thermal=np.asarray(image.convert('L'))
    with Image.open(files['rgb']) as image:rgb=np.asarray(image.convert('RGB'))
    height,width=thermal.shape;close=observation['pose']['z_m']<=4.5
    for u,v,radius,strength,quality in propose(observation,thermal,rgb,phase):
        point=pixel_to_ground(observation,u/width*1000,v/height*1000)
        if not any(a<=point[0]<=c and b<=point[1]<=d for a,b,c,d in regions.values()):continue
        radius=min(60,max(4,radius));bbox=dict(x_min=max(0,u-radius)/width*1000,y_min=max(0,v-radius)/height*1000,
            x_max=min(width,u+radius)/width*1000,y_max=min(height,v+radius)/height*1000)
        evidence=dict(observation_id=observation['observation_id'],bbox=bbox)
        near=[t for t in tracks if math.dist(t['_point'],point)<.75 and all(math.dist(p,point)<=1.2 for p in t['_points'])]
        track=min(near,key=lambda t:math.dist(t['_point'],point)) if near else None
        if track:
            if any(e['observation_id']==observation['observation_id'] for e in track['evidence']):continue
            track['evidence'].append(evidence);track['_points'].append(point)
            track['_point']=tuple(float(v) for v in np.median(track['_points'],axis=0))
            track['_close']|=close;track['_quality']=max(track['_quality'],quality);track['_strength']=max(track['_strength'],strength)
        elif len(tracks)<24:
            tracks.append(dict(_point=point,_points=[point],_close=close,_attempted=False,_quality=quality,_strength=strength,
                _first_view=(observation['pose']['x_m'],observation['pose']['y_m']),
                notes='Compact thermal residual with registered RGB and radial-shape support; provisional synthetic signature',evidence=[evidence]))


# Ascend while travelling after a low takeoff, preserving the actual dynamics.
call('launch','--altitude',2)
while pending:
    state=call('status');p=pose(state);pending.sort(key=lambda r:math.dist(r[1][:2],p[:2]))
    _,overview,bounds=pending.pop(0)
    if state['remaining_s']<leg(p,overview)+1+remaining_cost(overview,pending):break
    call('goto','--x',overview[0],'--y',overview[1],'--altitude',10);at(overview);capture()
    while True:
        state=call('status');p=pose(state);a,b,c,d=bounds
        options=[t for t in tracks if not t['_close'] and not t['_attempted'] and a<=t['_point'][0]<=c and b<=t['_point'][1]<=d]
        if not options:break
        options.sort(key=lambda t:-(t['_strength']*(.1+t['_quality'])**4)/(1+math.dist(t['_point'],p[:2])))
        target=options[0];x,y=target['_point']
        if math.dist((x,y),target['_first_view'])<1.2:x+=1.4
        close=(x,y,4)
        reserve=leg(p,close)+1+remaining_cost(close,pending)+6*len(pending)
        if state['remaining_s']<reserve or state['action_count']>57:break
        target['_attempted']=True
        call('goto','--x',x,'--y',y,'--altitude',4);at(close);capture()
call('land')
qualified=[];provisional=[]
for track in tracks:
    record={k:v for k,v in track.items() if not k.startswith('_')}
    (qualified if track['_close'] and track['_quality']>=.75 else provisional).append(record)
path=Path('/workspace/report.json');path.write_text(json.dumps(dict(candidates=qualified,provisional_candidates=provisional)))
print(json.dumps(call('submit','--report',path),indent=2))
