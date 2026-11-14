"""Public-only survey baseline: compact thermal components and geometric revisits.

No simulator files, reference locations, target counts or grading endpoints are
read. Polarity, camera calibration, regions and budgets come from public tools.
"""
import json,subprocess,time,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,label,find_objects,maximum_filter
from camera_geometry import pixel_to_ground


def call(command,*args):
    process=subprocess.run(['drone',command,*map(str,args)],capture_output=True,text=True)
    if process.returncode:raise RuntimeError(process.stdout+process.stderr)
    return json.loads(process.stdout)


def at(x,y,z):
    for _ in range(600):
        state=call('status');p=state['pose']
        if math.dist((p['x_m'],p['y_m'],p['z_m']),(x,y,z))<.45:return state
        if state['remaining_s']<math.hypot(p['x_m'],p['y_m'])/1.2+15:return state
        time.sleep(.3)
    raise RuntimeError('Navigation did not settle')


state=call('status');phase=state.get('thermal_condition',{}).get('phase','warming');regions=state['regions']
state=call('launch','--altitude',5);candidates=[]


def capture():
    for attempt in range(3):
        try:observation=call('observe');break
        except RuntimeError:
            if attempt==2:raise
            time.sleep(.5)
    item=next(i for i in observation['images'] if i['kind']=='thermal')
    pixels=np.asarray(Image.open(item['local_file']).convert('L'),dtype=np.float32)
    filtered=gaussian_filter(pixels,.8);signal=filtered if phase=='warming' else 255-filtered
    # Saturated compact components are a conservative first proposal source.
    components,count=label(signal>247);proposals=[]
    for index,slices in enumerate(find_objects(components),1):
        if slices is None:continue
        yy,xx=slices;mask=components[yy,xx]==index;area=int(mask.sum());height=yy.stop-yy.start;width=xx.stop-xx.start
        if not 35<=area<=2500 or min(width,height)<4 or max(width,height)>3*min(width,height) or area/(width*height)<.45:continue
        vs,us=np.nonzero(mask);u=float(us.mean()+xx.start);v=float(vs.mean()+yy.start)
        proposals.append((u,v,max(width,height)/2+3,float(signal[yy,xx][mask].mean())))
    proposals.sort(key=lambda r:r[3],reverse=True)
    for u,v,radius,_ in proposals[:8]:
        h,w=pixels.shape;point=pixel_to_ground(observation,u/w*1000,v/h*1000)
        if not any(a<=point[0]<=c and b<=point[1]<=d for a,b,c,d in regions.values()):continue
        radius=min(28,max(6,radius));bbox=dict(x_min=max(0,u-radius)/w*1000,y_min=max(0,v-radius)/h*1000,x_max=min(w,u+radius)/w*1000,y_max=min(h,v+radius)/h*1000)
        evidence=dict(observation_id=observation['observation_id'],bbox=bbox)
        existing=next((c for c in candidates if math.dist(c['_point'],point)<.75),None)
        if existing:existing['evidence'].append(evidence)
        elif len(candidates)<24:candidates.append(dict(_point=point,notes='Compact thermal signature; geometric association uses public capture calibration',evidence=[evidence]))


for x,y in [(7.5,-2.5),(12.5,-2.5),(12.5,2.5),(7.5,2.5),(-2.5,9.5),(2.5,9.5),(2.5,14.5),(-2.5,14.5)]:
    state=call('status');p=state['pose'];reserve=math.hypot(x,y)/1.2+15
    if state['remaining_s']<math.hypot(p['x_m']-x,p['y_m']-y)/1.2+reserve+8:break
    call('goto','--x',x,'--y',y,'--altitude',5);state=at(x,y,5)
    if state['remaining_s']<math.hypot(state['pose']['x_m'],state['pose']['y_m'])/1.2+12:break
    capture()

# Resolve known uncertain signatures before spending budget on redundant views.
pending=list(candidates)
while pending:
    state=call('status');p=state['pose'];pending.sort(key=lambda c:math.dist(c['_point'],(p['x_m'],p['y_m'])))
    candidate=pending.pop(0);x,y=candidate['_point']
    cost=math.hypot(p['x_m']-x,p['y_m']-y)/1.2+math.hypot(x,y)/1.2+20
    if state['remaining_s']<cost or state['action_count']>58:break
    call('goto','--x',x,'--y',y,'--altitude',3);at(x,y,3);capture()
call('land');path=Path('/workspace/report.json');path.write_text(json.dumps(dict(candidates=[{k:v for k,v in c.items() if not k.startswith('_')} for c in candidates])))
print(json.dumps(call('submit','--report',path),indent=2))
