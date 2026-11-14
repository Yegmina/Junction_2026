"""Public tools over native Gazebo; authoritative evidence and grading stay here."""
import copy,hashlib,json,math,threading,time,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import urllib.request,urllib.error
import numpy as np
from physics_camera import rotation

REGIONS={'east':(5,-5,15,5),'north':(-5,7,5,17)}
ROOT=Path('/scene');LOCK=threading.RLock()

def native(path,body=None):
    req=urllib.request.Request('http://127.0.0.1:18930'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=45) as response:return json.load(response)

def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def project(row,u,v):
    """Ray/plane intersection using the exact captured camera pose and model."""
    p=row['pose'];w=row['width'];h=row['height'];fx=row['intrinsics']['fx'];fy=row['intrinsics']['fy']
    r=rotation(p.get('roll_rad',0),p.get('pitch_rad',0),p.get('yaw_rad',0))
    position=np.array([p['x_m'],p['y_m'],p['z_m']])+r@np.array([0,0,-.12])
    ray=r@rotation(0,math.pi/2,0)@np.array([1,-(u*w/1000+.5-w/2)/fx,-(v*h/1000+.5-h/2)/fy])
    if ray[2]>=-1e-6:raise ValueError('Evidence ray does not intersect the ground')
    hit=position-ray*(position[2]/ray[2]);return (float(hit[0]),float(hit[1]))

def inside(point,polygon):
    sign=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):sign.append((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0]))
    return all(v>=-1e-7 for v in sign) or all(v<=1e-7 for v in sign)

class Investigation:
    def __init__(self):
        self.start=None;self.end=None;self.actions=0;self.observations={};self.trace=[];self.report=None;self.sealed=False;self.result=None;self.samples=[];self.budget=180.;self.errors=[]
        truth=json.loads((ROOT/'provenance/scene.truth.json').read_text())['targets']
        self.targets=[dict(id=str(t['id']),x=float(t['x_m']),y=float(t['y_m'])) for t in truth if any(a<=t['x_m']<=c and b<=t['y_m']<=d for a,b,c,d in REGIONS.values())]
        if len(self.targets)!=3:raise ValueError('Frozen benchmark requires three eligible inserted signatures')
        self.cells=[(x+.5,y+.5) for a,b,c,d in REGIONS.values() for x in range(a,c) for y in range(b,d)]
        self.covered=set();self.home_return=False;self.inference_freeze=False
    def elapsed(self,s):return 0. if self.start is None else max(0,(float(s['sim_time_s']) if self.end is None else self.end)-self.start)
    def status(self):
        s=native('/status');p=s.get('pose') or {};elapsed=self.elapsed(s)
        return dict(ready=s.get('ready',False),pose=p,velocity=s.get('velocity'),sim_time_s=s.get('sim_time_s'),flight_elapsed_s=round(elapsed,3),remaining_s=max(0,round(self.budget-elapsed,3)),action_count=self.actions,action_limit=64,regions=REGIONS,coverage_fraction=len(self.covered)/len(self.cells),observations=[{k:r[k] for k in ('id','pose','sim_time_s','width','height','images')} for r in self.observations.values()],report_submitted=self.report is not None,sealed=self.sealed,simulation_only=True)
    def wait(self,condition,timeout=300):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            s=native('/status');p=s.get('pose') or {}
            if condition(s,p):return s
            time.sleep(.15)
        raise ValueError('Native flight did not settle before wall-clock timeout')
    def goto(self,x,y,z):
        if not all(finite(v) for v in (x,y,z)) or not -24<=x<=24 or not -24<=y<=24 or not 2<=z<=10:raise ValueError('goto requires finite x/y in -24..24 and altitude in 2..10 metres')
        return native('/target',dict(x_m=x,y_m=y,z_m=z,yaw_rad=0.))
    def observe(self):
        state=native('/status');pair=native('/frame-pair');r=copy.deepcopy(pair['thermal']);rgb=pair['rgb']
        if r['pair_id']!=rgb['pair_id'] or r['pose']!=rgb['pose'] or r['sim_time_s']!=rgb['sim_time_s']:raise ValueError('Camera pair identity mismatch')
        if r['pose']['z_m']<1.5 or float(state['sim_time_s'])-r['sim_time_s']>1:raise ValueError('Airborne fresh camera evidence required; wait for the next frame')
        identifier=uuid.uuid4().hex[:12];images=[]
        for kind,receipt in pair.items():
            path=Path(receipt['path']).resolve()
            if not path.is_relative_to((ROOT/'media/captures').resolve()):raise ValueError('Capture outside native storage')
            content=path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=receipt['sha256']:raise ValueError('Native image hash mismatch')
            name=identifier+'-'+kind+'.png';destination=ROOT/'evidence'/name;destination.parent.mkdir(exist_ok=True);destination.write_bytes(content)
            images.append(dict(filename=name,url='/evidence/'+name,kind=kind,sha256=receipt['sha256']))
        row={**r,'id':identifier,'images':images};self.observations[identifier]=row
        polygon=[project(row,u,v) for u,v in [(0,0),(1000,0),(1000,1000),(0,1000)]]
        self.covered.update(i for i,cell in enumerate(self.cells) if inside(cell,polygon))
        return dict(observation_id=identifier,pose=row['pose'],sim_time_s=row['sim_time_s'],images=images,display_mapping=row.get('display_mapping'),status=self.status())
    def land(self):
        s=native('/status');z=max(2,min(5,s['pose']['z_m']));self.goto(0,0,z)
        self.wait(lambda s,p:math.hypot(p.get('x_m',99),p.get('y_m',99))<.5 and abs(p.get('z_m',99)-z)<.5)
        native('/command',{'command':'land'})
        self.wait(lambda s,p:math.hypot(p.get('x_m',99),p.get('y_m',99))<.8 and p.get('z_m',99)<.33 and math.sqrt(sum(v*v for v in s.get('velocity',{}).values()))<.3)
        self.home_return=True
        self.end=float(native('/status')['sim_time_s'])
    def tool(self,data):
        command=data.get('command')
        if command in ('status','result'):
            if command=='result' and not self.sealed:raise ValueError('Result is available only after the irreversible report submission')
            return self.result if command=='result' else self.status()
        if self.sealed:raise ValueError('Investigation is sealed; there is no reset or further evidence collection')
        if self.end is not None and command!='submit':raise ValueError('The flight has ended; submit the collected evidence')
        self.actions+=1
        if self.actions>64:raise ValueError('Tool action limit exceeded')
        if self.start is not None and self.elapsed(native('/status'))>self.budget and command not in ('land','submit'):raise ValueError('Flight budget exceeded; return and submit the collected evidence')
        if command=='launch':
            if self.start is not None:raise ValueError('This trial has one continuous flight budget; relaunch cannot reset it')
            altitude=data.get('altitude',5.)
            if not finite(altitude) or not 2<=altitude<=10:raise ValueError('Launch altitude must be 2..10m')
            self.start=float(native('/status')['sim_time_s']);native('/command',{'command':'takeoff','altitude_m':altitude})
            self.wait(lambda s,p:abs(p.get('z_m',99)-altitude)<.4)
        elif command=='goto':
            if self.start is None:raise ValueError('Launch first')
            self.goto(data.get('x'),data.get('y'),data.get('altitude'))
        elif command=='observe':return self.observe()
        elif command=='wait':
            seconds=data.get('seconds',2)
            if not finite(seconds) or not 0<=seconds<=5:raise ValueError('Wait must be 0..5 simulated seconds')
            native('/tick',{'seconds':seconds})
        elif command=='land':self.land()
        elif command=='submit':
            report=data.get('report')
            if not isinstance(report,dict) or not isinstance(report.get('candidates'),list) or len(report['candidates'])>32:raise ValueError('Report must contain a candidates list with at most 32 entries')
            self.report=copy.deepcopy(report);return self.seal()
        else:raise ValueError('Unknown public tool')
        return self.status()
    def score(self):
        s=native('/status');p=s['pose'];elapsed=self.elapsed(s)
        completed=bool(self.report is not None and self.observations and self.home_return and math.hypot(p['x_m'],p['y_m'])<.8 and p['z_m']<.4 and elapsed<=self.budget and self.actions<=64)
        candidates=(self.report or {}).get('candidates',[]);edges={};views={};invalid=0
        for i,candidate in enumerate(candidates):
            try:
                refs=candidate['evidence']
                if not isinstance(refs,list) or not 1<=len(refs)<=16:raise ValueError('Evidence count')
                for e in refs:
                    row=self.observations[e['observation_id']];b=e['bbox'];coords=[b[k] for k in ('x_min','y_min','x_max','y_max')]
                    if not all(finite(v) and 0<=v<=1000 for v in coords):raise ValueError('Invalid box')
                    x0,y0,x1,y1=coords
                    if not (x0<x1 and y0<y1 and (x1-x0)*(y1-y0)<=250000):raise ValueError('Invalid box size')
                    hit=project(row,(x0+x1)/2,(y0+y1)/2)
                    for target in self.targets:
                        if math.dist(hit,(target['x'],target['y']))<=.65:
                            edges.setdefault(i,set()).add(target['id']);views.setdefault((i,target['id']),{})[row['id']]=row['pose']
            except (KeyError,ValueError,TypeError):
                invalid+=1;edges.pop(i,None)
        assigned={}
        def match(index,seen):
            for target in sorted(edges.get(index,())):
                if target in seen:continue
                seen.add(target)
                if target not in assigned or match(assigned[target],seen):assigned[target]=index;return True
            return False
        for index in range(len(candidates)):match(index,set())
        tp=len(assigned);fp=len(candidates)-tp;fn=len(self.targets)-tp
        f2=5*tp/max(1,5*tp+4*fn+fp);coverage=len(self.covered)/len(self.cells)
        confirmed=0
        for target,index in assigned.items():
            poses=list(views.get((index,target),{}).values())
            confirmed+=int(any(math.dist((a['x_m'],a['y_m'],a['z_m']),(b['x_m'],b['y_m'],b['z_m']))>=1 for j,a in enumerate(poses) for b in poses[j+1:]))
        reward=(.65*f2+.2*coverage+.15*confirmed/len(self.targets)) if completed else 0.
        return dict(reward=round(reward,8),completed=completed,true_positives=tp,false_positives=fp,false_negatives=fn,f2=f2,qualified_coverage=coverage,multi_view_matches=confirmed,invalid_candidates=invalid,observations=len(self.observations),actions=self.actions,flight_elapsed_s=elapsed,budget_s=self.budget,returned_home=self.home_return,scope='Inserted synthetic-object retrieval in experimental composed thermal imagery; not physical mine confirmation')
    def seal(self):
        if self.sealed:return self.result
        self.result=self.score();self.sealed=True
        document=dict(result=self.result,report=self.report,trace=self.trace,observations=self.observations)
        (ROOT/'run.json').write_text(json.dumps(document,indent=2));return self.result

TASK=None
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,data,status=200,kind='application/json'):
        body=data if isinstance(data,bytes) else json.dumps(data,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        try:
            if self.path=='/health':
                s=native('/status')
                if not s.get('ready') or not {'thermal','rgb'}<=set(s.get('frames',{})):raise ValueError('Native cameras not ready')
                return self.reply({'ready':True,'backend':'native_gazebo'})
            if self.path.startswith('/evidence/'):
                path=(ROOT/'evidence'/self.path.removeprefix('/evidence/')).resolve()
                if not path.is_relative_to((ROOT/'evidence').resolve()) or path.suffix!='.png':raise ValueError('Unknown evidence')
                return self.reply(path.read_bytes(),kind='image/png')
            if self.path=='/ledger' and TASK.sealed:return self.reply(json.loads((ROOT/'run.json').read_text()))
            self.reply({'error':'Unknown public endpoint'},404)
        except Exception as error:self.reply({'error':str(error)},503)
    def do_POST(self):
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=100000:raise ValueError('Invalid payload size')
            data=json.loads(self.rfile.read(size))
            with LOCK:
                if self.path=='/seal':return self.reply(TASK.seal())
                if self.path!='/tool':return self.reply({'error':'Unknown public endpoint'},404)
                result=TASK.tool(data);TASK.trace.append(dict(at=time.time(),command=data,result=copy.deepcopy(result)))
            return self.reply(result)
        except Exception as error:
            with LOCK:TASK.errors.append(str(error))
            self.reply({'error':str(error)},400)

if __name__=='__main__':
    TASK=Investigation()
    ThreadingHTTPServer(('0.0.0.0',8080),Handler).serve_forever()
