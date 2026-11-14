"""Prompt-driven investigations using native Gazebo pose and camera evidence."""
from __future__ import annotations
import asyncio, copy, json, math, os, re, uuid
from pathlib import Path
from PIL import Image, ImageDraw
from .common import config, data_root, redacted_error, save, secret, utcnow

BOUNDS = dict(x_min=-24,y_min=-24,x_max=24,y_max=24)
AREAS = [dict(id=k,name=k.title(),center=dict(x_m=x,y_m=y),bounds=dict(x_min=x-5,y_min=y-5,x_max=x+5,y_max=y+5)) for k,x,y in [('west',-10,0),('east',10,0),('north',0,12),('south',0,-12)]]
FINAL = {'completed','stopped','failed','interrupted'}

def parse_mission_request(prompt):
    if not isinstance(prompt,str) or not 4 <= len(prompt.strip()) <= 2000: raise ValueError('Describe a mission in 4–2000 characters')
    text=prompt.lower(); forbidden=[]
    for area in AREAS:
        if re.search(r"(?:avoid|skip|do not enter|do not inspect|don't inspect)\s+(?:the\s+)?"+area['id']+r'\b',text):forbidden.append(area['id'])
    requested=[a['id'] for a in AREAS if re.search(r'\b'+a['id']+r'\b',text) and a['id'] not in forbidden]
    allowed=requested or [a['id'] for a in AREAS if a['id'] not in forbidden]
    if not allowed:raise ValueError('Mission excludes all inspection areas')
    match=re.search(r'(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?)\b',text)
    budget=float(match[1])*(60 if match[2].startswith('m') else 1) if match else 180
    match=re.search(r'(?:altitude\s*(?:of\s*)?|at\s+)(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b',text)
    return dict(allowed_areas=allowed,forbidden_areas=forbidden,budget_s=max(30,min(600,budget)),altitude_m=max(2,min(10,float(match[1]) if match else 5)),revisit_uncertain=bool(re.search('uncertain|revisit|careful|thorough|subtle',text)) and 'do not revisit' not in text)

class MissionManager:
    def __init__(self,root=None,bridge=None,provider_factory=None,detector=None):
        self.root=Path(root or data_root());self.folder=self.root/'missions';self.folder.mkdir(parents=True,exist_ok=True)
        self._bridge=bridge;self.provider_factory=provider_factory;self.detector=detector;self.states={};self.tasks={};self.active=None;self._create_lock=asyncio.Lock()
        for path in self.folder.glob('*/state.json'):
            try:
                state=json.loads(path.read_text())
                if state['status'] not in FINAL:state.update(status='interrupted',error='Mission service restarted; inspect simulator before restarting.')
                self.states[state['id']]=state
            except (ValueError,KeyError,OSError):pass
    @property
    def bridge(self):
        if self._bridge is None:
            from .gazebo_bridge import GazeboBridge
            self._bridge=GazeboBridge()
        return self._bridge
    def simulation_status(self):
        status=self.bridge.status()
        local=os.environ.get('AEROROZUM_MISSION_PROVIDER')=='local_gemma'
        return dict(available=bool(status.get('ready')),status='ready' if status.get('ready') else 'unavailable',backend='gazebo_native_multirotor',agent_enabled=bool(self.provider_factory or local or secret('GEMINI_API_KEY','AEROROZUM_GEMINI_KEY_FILE')),agent_provider='local_gemma' if local else 'gemini_api',agent_model=self._agent_model(),map_bounds=BOUNDS,home=dict(x_m=0,y_m=0),thermal_fidelity='DuMux-derived grayscale thermal + native RGB; experimental composed ground, not field validated',simulation_only=True,error=status.get('error'))
    def _agent_model(self):
        return os.environ.get('AEROROZUM_MISSION_MODEL','google/gemma-4-12B-it') if os.environ.get('AEROROZUM_MISSION_PROVIDER')=='local_gemma' else config()['gemini']['bulk_model']
    def get(self,identifier):
        state=self.states[identifier]
        return copy.deepcopy({k:v for k,v in state.items() if not k.startswith('_')})
    def list(self):return [self.get(k) for k in self.states]
    def _save(self,s):save(self.folder/s['id']/'state.json',s)
    def _event(self,s,action,detail):
        s['events'].append(dict(at=utcnow(),time_s=s['budget']['elapsed_s'],action=action,detail=detail));self._save(s)
    async def create(self,prompt,controller='agent',seed=0,physics_phase='warming'):
        async with self._create_lock:
            return await self._create(prompt,controller,seed,physics_phase)
    async def _create(self,prompt,controller='agent',seed=0,physics_phase='warming'):
        plan=parse_mission_request(prompt)
        if controller not in ('agent','reference'):raise ValueError('Controller must be agent or reference')
        if physics_phase not in ('warming','cooling'):raise ValueError('Thermal phase must be warming or cooling')
        if self.active:raise ValueError('Another simulated mission is active')
        sim=await asyncio.to_thread(self.simulation_status)
        if not sim['available']:raise ValueError('Native Gazebo is not ready')
        if controller=='agent' and not sim['agent_enabled']:raise ValueError('Agent API access is not configured')
        if hasattr(self.bridge,'set_thermal_phase'):await self._call('set_thermal_phase',physics_phase)
        identifier=uuid.uuid4().hex[:16]
        state=dict(id=identifier,prompt=prompt,controller=controller,seed=seed,physics_phase=physics_phase,world_seed_applied=False,status='planning',created_at=utcnow(),plan=plan,simulation=sim,map_bounds=BOUNDS,areas=AREAS,pose=None,route=[],observations=[],events=[],budget=dict(limit_s=plan['budget_s'],elapsed_s=0,remaining_s=plan['budget_s']),report=None,error=None,_stop=False,_start_sim=None,_inference_s=0)
        self.states[identifier]=state;self.active=identifier;self._save(state)
        self.tasks[identifier]=asyncio.create_task(self._run(state));return self.get(identifier)
    async def stop(self,identifier):
        s=self.states[identifier]
        if s['status'] not in FINAL:s['_stop']=True;self._event(s,'stop_requested','Return home and preserve collected evidence')
        return self.get(identifier)
    async def _call(self,name,*args):return await asyncio.to_thread(getattr(self.bridge,name),*args)
    async def _world_pause(self,paused):
        for attempt in range(3):
            try:return await self._call('pause' if paused else 'resume')
            except RuntimeError:
                if attempt==2:raise
                await asyncio.sleep(.2)
    async def _refresh(self,s):
        native=await self._call('status');p=native.get('pose')
        if not native.get('ready') or not isinstance(p,dict):raise RuntimeError('Native Gazebo odometry unavailable')
        pose=dict(x_m=float(p['x_m']),y_m=float(p['y_m']),altitude_m=float(p.get('z_m',p.get('altitude_m'))),yaw_deg=math.degrees(p.get('yaw_rad',0)))
        now=float(native['sim_time_s'])
        if s['_start_sim'] is None:s['_start_sim']=now
        elapsed=max(0,now-s['_start_sim']-s['_inference_s'])
        s['pose']=pose;s['budget'].update(elapsed_s=round(elapsed,2),remaining_s=round(max(0,s['budget']['limit_s']-elapsed),2))
        if not s['route'] or sum((pose[k]-s['route'][-1][k])**2 for k in ('x_m','y_m','altitude_m'))>.01:s['route'].append(dict(**pose,time_s=round(elapsed,2)))
        self._save(s);return native
    async def _wait(self,s,target,allow_stop=True,timeout=50):
        started=asyncio.get_running_loop().time();stable=0
        while asyncio.get_running_loop().time()-started<timeout:
            native=await self._refresh(s)
            if allow_stop and s['_stop']:return False
            distance=math.dist([s['pose'][k] for k in ('x_m','y_m','altitude_m')],target)
            velocity=native.get('velocity',{})
            if not all(isinstance(velocity.get(k),(int,float)) and math.isfinite(velocity[k]) for k in ('x_m_s','y_m_s','z_m_s')):raise RuntimeError('Native velocity unavailable for arrival verification')
            speed=math.sqrt(sum(velocity[k]**2 for k in ('x_m_s','y_m_s','z_m_s')))
            stable=stable+1 if distance<.65 and speed<.4 else 0
            if stable>=3:return True
            await self._call('tick',.2);await asyncio.sleep(.08)
        raise RuntimeError('Native drone did not reach commanded position within timeout')
    async def _move(self,s,x,y,z,allow_stop=True):
        if not -24<=x<=24 or not -24<=y<=24 or not 2<=z<=10:raise ValueError('Action outside mission geofence')
        # Route via home: the four public inspection areas do not overlap either
        # home-to-area corridor, so excluded areas are never crossed.
        p=s['pose']
        if p and math.hypot(p['x_m']-x,p['y_m']-y)>2 and abs(p['x_m'])+abs(p['y_m'])>1 and abs(x)+abs(y)>1:
            await self._call('set_target',0,0,z)
            if not await self._wait(s,[0,0,z],allow_stop):return False
        await self._call('set_target',x,y,z)
        s['plan'].setdefault('waypoints',[]).append(dict(x_m=x,y_m=y,altitude_m=z))
        return await self._wait(s,[x,y,z],allow_stop)
    def _boxes(self,s,row,detections,model):
        from .label_schema import normalize_thermal_output
        from .evaluation import valid_box
        output,trace=normalize_thermal_output({'detections':detections});valid=[]
        for d in output['detections'][:32]:
            if not isinstance(d,dict) or not valid_box(d.get('bbox')):continue
            confidence=d.get('confidence')
            if isinstance(confidence,(int,float)) and not isinstance(confidence,bool) and math.isfinite(confidence) and 0<=confidence<=1:valid.append(d)
        row.update(boxes=[d for d in valid if d['confidence']>=.6],uncertain_boxes=[d for d in valid if d['confidence']<.6],detections=valid,model=model,normalization=trace,rejected_detection_count=len(detections)-len(valid),analysis_status='partial_invalid' if len(detections)!=len(valid) else 'analyzed')
        path=self.root/row['image'];im=Image.open(path).convert('RGB');draw=ImageDraw.Draw(im)
        for d in valid:
            b=d['bbox'];coords=[b['x_min']*im.width/1000,b['y_min']*im.height/1000,b['x_max']*im.width/1000,b['y_max']*im.height/1000];color='#8cdb9c' if d['confidence']>=.6 else '#eac17a'
            draw.rectangle(coords,outline=color,width=2);draw.text((coords[0]+2,coords[1]+2),f"{d['confidence']:.2f}",fill=color)
        target=path.with_name(path.stem+'-annotated.png');im.save(target);row['annotated_image']=target.relative_to(self.root).as_posix();self._save(s)
        from .mission_evidence import record_observation_evidence,context_from_capture
        receipt=row['capture']
        receipt.setdefault('camera',{k:v for k,v in receipt.get('intrinsics',{}).items() if k in ('fx_px','fy_px','cx_px','cy_px')})
        background=receipt.get('recorded_background_appearance')
        appearance=dict(kind='recorded_texture',source_sha256=background['source_sha256'],source_kind=background.get('kind','recorded_thermal_crop'),screening_status=background.get('screening_status','teacher_screened_not_independently_target_free')) if isinstance(background,dict) and background.get('source_sha256') else None
        context=context_from_capture(path,receipt,mission_id=s['id'],observation_id=row['id'],source_kind=receipt.get('source_kind','gazebo_camera'),appearance=appearance)
        local=model=='local_thermal_contrast'
        detection=dict(boxes=row['boxes'],uncertain_boxes=row['uncertain_boxes'],model=model,provenance='pixel_only_heuristic_not_validated_mine_detection' if local else 'caller_declared_multimodal_agent_camera_analysis',paid_request_issued=not local)
        evidence=record_observation_evidence(path,context,detection,output_dir=path.parent/'evidence')
        row.setdefault('evidence_history',[]).append(evidence['evidence_id']);row['evidence']=evidence;self._save(s)
    async def _capture(self,s,area):
        await self._refresh(s)
        if s['pose']['altitude_m']<1.5:raise RuntimeError('Ground frames are excluded from mission evidence')
        identifier=uuid.uuid4().hex[:12];path=self.folder/s['id']/'observations'/f'{identifier}.png'
        if hasattr(self.bridge,'capture_pair'):
            pair=await self._call('capture_pair',path.parent,identifier);receipt=pair['thermal']
        else:pair=None;receipt=await self._call('capture',path,'thermal')
        with Image.open(path) as im:
            if im.mode not in ('L','RGB','RGBA','P'):raise ValueError('Camera must provide explicitly mapped 8-bit display pixels')
        row=dict(id=identifier,area_id=area,source_kind=receipt.get('source_kind','gazebo_camera'),image=path.relative_to(self.root).as_posix(),pose=copy.deepcopy(s['pose']),time_s=s['budget']['elapsed_s'],capture=receipt,boxes=[],uncertain_boxes=[])
        if pair:
            row.update(rgb_image=Path(pair['rgb']['path']).relative_to(self.root).as_posix(),rgb_capture=pair['rgb'],paired_capture_id=receipt['pair_id'])
        overview=path.with_name(identifier+'-overview.png')
        row['overview_capture']=await self._call('capture',overview,'overview')
        row['overview_image']=overview.relative_to(self.root).as_posix()
        # Adapter capture metadata binds the camera timestamp and native pose.
        if isinstance(receipt.get('pose'),dict):
            cp=receipt['pose'];row['capture_pose']=cp
            row['pose']=dict(x_m=cp['x_m'],y_m=cp['y_m'],altitude_m=cp.get('z_m',cp.get('altitude_m')),yaw_deg=math.degrees(cp.get('yaw_rad',0)))
        s['observations'].append(row)
        from .mission_evidence import local_thermal_proposals
        proposal=await asyncio.to_thread(self.detector or local_thermal_proposals,path)
        row['reference_proposals']=proposal
        self._boxes(s,row,proposal.get('boxes',[])+proposal.get('uncertain_boxes',[]),proposal.get('model','local_thermal_contrast'))
        self._event(s,'capture',f'Paired thermal and RGB observation of {area}' if pair else f'Actual Gazebo thermal observation of {area}');return row
    async def _decide(self,s,provider):
        public=dict(prompt=s['prompt'],plan=s['plan'],pose=s['pose'],budget=s['budget'],areas=s['areas'],observations=[{k:o.get(k) for k in ('id','area_id','pose','reference_proposals')} for o in s['observations']])
        instruction='''Manage this simulated drone investigation. Use only public observations and the latest attached camera image. Return one JSON object: {"action":"inspect_area"|"revisit"|"return_home","area_id":"east"|"west"|"north"|"south","altitude_m":2..10,"reason":"brief operational explanation","detections":[{"bbox":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000},"confidence":0..1,"evidence":"visible evidence"}]}. Detections apply only to the latest attached image; omit if no image. Include subtle plausible candidates with appropriately lower scores. Respect allowed_areas, revisit preference and return budget. Never inspect an area more than three times. After sufficient inspection return_home. Coordinates are image coordinates normalized 0..1000. Do not output private reasoning. Public mission state: '''
        instruction+=' If two images are attached, image 1 is white-hot physics-derived thermal; image 2 is the native RGB context at the same pose and timestamp. All detection boxes refer only to image 1. The ground may contain buried targets with no exposed RGB object. '
        prompt=instruction+json.dumps(public);save(self.folder/s['id']/'agent'/f'{len(s["events"]):03d}-prompt.json',{'prompt':prompt,'prompt_version':'gazebo-paired-physics-v2'})
        images=[]
        if s['observations']:
            last=s['observations'][-1];images=[self.root/last['image']]
            if last.get('rgb_image'):images.append(self.root/last['rgb_image'])
        try:
            await self._world_pause(True)
            result=await provider.generate('gazebo_mission_decision',prompt,images=images,model=self._agent_model(),max_output_tokens=1536)
        finally:
            await self._world_pause(False)
        save(self.folder/s['id']/'agent'/f'{len(s["events"]):03d}-response.json',result)
        if not isinstance(result,dict) or result.get('action') not in ('inspect_area','revisit','return_home'):raise ValueError('Agent returned an invalid bounded action')
        if s['observations'] and isinstance(result.get('detections'),list):self._boxes(s,s['observations'][-1],result['detections'],self._agent_model())
        self._event(s,result['action'],str(result.get('reason',''))[:500]);return result
    async def _return(self,s):
        s['status']='returning';self._event(s,'return_home','Return to simulated launch position and land')
        await self._refresh(s);z=max(2,min(5,s['pose']['altitude_m']))
        await self._move(s,0,0,z,False);await self._call('land');await self._wait(s,[0,0,.15],False)
    def _report(self,s):
        findings=[dict(observation_id=o['id'],area_id=o['area_id'],image=o['annotated_image'],detections=o.get('detections',[])) for o in s['observations'] if o.get('detections')]
        report=dict(summary=f"{len(s['observations'])} camera observations collected; {len(findings)} observations contain provisional proposals.",findings=findings,controller=s['controller'],budget=s['budget'],returned_home=bool(s['pose'] and math.hypot(s['pose']['x_m'],s['pose']['y_m'])<1 and s['pose']['altitude_m']<.8),limitations=['Thermal images compose experimental DuMux responses with empirical ground texture; full numerical and field validation remain pending.','RGB is native Gazebo imagery using a separate photographic terrain material.','Proposals are image evidence; physical mine identities and geographic coordinates are unverified.','Local contrast scores are uncalibrated. Cross-frame proposals have not been merged into physical identities.'])
        from .mission_evidence import build_mission_report
        visited=sorted({o['area_id'] for o in s['observations']});requested=s['plan']['allowed_areas']
        report['coverage']=dict(measure='Requested regions with at least one observation; not complete ground coverage',qualified_ground_coverage_available=False,requested_areas=requested,inspected_areas=visited,unvisited_areas=[a for a in requested if a not in visited],area_fraction=len(visited)/len(requested),all_requested_areas_inspected=set(requested)<=set(visited),within_flight_budget=s['budget']['elapsed_s']<=s['budget']['limit_s'])
        report['evidence_report']=build_mission_report(s['id'],[o['evidence'] for o in s['observations'] if o.get('evidence')])
        s['report']=report;save(self.folder/s['id']/'report.json',report);self._save(s)
    async def _run(self,s):
        provider=None
        try:
            if s['controller']=='agent':
                if self.provider_factory:provider=self.provider_factory()
                elif os.environ.get('AEROROZUM_MISSION_PROVIDER')=='local_gemma':
                    from .local_mission_provider import LocalMissionProvider
                    provider=LocalMissionProvider(self.folder/s['id']/'agent/provider')
                else:
                    from .providers import GeminiProvider
                    cfg={**config()['gemini'],'requests_per_minute':24,'input_tokens_per_minute':200000,'concurrency':1,'max_requests':16,'max_retries':2,'thinking_budget':0}
                    provider=GeminiProvider(cfg,self.folder/s['id']/'agent/provider')
            await self._world_pause(False);await self._refresh(s);s['status']='starting'
            await self._call('takeoff',s['plan']['altitude_m']);await self._wait(s,[0,0,s['plan']['altitude_m']]);s['status']='running'
            visits={};queue=list(s['plan']['allowed_areas']);revisited=False
            for _ in range(14):
                await self._refresh(s)
                if s['_stop'] or s['budget']['remaining_s']<25:break
                if provider:action=await self._decide(s,provider)
                elif queue:action=dict(action='inspect_area',area_id=queue.pop(0),altitude_m=s['plan']['altitude_m'])
                elif s['plan']['revisit_uncertain'] and not revisited and any(o.get('detections') for o in s['observations']):
                    revisited=True;action=dict(action='revisit',area_id=next(o['area_id'] for o in s['observations'] if o.get('detections')),altitude_m=max(2,s['plan']['altitude_m']-2))
                else:break
                if action['action']=='return_home':break
                area_id=action.get('area_id')
                if area_id not in s['plan']['allowed_areas']:raise ValueError('Agent action violates requested area constraints')
                if visits.get(area_id,0)>=3:break
                area=next(a for a in AREAS if a['id']==area_id);z=float(action.get('altitude_m',s['plan']['altitude_m']))
                self._event(s,action['action'],f"Inspect {area_id} at {z:g}m")
                if not await self._move(s,area['center']['x_m'],area['center']['y_m'],z):break
                await self._capture(s,area_id);visits[area_id]=visits.get(area_id,0)+1
            await self._return(s);s['status']='stopped' if s['_stop'] else 'completed';self._report(s)
        except Exception as error:
            s['error']=redacted_error(error);self._event(s,'mission_error',s['error'])
            try:await self._return(s)
            except Exception as recovery:s['recovery_error']=redacted_error(recovery)
            s['status']='failed';self._report(s)
        finally:
            try:
                if provider and hasattr(provider,'close'):await provider.close()
            finally:self.active=None;self._save(s)
