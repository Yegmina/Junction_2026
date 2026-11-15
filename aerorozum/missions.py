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
    match=re.search(r'(?:(?:altitude|height|z)(?:_m)?\s*(?:of\s*|[=:]\s*)?|at\s+)(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)?\b',text)
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
        plan['navigation_mode']='free_coordinates' if controller=='agent' else 'reference_regions'
        if controller not in ('agent','reference'):raise ValueError('Controller must be agent or reference')
        if physics_phase not in ('warming','cooling'):raise ValueError('Thermal phase must be warming or cooling')
        if self.active:raise ValueError('Another simulated mission is active')
        sim=await asyncio.to_thread(self.simulation_status)
        if not sim['available']:raise ValueError('Native Gazebo is not ready')
        if controller=='agent' and not sim['agent_enabled']:raise ValueError('Agent API access is not configured')
        if hasattr(self.bridge,'set_thermal_phase'):await self._call('set_thermal_phase',physics_phase)
        identifier=uuid.uuid4().hex[:16]
        state=dict(id=identifier,prompt=prompt,controller=controller,seed=seed,physics_phase=physics_phase,world_seed_applied=False,status='planning',created_at=utcnow(),plan=plan,simulation=sim,map_bounds=BOUNDS,areas=AREAS,pose=None,route=[],observations=[],events=[],agent_steps=[],agent_phase='starting',active_target=None,navigation={},budget=dict(limit_s=plan['budget_s'],elapsed_s=0,remaining_s=plan['budget_s']),report=None,error=None,_stop=False,_start_sim=None,_inference_s=0)
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
        if s.get('active_target'):
            target=s['active_target'];distance=math.dist([pose[k] for k in ('x_m','y_m','altitude_m')],[target[k] for k in ('x_m','y_m','altitude_m')])
            speed=math.sqrt(sum(float(native.get('velocity',{}).get(k,0))**2 for k in ('x_m_s','y_m_s','z_m_s')))
            s['navigation']=dict(target=target,distance_m=round(distance,2),status='arrived' if distance<.65 and speed<.4 else 'moving',speed_m_s=round(speed,2))
        if not s['route'] or sum((pose[k]-s['route'][-1][k])**2 for k in ('x_m','y_m','altitude_m'))>.01:s['route'].append(dict(**pose,time_s=round(elapsed,2)))
        self._save(s);return native
    async def _wait(self,s,target,allow_stop=True,timeout=50):
        started=asyncio.get_running_loop().time();start_sim=None;stable=0
        while asyncio.get_running_loop().time()-started<max(60,timeout*8):
            native=await self._refresh(s)
            if start_sim is None:start_sim=float(native['sim_time_s'])
            if float(native['sim_time_s'])-start_sim>timeout:break
            if allow_stop and s['_stop']:return False
            distance=math.dist([s['pose'][k] for k in ('x_m','y_m','altitude_m')],target)
            velocity=native.get('velocity',{})
            if not all(isinstance(velocity.get(k),(int,float)) and math.isfinite(velocity[k]) for k in ('x_m_s','y_m_s','z_m_s')):raise RuntimeError('Native velocity unavailable for arrival verification')
            speed=math.sqrt(sum(velocity[k]**2 for k in ('x_m_s','y_m_s','z_m_s')))
            stable=stable+1 if distance<.65 and speed<.4 else 0
            if stable>=3 and (target[2]>=.5 or abs(s['pose']['altitude_m']-target[2])<.18):return True
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
        if s['controller']=='agent':s['agent_phase']='observing';self._save(s)
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
        if s['controller']=='agent':row['area_id']=self._area_at_pose({'pose':row['pose']})
        s['observations'].append(row)
        from .mission_evidence import local_thermal_proposals
        proposal=await asyncio.to_thread(self.detector or local_thermal_proposals,path)
        row['reference_proposals']=proposal
        self._boxes(s,row,proposal.get('boxes',[])+proposal.get('uncertain_boxes',[]),proposal.get('model','local_thermal_contrast'))
        self._event(s,'capture',f'Paired thermal and RGB observation of {area}' if pair else f'Actual Gazebo thermal observation of {area}');return row
    def _exclusions(self,s):
        return [area['bounds'] for area in AREAS if area['id'] in s['plan']['forbidden_areas']]
    def _area_at_pose(self,s):
        p=s['pose']
        return next((a['id'] for a in AREAS if a['bounds']['x_min']<=p['x_m']<=a['bounds']['x_max'] and a['bounds']['y_min']<=p['y_m']<=a['bounds']['y_max']),'transit')
    def _return_reserve(self,s):
        from .mission_navigation import return_route
        p=s['pose'];start=(p['x_m'],p['y_m']);route=return_route(start,(0.,0.),BOUNDS,self._exclusions(s))
        distance=sum(math.dist(a,b) for a,b in zip([start]+route,route))
        return max(20.,distance/1.2+10+2*max(0,len(route)-1))
    async def _decide(self,s,provider):
        observation=s['observations'][-1] if s['observations'] else None
        step=dict(id=uuid.uuid4().hex[:12],status='deciding',started_at=utcnow(),model=self._agent_model(),
                  input_pose=copy.deepcopy(s['pose']),observation_id=observation['id'] if observation else None)
        s['agent_steps'].append(step);s['agent_phase']='deciding';self._save(s)
        public=dict(prompt=s['prompt'],plan=s['plan'],pose=s['pose'],budget=s['budget'],geofence=BOUNDS,
                    coordinate_frame='Local ENU metres: x=east, y=north, altitude=height above the flat ground; home=(0,0)',
                    named_regions=AREAS,navigation=s.get('navigation',{}),return_reserve_s=self._return_reserve(s),
                    recent_tools=[{k:v for k,v in item.items() if k in ('tool','arguments','result','error','rationale')} for item in s['agent_steps'][-9:-1]],
                    observations=[{k:o.get(k) for k in ('id','area_id','pose','reference_proposals')} for o in s['observations'][-8:]])
        if observation:
            public['latest_image']=dict(id=observation['id'],pose=observation['pose'],intrinsics=observation['capture'].get('intrinsics',{}),camera_mount=observation['capture'].get('camera_mount',{}))
        instruction='You operate a simulated drone through tools in a live observation/action/feedback loop. The aircraft and flight clock CONTINUE while you review images. You may retarget it during motion. Use arbitrary safe local coordinates, not just region centres. Named regions are navigation landmarks; plan.forbidden_areas are exclusions. Follow the user\'s requested coordinates, ordering and height.\nReturn one JSON object: {"tool":"goto"|"observe"|"hover"|"return_home","arguments":{},"observation_summary":"brief visible/telemetry facts","rationale":"brief operational justification for this decision","next_check":"what evidence or arrival condition to review next","review_after_s":1.5,"observation_id":"attached image id","detections":[]}. These are concise user-facing decision summaries.\ngoto arguments: {"x_m":number,"y_m":number,"altitude_m":number,"yaw_deg":number(optional)}. Coordinates must be inside the supplied geofence; altitude 2..10m. A goto starts movement immediately; a subsequent goto can change course or height before arrival. Observe keeps the current destination while checking fresh evidence. Hover holds the current position. return_home ends the investigation and lands. review_after_s must be 0.5..5 simulated seconds. Use navigation.status to track whether the previous waypoint has been reached; avoid repeatedly resetting an unchanged destination. On tool rejection, correct the arguments using the error feedback. Return while the return reserve remains.\nImage 1 is white-hot thermal and image 2 is simultaneous native RGB. Boxes apply ONLY to image 1 and the supplied observation_id. Detections: [{"bbox":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000},"confidence":0..1,"evidence":"brief visible evidence"}]. Include subtle plausible candidates at appropriately lower scores. At most 10 detections. Public mission state: '
        prompt=instruction+json.dumps(public);folder=self.folder/s['id']/'agent'
        save(folder/(step['id']+'-prompt.json'),dict(prompt=prompt,prompt_version='coordinate-flight-v3',observation_id=step['observation_id']))
        images=[] if observation is None else [self.root/observation['image']]+([self.root/observation['rgb_image']] if observation.get('rgb_image') else [])
        task=asyncio.create_task(provider.generate('gazebo_mission_decision',prompt,images=images,model=self._agent_model(),max_output_tokens=1536))
        try:
            while True:
                done,_=await asyncio.wait({task},timeout=.2)
                await self._refresh(s)
                if s['_stop'] or s['budget']['remaining_s']<self._return_reserve(s):
                    if not task.done():task.cancel()
                    await asyncio.gather(task,return_exceptions=True)
                    step.update(status='cancelled',rationale='Controller requested return: stop instruction or return-time reserve.',finished_at=utcnow())
                    self._save(s);return dict(tool='return_home',arguments={},controller_guard=True)
                if done:break
            result=await task
            save(folder/(step['id']+'-response.json'),result)
            if not isinstance(result,dict):raise ValueError('Agent must return a JSON tool call')
            tool=result.get('tool',result.get('action'))
            step.update(tool=tool,arguments=result.get('arguments',{}),observation_summary=str(result.get('observation_summary',''))[:1000],
                        rationale=str(result.get('rationale',result.get('reason','')))[:1000],next_check=str(result.get('next_check',''))[:500],status='ready',decided_at=utcnow())
            if observation and isinstance(result.get('detections'),list):
                if result.get('observation_id',observation['id'])==observation['id']:
                    self._boxes(s,observation,result['detections'],self._agent_model())
                else:step['detection_warning']='Detection observation id did not match the attached frame; raw response retained.'
            self._save(s);return {**result,'tool':tool}
        except Exception as error:
            step.update(status='rejected',error=redacted_error(error));self._save(s);raise
        finally:
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
    async def _advance(self,s,duration):
        native=await self._refresh(s);start=float(native['sim_time_s'])
        deadline=asyncio.get_running_loop().time()+max(20,duration*15)
        while asyncio.get_running_loop().time()<deadline:
            native=await self._refresh(s)
            if s['_stop'] or s['budget']['remaining_s']<self._return_reserve(s):return False
            if float(native['sim_time_s'])-start>=duration:return True
            await self._call('tick',.2);await asyncio.sleep(.05)
        raise RuntimeError('Simulation clock stalled while executing the agent tool')
    async def _run_agent(self,s,provider):
        from .mission_navigation import validate_target
        await self._capture(s,self._area_at_pose(s))
        consecutive_errors=0
        for _ in range(16):
            await self._refresh(s)
            if s['_stop'] or s['budget']['remaining_s']<self._return_reserve(s):break
            step=None
            try:
                action=await self._decide(s,provider);step=s['agent_steps'][-1];tool=action.get('tool')
                if tool=='return_home':
                    step.update(tool='return_home',source='controller_guard' if action.get('controller_guard') else 'agent',status='executing',commanded_at=utcnow(),result=dict(status='returning',pose=copy.deepcopy(s['pose'])));break
                arguments=action.get('arguments',{})
                if not isinstance(arguments,dict):raise ValueError('Tool arguments must be a JSON object')
                duration=action.get('review_after_s',1.5)
                if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or not .5<=duration<=5:raise ValueError('review_after_s must be between 0.5 and 5 simulated seconds')
                if tool in ('inspect_area','revisit'):
                    area_id=action.get('area_id')
                    if area_id not in s['plan']['allowed_areas']:raise ValueError('Agent action violates requested area constraints')
                    area=next(a for a in AREAS if a['id']==area_id)
                    arguments=dict(x_m=area['center']['x_m'],y_m=area['center']['y_m'],altitude_m=action.get('altitude_m',s['plan']['altitude_m']))
                    tool='goto'
                if tool=='goto':
                    x,y,z=validate_target(arguments.get('x_m'),arguments.get('y_m'),arguments.get('altitude_m',arguments.get('z_m')),BOUNDS,self._exclusions(s),start=(s['pose']['x_m'],s['pose']['y_m']))
                    yaw=arguments.get('yaw_deg',0)
                    if isinstance(yaw,bool) or not isinstance(yaw,(int,float)) or not math.isfinite(yaw) or not -360<=yaw<=360:raise ValueError('yaw_deg must be a finite heading between -360 and 360')
                    changed=bool(s.get('active_target') and s.get('navigation',{}).get('status')=='moving' and any(abs(s['active_target'][k]-v)>.1 for k,v in zip(('x_m','y_m','altitude_m'),(x,y,z))))
                    await self._call('set_target',x,y,z,math.radians(yaw))
                    arguments=dict(x_m=x,y_m=y,altitude_m=z,yaw_deg=yaw)
                    s['active_target']=arguments;s['plan'].setdefault('waypoints',[]).append(dict(**arguments,command_id=step['id']))
                    step['adjusted_during_motion']=changed
                elif tool=='hover':
                    await self._call('hover');s['active_target']={k:s['pose'][k] for k in ('x_m','y_m','altitude_m')}
                elif tool!='observe':raise ValueError('Unknown tool. Use goto, observe, hover or return_home')
                step.update(tool=tool,arguments=arguments,status='executing',commanded_at=utcnow());s['agent_phase']='executing'
                self._event(s,'tool_call',dict(tool=tool,arguments=arguments,rationale=step.get('rationale','')))
                ongoing=await self._advance(s,duration)
                if ongoing:await self._capture(s,self._area_at_pose(s))
                await self._refresh(s)
                step.update(status='done',finished_at=utcnow(),result=dict(status=s.get('navigation',{}).get('status','observed'),pose=copy.deepcopy(s['pose']),navigation=copy.deepcopy(s.get('navigation',{})),remaining_s=s['budget']['remaining_s'],observation_id=s['observations'][-1]['id']))
                self._event(s,'tool_result',step['result']);consecutive_errors=0
                if not ongoing:break
            except (ValueError,RuntimeError) as error:
                consecutive_errors+=1
                if step is None and s['agent_steps']:step=s['agent_steps'][-1]
                if step is not None:step.update(status='rejected',error=redacted_error(error),finished_at=utcnow())
                self._event(s,'tool_rejected',redacted_error(error))
                await self._call('hover');s['active_target']={k:s['pose'][k] for k in ('x_m','y_m','altitude_m')}
                if consecutive_errors>=3:raise ValueError('Agent tool failed three times: '+str(error)) from error
        s['agent_phase']='returning';self._save(s)
    async def _return(self,s):
        s['status']='returning';self._event(s,'return_home','Return to simulated launch position and land')
        await self._refresh(s);z=max(2,min(5,s['pose']['altitude_m']))
        from .mission_navigation import return_route
        route=return_route((s['pose']['x_m'],s['pose']['y_m']),(0.,0.),BOUNDS,self._exclusions(s))
        for x,y in route:
            s['active_target']=dict(x_m=x,y_m=y,altitude_m=z)
            await self._call('set_target',x,y,z);await self._wait(s,[x,y,z],False)
        await self._call('land');await self._wait(s,[0,0,.15],False)
        s['active_target']=None;s['navigation']=dict(status='landed')
    def _report(self,s):
        findings=[dict(observation_id=o['id'],area_id=o['area_id'],image=o['annotated_image'],detections=o.get('detections',[])) for o in s['observations'] if o.get('detections')]
        report=dict(summary=f"{len(s['observations'])} camera observations collected; {len(findings)} observations contain provisional proposals.",findings=findings,controller=s['controller'],budget=s['budget'],returned_home=bool(s['pose'] and math.hypot(s['pose']['x_m'],s['pose']['y_m'])<1 and s['pose']['altitude_m']<.8),limitations=['Thermal images compose experimental DuMux responses with empirical ground texture; full numerical and field validation remain pending.','RGB is native Gazebo imagery using a separate photographic terrain material.','Proposals are image evidence; physical mine identities and geographic coordinates are unverified.','Local contrast scores are uncalibrated. Cross-frame proposals have not been merged into physical identities.'])
        from .mission_evidence import build_mission_report
        visited=sorted({o['area_id'] for o in s['observations'] if o['area_id'] in {a['id'] for a in AREAS}});requested=s['plan']['allowed_areas']
        if s['controller']=='agent':requested=[a for a in requested if re.search(r'\b'+a+r'\b',s['prompt'].lower())]
        report['coverage']=dict(measure='Named regions with a camera observation; arbitrary waypoint flights are reported separately',qualified_ground_coverage_available=False,requested_areas=requested,inspected_areas=visited,unvisited_areas=[a for a in requested if a not in visited],area_fraction=len(set(visited)&set(requested))/len(requested) if requested else None,all_requested_areas_inspected=set(requested)<=set(visited) if requested else None,within_flight_budget=s['budget']['elapsed_s']<=s['budget']['limit_s'])
        report['agent_steps']=s.get('agent_steps',[]);report['commanded_waypoints']=s['plan'].get('waypoints',[])
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
            if provider:
                await self._run_agent(s,provider)
            else:
                visits={};queue=list(s['plan']['allowed_areas']);revisited=False
                for _ in range(14):
                    await self._refresh(s)
                    if s['_stop'] or s['budget']['remaining_s']<25:break
                    if queue:action=dict(action='inspect_area',area_id=queue.pop(0),altitude_m=s['plan']['altitude_m'])
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
            await self._return(s)
            if s.get('agent_steps') and s['agent_steps'][-1].get('tool')=='return_home':
                s['agent_steps'][-1].update(status='done',finished_at=utcnow(),result=dict(status='landed',pose=copy.deepcopy(s['pose']),remaining_s=s['budget']['remaining_s']))
            s['status']='stopped' if s['_stop'] else 'completed';self._report(s)
        except Exception as error:
            s['error']=redacted_error(error);self._event(s,'mission_error',s['error'])
            try:await self._return(s)
            except Exception as recovery:s['recovery_error']=redacted_error(recovery)
            s['status']='failed';self._report(s)
        finally:
            try:
                if provider and hasattr(provider,'close'):await provider.close()
            finally:s['agent_phase']=s['status'];self.active=None;self._save(s)
