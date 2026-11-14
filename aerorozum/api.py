"""Local dashboard adapter. Read endpoints never start paid work."""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = re.compile(r'(secret|password|credential|api[_-]?key|access[_-]?token|refresh[_-]?token|key[_-]?file)', re.I)

def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items() if not SENSITIVE.search(str(k))}
    if isinstance(value, list): return [safe(v) for v in value]
    if isinstance(value, Path): return value.name
    if isinstance(value, str):
        if value.startswith(('/home/', '/media/', '/tmp/')): return Path(value).name
        return re.sub(r'AIza[\w-]{20,}', '[redacted]', value)
    return value

def read_json(path: Path, default=None):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default

class CreateCampaign(BaseModel):
    seed: int = 20261114
    config: dict[str, Any] = Field(default_factory=dict)

class Action(BaseModel):
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class CreateMission(BaseModel):
    prompt: str = Field(min_length=4,max_length=2000)
    controller: str = 'agent'
    seed: int = 0
    physics_phase: str = 'warming'


def create_app(data_root: Path | str | None = None, campaign_factory=None, evaluator_enabled: bool | None = None) -> FastAPI:
    app = FastAPI(title='AeroRozum Thermal Research Lab', version='0.1.0')
    app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:18900', 'http://localhost:18900'], allow_methods=['GET','POST'], allow_headers=['Content-Type','X-AeroRozum-Evaluator','Authorization'])
    external = Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/artifacts')
    app.state.data_root = Path(data_root or os.environ.get('AEROROZUM_DATA_ROOT', external if external.parent.exists() else ROOT/'artifacts')).resolve()
    app.state.campaigns = {}
    app.state.locks = {}
    app.state.evaluator_enabled = evaluator_enabled if evaluator_enabled is not None else os.environ.get('AEROROZUM_ENABLE_EVALUATOR') == '1'
    app.state.campaign_factory = campaign_factory
    from .missions import MissionManager
    app.state.missions = MissionManager(app.state.data_root)

    @app.get('/api/missions')
    async def missions():
        simulation=await run_in_threadpool(app.state.missions.simulation_status)
        return safe(dict(missions=app.state.missions.list(),simulation=simulation))

    @app.get('/api/mission-eval')
    def mission_eval():
        jobs=[]
        for condition,run in [('paired','paired-decisions-v1'),('blank','paired-decisions-blank-v1')]:
            folder=app.state.data_root/'mission-eval'/run/'results'
            for gpu,name in enumerate(('base','real_sft','real_grpo','stream_grpo')):
                status=read_json(folder/name/'status.json',{}) or {}
                if status:jobs.append(dict(gpu=gpu,condition=condition,**status,summary=read_json(folder/name/'summary.json')))
        return safe(dict(jobs=jobs,total_cases_per_checkpoint=128,scope='Offline routing and instruction-contract probes; not full mission completion or field detection accuracy.'))

    @app.post('/api/missions')
    async def create_mission(body: CreateMission):
        try:return safe(await app.state.missions.create(body.prompt,body.controller,body.seed,physics_phase=body.physics_phase))
        except ValueError as error:raise HTTPException(409,str(error))

    @app.get('/api/missions/camera/{kind}')
    async def mission_camera(kind: str):
        if kind not in ('overview','thermal','rgb'):raise HTTPException(404,'Unknown simulator camera')
        try:
            pixels=await run_in_threadpool(app.state.missions.bridge.read_frame,kind)
            return Response(pixels,media_type='image/png',headers={'Cache-Control':'no-store'})
        except RuntimeError:raise HTTPException(503,'Native simulator camera unavailable')

    @app.get('/api/missions/{identifier}')
    def get_mission(identifier: str):
        try:return safe(app.state.missions.get(identifier))
        except KeyError:raise HTTPException(404,'Mission not found')

    @app.post('/api/missions/{identifier}/stop')
    async def stop_mission(identifier: str):
        try:return safe(await app.state.missions.stop(identifier))
        except KeyError:raise HTTPException(404,'Mission not found')

    def campaign(identifier):
        if identifier not in app.state.campaigns: raise HTTPException(404,'Campaign not found')
        return app.state.campaigns[identifier]

    def external_campaigns():
        result={}
        root=app.state.data_root
        for path in sorted((root/'campaigns').rglob('state.json')):
            state=read_json(path,{}) or {}
            trajectory=state.get('trajectory',[])
            if not trajectory:continue
            public=trajectory[-1].get('result',{}).get('status')
            if not isinstance(public,dict) or not public.get('id'):continue
            public=json.loads(json.dumps(public))
            public.update(read_only=True,controller=state.get('model','External agent'),run_status=state.get('status'),last_action=trajectory[-1].get('action'))
            public['observations']=[]
            for observation in trajectory[-1]['result']['status'].get('observations',[]):
                target=(path.parent/'artifacts'/observation.get('image','')).resolve()
                if target.is_relative_to(root) and target.is_file():
                    public['observations'].append({**observation,'image':target.relative_to(root).as_posix()})
            result[public['id']]=public
        return result

    def research_status():
        config = read_json(ROOT/'configs/research.json', {})
        root = app.state.data_root
        snapshot = read_json(root/'status.json', {}) or {}
        jobs = snapshot.get('jobs')
        if jobs is None: jobs = [read_json(p,{}) for p in sorted((root/'jobs').glob('*.json'))][-50:]
        audit = snapshot.get('dataset') or read_json(root/'audits/dataset.json') or read_json(root/'dataset_audit.json')
        models = snapshot.get('models') or [
            {'id':config.get('primary_model','google/gemma-4-12B-it'),'role':'Primary trainable model','status':'planned'},
            {'id':config.get('reference_model','google/gemma-4-31B-it'),'role':'Reference model','status':'planned'},
            {'id':'Classified model number 2.0','role':'Independent API reviewer','status':'not_configured'},
        ]
        return safe({'project':'AeroRozum','experiment':config.get('experiment'),'counts':config.get('counts',{}),'budget_eur':config.get('google_cloud_budget_eur',1000),'campaigns':[c.status() for c in app.state.campaigns.values()]+list(external_campaigns().values()), 'dataset':audit,'jobs':jobs,'providers':snapshot.get('providers',[]),'models':models,'physics':snapshot.get('physics',{'solver':'DuMux 3.10.0','status':'planned','validation':'No accepted physical field loaded'}),'training':snapshot.get('training',{'status':'not_started','metrics':[]}), 'sponsors':snapshot.get('sponsors', []),'evaluator_available':app.state.evaluator_enabled})

    @app.get('/api/status')
    def status(): return research_status()

    @app.get('/api/pipeline')
    def pipeline_status():
        from .pipeline_view import snapshot
        return safe(snapshot(app.state.data_root))

    @app.get('/api/labels/recent')
    def recent_labels():
        folder=app.state.data_root/'labelling/robotics-er2-5fps/labels'
        recent=read_json(folder/'recent.json',[])
        if isinstance(recent,dict):recent=recent.get('recent',recent.get('images',[]))
        priority_folder=app.state.data_root/'labelling/robotics-er2-airborne-priority/labels'
        priority=read_json(priority_folder/'recent.json',{})
        priority=priority.get('images',[]) if isinstance(priority,dict) else priority
        combined=[{**row,'channel':'airborne_priority'} for row in priority[-4:]]+[{**row,'channel':'main'} for row in recent[-8:]] if recent else priority[-12:]
        candidates=read_json(priority_folder/'positives.json',{}).get('images',[])
        public_fields={'id','image','annotated','boxes','uncertain_boxes','source','timestamp_s','source_offset_s','frame_index','split','user_threshold','labelled_at','label_prompt_version','label_prompt_sha256'}
        examples=[{**{k:v for k,v in row.items() if k in public_fields},'channel':'saved_candidate_example','cloud_sync_status':row.get('cloud_sync',{}).get('status','unknown')} for row in candidates[-4:] if row.get('split')=='train']
        return safe({'progress':read_json(folder/'progress.json',{'status':'not_started'}),'priority_progress':read_json(priority_folder/'progress.json',{}),'recent':combined,'accepted_examples':examples})

    @app.post('/api/campaigns')
    def create(body: CreateCampaign):
        factory = app.state.campaign_factory
        if factory is None:
            try:
                from .environment import Campaign
                factory = Campaign
            except ImportError as ex: raise HTTPException(503,'Environment module is not ready') from ex
        cfg = read_json(ROOT/'configs/research.json', {})
        defaults = {**cfg.get('campaign',{}), 'camera':cfg.get('camera',{})}
        allowed = {'width_m','height_m','max_sorties','sortie_seconds','max_actions','turnaround_seconds'}
        defaults.update({k:v for k,v in body.config.items() if k in allowed})
        c = factory(config=defaults, seed=body.seed, output_dir=app.state.data_root/'campaigns')
        s = c.status(); identifier = s.get('id') or getattr(c,'id',None)
        if not identifier: raise HTTPException(500,'Environment did not return a campaign identifier')
        app.state.campaigns[identifier] = c
        app.state.locks[identifier] = threading.RLock()
        return safe(s)

    @app.get('/api/campaigns/{identifier}')
    def get(identifier: str):
        if identifier in app.state.campaigns:return safe(campaign(identifier).status())
        external=external_campaigns().get(identifier)
        if external:return safe(external)
        raise HTTPException(404,'Campaign not found')

    @app.post('/api/campaigns/{identifier}/actions')
    async def act(identifier: str, body: Action):
        c = campaign(identifier)
        def work():
            with app.state.locks[identifier]: return safe(c.act(body.action, body.arguments))
        return await run_in_threadpool(work)

    @app.get('/api/campaigns/{identifier}/evaluator')
    def evaluator(identifier: str, request: Request):
        if not app.state.evaluator_enabled or request.headers.get('X-AeroRozum-Evaluator') != '1':
            raise HTTPException(403,'Evaluator view is disabled; enable it explicitly on the local server')
        token = os.environ.get('AEROROZUM_EVALUATOR_TOKEN')
        if token and request.headers.get('Authorization') != f'Bearer {token}': raise HTTPException(403,'Evaluator authorization required')
        c = campaign(identifier)
        if not hasattr(c,'evaluator_snapshot'): raise HTTPException(503,'Evaluator is not available')
        return safe(c.evaluator_snapshot())

    @app.get('/api/events')
    async def events(request: Request):
        async def stream():
            previous = None
            while not await request.is_disconnected():
                payload = json.dumps(research_status(), separators=(',',':'), default=str)
                digest = hashlib.sha256(payload.encode()).hexdigest()
                if digest != previous:
                    yield f'event: status\ndata: {payload}\n\n'; previous = digest
                else: yield ': keepalive\n\n'
                await asyncio.sleep(2)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.get('/artifacts/{relative_path:path}')
    def artifact(relative_path: str):
        root = app.state.data_root
        target = (root/relative_path).resolve()
        try: target.relative_to(root)
        except ValueError: raise HTTPException(403,'Artifact path outside data root')
        parts = [p.lower() for p in (*Path(relative_path).parts,*target.relative_to(root).parts)]
        if any(p in ('private','truth','ground_truth','evaluator') for p in parts): raise HTTPException(403,'Private artifact')
        pipeline_media=target.is_relative_to(root/'pipeline/media')
        presentation_media=target.is_relative_to(root/'presentation')
        public_media=('observations' in parts or ('labelling' in parts and ('frames' in parts or 'annotated' in parts)) or pipeline_media or presentation_media)
        if target.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.mp4'} or not public_media:
            raise HTTPException(403,'Only public observation media can be served')
        if not target.is_file(): raise HTTPException(404,'Artifact not found')
        return FileResponse(target)

    @app.get('/presentation/')
    def presentation():
        target=ROOT/'docs/presentation/index.html'
        if not target.exists():raise HTTPException(404,'Presentation snapshot has not been generated')
        return FileResponse(target,media_type='text/html')

    @app.get('/presentation/{relative_path:path}')
    def presentation_asset(relative_path: str):
        root=(ROOT/'docs/presentation').resolve();target=(root/relative_path).resolve()
        try:target.relative_to(root)
        except ValueError:raise HTTPException(403,'Invalid presentation path')
        if target.suffix.lower() not in {'.png','.jpg','.svg','.md','.html','.json','.pdf','.mp4','.zip'} or not target.is_file():raise HTTPException(404,'Presentation artifact not found')
        return FileResponse(target)

    dist = ROOT/'frontend/dist'
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount('/',StaticFiles(directory=dist,html=True),name='dashboard')
    return app

app = create_app()
