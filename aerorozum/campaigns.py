"""Resumable policies using only public camera observations and campaign tools."""
from __future__ import annotations
import asyncio
import copy
import json
import hashlib
import marshal
from pathlib import Path
import re
from .common import config as load_config, data_root, digest, redacted_error, save, utcnow
from .environment import Campaign
from .renderer import ThermalRenderer

ACTIONS = ("launch","move_to","capture","inspect_observation","inspect_crop","wait_until","land","record_candidate","submit_report")
INSTRUCTIONS = '''Operate a simulated thermal SURVEY with public tools. Heat or circularity alone never confirms a mine. Collect useful observations, reject clutter, and merge repeated sightings when supported. Respect the clock, battery return reserve, turnaround and six-sortie allocation. Ground waiting changes the thermal day without flight battery use. Never invent geographic coordinates or change configuration. Images are the LAST THREE actually captured frames; image_order maps them to observation IDs. Boxes are named normalized coordinates {x_min,y_min,x_max,y_max} in 0..1000.
Return ONE JSON object {"action":"...","arguments":{},"evidence_note":"brief visible evidence"}; no private chain-of-thought.
Tools: launch {altitude_m:2..30}; move_to {x_m,y_m,altitude_m,yaw_deg}; capture {}; inspect_observation {observation_id}; inspect_crop {observation_id,box}; wait_until {time_s:seconds since midnight}; land {}; record_candidate {candidate_id:optional existing ID,evidence:[{observation_id,box}],confidence:0..1,notes:brief visible evidence}; submit_report {}.
Only capture creates a new physical observation; crops cannot add physical resolution. Survey morning, heating and cooling phases when budgets permit. Empty candidate lists are valid. Submit when collection is complete.'''

def _implementation_hash():
    # Hash loaded code, not mutable source files: a running process retains its
    # implementation even while another worker changes the next revision.
    functions=(Campaign._act,Campaign._advance,Campaign._capture,Campaign._mark_coverage,
        ThermalRenderer.temperature,ThermalRenderer.render,ThermalRenderer.ground_grid,
        ThermalRenderer.projected_box,ThermalRenderer.evidence_targets)
    return hashlib.sha256(b''.join(marshal.dumps(f.__code__) for f in functions)).hexdigest()

def _restore(state, root):
    if state.get('implementation_hash') and state['implementation_hash']!=_implementation_hash():
        raise ValueError('Simulator implementation changed; preserve this run and start a new revision')
    campaign = Campaign(state['campaign_config'],seed=state['seed'],output_dir=Path(root)/'artifacts')
    cid = state['campaign_id']
    if not isinstance(cid,str) or not re.fullmatch(r'[a-f0-9]{24}',cid):
        raise ValueError('Unsafe saved campaign ID')
    campaign.id=cid;campaign._dir=campaign.output_dir/cid
    campaign._dir.mkdir(parents=True,exist_ok=True)
    for item in state['trajectory']:
        result=campaign.act(item['action'],item['arguments'])
        if result['ok']!=item['result']['ok'] or result['status']['time_s']!=item['result']['status']['time_s']:
            raise ValueError('Saved actions no longer replay deterministically')
    return campaign

def _prompt(campaign, trajectory):
    status=campaign.status();recent=status['observations'][-3:]
    context={'status':{**status,'observations':status['observations'][-8:]},
        'image_order':[{'image_number':i+1,'observation_id':o['id'],'time_s':o['time_s'],'pose':o['pose']} for i,o in enumerate(recent)],
        'recent_actions':[{'action':t['action'],'arguments':t['arguments'],'ok':t['result']['ok'],'error':t['result'].get('error')} for t in trajectory[-6:]]}
    images=[]
    for observation in recent:
        path=(campaign.output_dir/observation['image']).resolve()
        if not path.is_relative_to(campaign.output_dir):raise ValueError('Unsafe observation path')
        images.append(path)
    return INSTRUCTIONS+'\nPublic context: '+json.dumps(context,allow_nan=False),images

def _decode(output):
    if not isinstance(output,dict) or set(output)-{'action','arguments','evidence_note'}:
        raise ValueError('Policy output does not match bounded action schema')
    action=output.get('action');arguments=output.get('arguments',{})
    if action not in ACTIONS or not isinstance(arguments,dict):raise ValueError('Invalid policy action')
    json.dumps(arguments,allow_nan=False)
    return action,arguments,str(output.get('evidence_note',''))[:2000]

def _persist(root,state):
    save(root/'state.json',state)
    temp=root/'.trajectory.jsonl.tmp'
    temp.write_text(''.join(json.dumps(t,allow_nan=False)+'\n' for t in state['trajectory']))
    temp.replace(root/'trajectory.jsonl')

async def run_campaign(cfg=None,seed=0,run_dir=None,provider=None,model=None,max_actions=None,resume=True,allow_demo=False):
    """Never requests evaluator state. evaluate_run is a separate offline step."""
    cfg=copy.deepcopy(cfg or load_config())
    maximum=int(max_actions if max_actions is not None else cfg.get('campaign',{}).get('max_actions',128))
    if not 1<=maximum<=int(cfg.get('campaign',{}).get('max_actions',128)):
        raise ValueError('Requested allowance exceeds configured action budget')
    model=model or cfg.get('gemini',{}).get('teacher_model')
    if not model:raise ValueError('Campaign model must be configured')
    root=Path(run_dir or data_root()/'campaigns'/digest([seed,model,utcnow()])[:20]).resolve();root.mkdir(parents=True,exist_ok=True)
    cc=copy.deepcopy(cfg);cc.setdefault('campaign',{})['max_actions']=maximum
    identity=digest({'config':cc,'seed':int(seed),'model':model,'max_actions':maximum})
    path=root/'state.json'
    if path.exists():
        if not resume:raise ValueError('Run exists; choose another run directory')
        state=json.loads(path.read_text())
        if state.get('identity')!=identity:raise ValueError('Resume configuration, seed or model differs')
        campaign=_restore(state,root)
        if state['status']=='completed':
            return {'status':'completed','run_dir':str(root),'actions':len(state['trajectory']),'campaign':campaign.status(),'resumed':True}
    else:
        campaign=Campaign(cc,seed=seed,output_dir=root/'artifacts')
        state={'schema_version':1,'identity':identity,'implementation_hash':_implementation_hash(),'created_at':utcnow(),'seed':int(seed),'model':model,
            'campaign_id':campaign.id,'campaign_config':cc,'trajectory':[],'status':'running',
            'research_accepted':campaign.status()['research_accepted'],'demo':not campaign.status()['research_accepted'],
            'split':cfg.get('campaign_split',cfg.get('split','development'))}
    if not campaign.status()['research_accepted'] and not allow_demo:
        state.update(status='blocked',error='Accepted physical fields and certified observability required; allow_demo must be explicit')
        _persist(root,state)
        return {'status':'blocked','run_dir':str(root),'error':state['error'],'research_accepted':False}
    owned=provider is None
    if owned:
        from .providers import GeminiProvider
        provider=GeminiProvider(cfg.get('gemini'),output_dir=root/'provider')
    state.update(status='running',updated_at=utcnow());state.pop('error',None);_persist(root,state)
    try:
        while not campaign.status()['completed']:
            prompt,images=_prompt(campaign,state['trajectory'])
            forced=len(state['trajectory'])>=maximum-1
            output=({'action':'submit_report','arguments':{},'evidence_note':'Controller finalization at action allowance; not a model decision'} if forced else await provider.generate('campaign_teacher',prompt,images,model=model))
            try:action,arguments,note=_decode(output)
            except (ValueError,TypeError):action,arguments,note='invalid_policy_action',{},'Rejected policy action schema'
            result=campaign.act(action,arguments)
            state['trajectory'].append({'index':len(state['trajectory']),'prompt':prompt,'images':[str(p) for p in images],
                'action':action,'arguments':arguments,'evidence_note':note,'forced':forced,'result':result})
            state.update(updated_at=utcnow(),status='completed' if result['status']['completed'] else 'running');_persist(root,state)
        return {'status':'completed','run_dir':str(root),'actions':len(state['trajectory']),'campaign':campaign.status(),
                'research_accepted':state['research_accepted'],'demo':state['demo']}
    except Exception as exc:
        state.update(status='failed',error=redacted_error(exc),updated_at=utcnow());_persist(root,state)
        return {'status':'failed','run_dir':str(root),'actions':len(state['trajectory']),'error':state['error'],'demo':state['demo']}
    finally:
        if owned:await provider.close()

def evaluate_run(run_dir):
    """Privileged offline evaluation; not passed to the model provider."""
    root=Path(run_dir).resolve();state=json.loads((root/'state.json').read_text())
    if state['status']!='completed':raise ValueError('Only completed runs can be evaluated')
    result=_restore(state,root).evaluator_snapshot();result['demo']=state['demo']
    save(root/'evaluator.json',result);(root/'evaluator.json').chmod(0o600)
    return result

async def run_demo_campaigns(cfg=None,output_dir=None,count=2,max_actions=16):
    cfg=copy.deepcopy(cfg or load_config());root=Path(output_dir or data_root()/'campaigns'/'development-demo')
    async def one(i):
        return await run_campaign(cfg,seed=cfg.get('seed',0)+i,run_dir=root/f'run-{i:02d}',model=cfg.get('gemini',{}).get('bulk_model'),max_actions=max_actions,allow_demo=True)
    return await asyncio.gather(*(one(i) for i in range(count)))

async def run_campaign_batch(frozen_manifest,field_registry,cfg=None,run_root=None,split='train',model=None,concurrency=None):
    """Bounded parallel workers share one provider and project quota limiter.

    No provider is constructed until every selected field passes acceptance.
    Evaluation remains a separate privileged step after policy completion.
    """
    from .generation import _load,plan_generation
    cfg=copy.deepcopy(cfg or load_config());frozen=_load(frozen_manifest)
    selected={**frozen,'campaigns':[row for row in frozen.get('campaigns',[]) if row['split']==split]}
    if not selected['campaigns']:raise ValueError('Selected split contains no frozen campaigns')
    plan=plan_generation(selected,field_registry)
    if plan['status']!='ready':
        return {'status':'blocked','split':split,'requested_campaigns':len(selected['campaigns']),
                'completed_campaigns':0,'problems':plan['problems'],'research_accepted':False}
    from .providers import GeminiProvider
    root=Path(run_root or data_root()/'campaigns'/split).resolve();root.mkdir(parents=True,exist_ok=True)
    provider=GeminiProvider(cfg.get('gemini'),output_dir=root/'provider')
    workers=int(concurrency or cfg.get('gemini',{}).get('concurrency',8))
    if not 1<=workers<=int(cfg.get('gemini',{}).get('max_concurrency',128)):
        await provider.close();raise ValueError('Campaign concurrency exceeds configured maximum')
    queue=asyncio.Queue();results=[]
    for field in plan['accepted_fields']:queue.put_nowait(field)
    async def worker():
        while not queue.empty():
            try:field=queue.get_nowait()
            except asyncio.QueueEmpty:return
            try:
                spec=field['spec'];local=copy.deepcopy(cfg)
                local.update(campaign_split=spec['split'],thermal_field_path=field['field_path'],
                             observability_manifest_path=field['observability_manifest_path'],mode='research')
                result=await run_campaign(local,seed=spec['seed'],run_dir=root/spec['id'],provider=provider,model=model)
                results.append({'campaign_id':spec['id'],**{k:v for k,v in result.items() if k!='campaign'}})
                save(root/'batch-state.json',{'status':'running','split':split,'requested_campaigns':len(selected['campaigns']),
                     'completed_campaigns':sum(r['status']=='completed' for r in results),'results':results})
            except Exception as exc:
                results.append({'campaign_id':field['spec']['id'],'status':'failed','error':redacted_error(exc)})
            finally:queue.task_done()
    try:await asyncio.gather(*(worker() for _ in range(workers)))
    finally:await provider.close()
    report={'status':'completed' if all(r['status']=='completed' for r in results) else 'failed',
            'split':split,'requested_campaigns':len(selected['campaigns']),
            'completed_campaigns':sum(r['status']=='completed' for r in results),'results':results,
            'research_accepted':all(r.get('research_accepted',False) for r in results)}
    save(root/'batch-state.json',report)
    return report

def export_policy_trajectories(run_dirs,output_path,*,split='train',allow_demo=False):
    if split not in ('train','validation','test'):raise ValueError('Unknown trajectory split')
    rows=[]
    for directory in run_dirs:
        root=Path(directory).resolve();state=json.loads((root/'state.json').read_text())
        if state['status']!='completed':raise ValueError('Incomplete trajectory excluded from training')
        if state['demo'] and not allow_demo:raise ValueError('Demo trajectory excluded from research training')
        if split=='train' and state.get('split') in ('validation','test','fast_test','direct_test'):
            raise ValueError('Held-out campaign trajectories cannot enter training')
        for entry in state['trajectory']:
            if entry['forced'] or not entry['result']['ok']:continue
            artifact_root=data_root().resolve()
            image_refs=[]
            for image in entry['images']:
                path=Path(image).resolve()
                image_refs.append(str(path.relative_to(artifact_root)) if path.is_relative_to(artifact_root) else str(path))
            rows.append({'id':digest([state['identity'],entry['index']]),'run_id':state['identity'],'seed':state['seed'],
                'action_index':entry['index'],'split':split,'provenance':'demo_gemini_policy_trajectory' if state['demo'] else 'gemini_policy_trajectory',
                'research_accepted':state['research_accepted'],'prompt':entry['prompt'],'images':image_refs,
                'completion':json.dumps({'action':entry['action'],'arguments':entry['arguments'],'evidence_note':entry['evidence_note']}),'model':state['model'],
                'campaign_id':state['campaign_id'],'campaign_config':state['campaign_config'],
                'prefix_actions':[{'action':p['action'],'arguments':p['arguments']} for p in state['trajectory'][:entry['index']]]})
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    return {'status':'completed','rows':len(rows),'path':str(path),'research_accepted':bool(rows) and all(r['research_accepted'] for r in rows)}
