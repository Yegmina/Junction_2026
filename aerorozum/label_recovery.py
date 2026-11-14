"""Durable recovery, preserving active paid requests and checking every cache."""
import asyncio,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from .common import ROOT,config,digest,save,utcnow
from .labelling import MODEL,PROMPT,classify_response,inference_cache_key,label_frames

def provider_key(row,cfg):
 return digest(dict(role='robotics_er2_thermal_labels',prompt=PROMPT,images=[row['sha256']],model=MODEL,temperature=.1,max_output_tokens=cfg.get('reserve_output_tokens',2048),api_version=cfg.get('api_version','v1beta'),thinking_budget=cfg.get('thinking_budget')))

def inventory(document,root,cfg):
 root=Path(root);groups={x:[] for x in ('completed','cached','blocked','pending')}
 result_paths={p.stem:p for p in (root/'results').glob('*.json')};paid_keys={p.stem for p in (root/'paid-responses').glob('*.json')};response_keys={p.stem for p in (root/'provider/responses').glob('*.json')};raw_keys={p.stem.rsplit('-',1)[0] for p in (root/'provider/raw').glob('*.json')}
 class Identity:pass
 provider=Identity();provider.cfg=cfg
 for row in document['images']:
  key=inference_cache_key(row,row['sha256'],provider);path=root/'results'/(key+'.json');group='pending'
  if key in result_paths:
   result=json.loads(path.read_text())
   try:
    assert result.get('sha256')==row['sha256'] and result.get('model')==MODEL
    classify_response(result['model_output']);group='completed'
   except (AssertionError,ValueError,KeyError,TypeError):group='blocked'
  elif key in paid_keys or provider_key(row,cfg) in response_keys:group='cached'
  elif provider_key(row,cfg) in raw_keys:group='blocked'
  groups[group].append(row)
 return groups

def paid_permission(path,main_running):
 if Path(path).is_file():
  d=json.loads(Path(path).read_text());expiry=d.get('expires_at');live=not expiry or datetime.fromisoformat(expiry.replace('Z','+00:00'))>datetime.now(timezone.utc)
  if d.get('status')=='granted' and live and 0<d.get('requests_per_minute',0)<=120:return {'allowed':True,'requests_per_minute':int(d['requests_per_minute']),'reason':'explicit recovery allowance lease'}
  return {'allowed':False,'requests_per_minute':0,'reason':'explicit recovery allowance paused, revoked or expired'}
 if not main_running:return {'allowed':True,'requests_per_minute':120,'reason':'main worker finished; no additional allowance override exists'}
 return {'allowed':False,'requests_per_minute':0,'reason':'preserving active main requests; waiting for allowance lease or normal main completion'}

async def recover_pass(manifest,labels_root,lease_path,main_running=True,batch_size=64,provider_factory=None,on_failure=None):
 root=Path(labels_root);recovery=root.parent/'recovery';out=recovery/'labels';out.mkdir(parents=True,exist_ok=True)
 cfg={**config()['gemini'],**json.loads((ROOT/'configs/labelling.json').read_text())};cfg.update(thinking_budget=0,api_version='v1beta')
 doc=json.loads(Path(manifest).read_text());groups=inventory(doc,root,cfg);permission=paid_permission(lease_path,main_running)
 chosen=groups['cached'][:batch_size] or (groups['pending'][:batch_size] if permission['allowed'] else [])
 status='running' if chosen else 'completed' if not any(groups[k] for k in ('cached','pending','blocked')) else 'waiting_for_labeling_allowance'
 state={'status':status,'total':len(doc['images']),**{k:len(v) for k,v in groups.items()},'paid_permission':permission,'main_worker_running':main_running,'active_batch':len(chosen),'updated_at':utcnow(),'model':MODEL};save(recovery/'status.json',state)
 if not chosen:return state
 for name in ('results','paid-responses','provider','checkpoints','failures'):
  (root/name).mkdir(parents=True,exist_ok=True)
  if not (out/name).exists():(out/name).symlink_to(root/name,target_is_directory=True)
 if provider_factory is None:
  from .providers import GeminiProvider
  class Guarded(GeminiProvider):
   async def generate(self,role,prompt,images=(),model=None,**kwargs):
    if permission['allowed']:return await super().generate(role,prompt,images,model=model,**kwargs)
    record=self.response_record(role,prompt,images,model=model)
    if record and isinstance(record.get('output'),dict):return record['output']
    raise RuntimeError('Cache-only recovery guard: paid inference forbidden while main allowance is active')
  provider=Guarded({**cfg,'concurrency':8,'requests_per_minute':permission['requests_per_minute'] or 120},output_dir=root/'provider')
 else:provider=provider_factory(cfg,permission)
 try:
  result=await label_frames(chosen,out,provider=provider,concurrency=8,image_root=root.parents[2],on_failure=on_failure)
  save(recovery/'last-batch.json',{'at':utcnow(),'ids':[r['id'] for r in chosen],'result':result,'paid_permission':permission})
 finally:
  if provider_factory is None and hasattr(provider,'close'):await provider.close()
 return state

def seed_recovery_window(provider,labels_root,document):
 """Restore recent reservations after graceful handoff, conservatively using
 response/error arrival times so a process restart cannot reset the30RPM cap.
 """
 import time
 root=Path(labels_root);keys={provider_key(row,provider.cfg) for row in document['images']};recent={};wall=time.time()
 for directory in ('provider/raw','provider/errors'):
  for path in (root/directory).glob('*.json'):
   if path.stat().st_mtime<wall-60 or path.stem.rsplit('-',1)[0] not in keys:continue
   record=json.loads(path.read_text());stamp=datetime.fromisoformat(record['created_at'].replace('Z','+00:00')).timestamp()
   if stamp>wall-60:recent[path.stem]=max(recent.get(path.stem,0),stamp)
 estimate=max(1,len(PROMPT)//3)+4096;monotonic=time.monotonic()
 for stamp in sorted(recent.values()):provider.limiter.calls.append((monotonic-max(0,wall-stamp),estimate))
 return len(recent)
