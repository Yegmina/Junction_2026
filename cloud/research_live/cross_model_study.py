"""Compare Gemini 4 Carbon/Argon with cached Robotics ER2 on Aki-reviewed images."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,signal,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from aerorozum.common import config,digest,save,utcnow
from aerorozum.providers import GeminiProvider,ProviderDraining,WindowLimiter
from aerorozum.label_schema import normalize_thermal_output
from aerorozum.labelling import classify_response
from score_tolerant import score as tolerant_score
from aerorozum.training import detection_score

MODELS=('gemini-4-carbon','gemini-4-argon')
LEVELS=('low','medium','high')
REPLICAS=2
MAX_OUTPUT=8192
RPM=20
RPM_STEPS=(20,40,60,90,120)
TPM=600000
CONCURRENCY=12

def file_hash(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def boxscore(text,row):
 try:
  output=json.loads(text);normalized,_=normalize_thermal_output(output);items=classify_response(normalized,.6)['accepted'];predictions=[x['bbox'] for x in items];valid=True
 except (ValueError,TypeError,KeyError):output=None;predictions=[];valid=False
 return output,predictions,valid,tolerant_score(predictions,row.get('boxes',[]),edge_tolerance=.75)

def protocol(source,output,root):
 previous=json.loads((source/'protocol.json').read_text())
 if len(previous.get('images',[]))!=88 or previous.get('annotation_origin_counts')!={'original_manual':30,'reviewed_ai':58}:raise ValueError('Expected all 88 Aki-validated Telegram frames')
 rows=previous['images']
 for row in rows:
  path=(root/row['image']).resolve()
  if not path.is_relative_to(root.resolve()) or file_hash(path)!=row['sha256']:raise ValueError('Frozen image missing or changed')
 doc=dict(schema_version=1,created_at=utcnow(),dataset_protocol_sha256=previous['sha256'],images=rows,models=list(MODELS),prompts=previous['prompts'],thinking_levels=list(LEVELS),replicas=REPLICAS,confidence_threshold=.6,iou_threshold=.5,edge_tolerance_fraction=.75,scoring_rule='aki-box-tolerant-v4-75pct; one-to-one IoU >= 0.5 OR all four predicted edges within 75% of the Aki box width/height; small contained boxes count',maximum_output_tokens=MAX_OUTPUT,request_caps={m:dict(initial_rpm=RPM,ramp_rpm=RPM_STEPS,tpm=TPM,concurrency=CONCURRENCY) for m in MODELS},test_accessed=False,training_eligible=False,interpretation='Development annotation agreement only. All 88 examples are positive; not a mine-free specificity estimate.')
 doc['sha256']=digest(doc);save(output/'protocol.json',doc);return doc

def summarize(doc,out):
 groups=defaultdict(list)
 for p in (out/'results').glob('*.json'):
  try:r=json.loads(p.read_text());groups[(r['model'],r['prompt_version'],r['thinking_level'])].append(r)
  except Exception:pass
 arms=[]
 for model in MODELS:
  for prompt in doc['prompts']:
   for level in LEVELS:
    rr=groups[(model,prompt,level)];tp=sum(x['score_tolerant']['tp'] for x in rr if x.get('score_tolerant'));fp=sum(x['score_tolerant']['fp'] for x in rr if x.get('score_tolerant'));fn=sum(x['score_tolerant']['fn'] for x in rr if x.get('score_tolerant'));arms.append(dict(model=model,prompt=prompt,thinking_level=level,completed=len(rr),expected=len(doc['images'])*REPLICAS,tp=tp,fp=fp,fn=fn,f2=5*tp/max(1,5*tp+4*fn+fp),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),invalid=sum(not x.get('valid_output',False) for x in rr),transport_errors=sum(x.get('status')=='transport_error' for x in rr),thinking_tokens=sum(x.get('usage',{}).get('thoughts_token_count',0) or 0 for x in rr),prompt_tokens=sum(x.get('usage',{}).get('prompt_token_count',0) or 0 for x in rr),answer_tokens=sum(x.get('usage',{}).get('candidates_token_count',0) or 0 for x in rr)))
 expected=len(doc['images'])*len(MODELS)*len(doc['prompts'])*len(LEVELS)*REPLICAS
 return dict(status='completed' if sum(a['completed'] for a in arms)==expected else 'running',updated_at=utcnow(),protocol_sha256=doc['sha256'],completed=sum(a['completed'] for a in arms),total=expected,images=len(doc['images']),models=list(MODELS),prompts=list(doc['prompts']),thinking_levels=list(LEVELS),replicas=REPLICAS,arms=arms,heldout_test_accessed=False,training_eligible=False)

async def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='/opt/aerorozum');ap.add_argument('--work',required=True);args=ap.parse_args();root=Path(args.root)/'artifacts';work_dir=Path(args.work);out=work_dir/'output';out.mkdir(parents=True,exist_ok=True)
 os.environ['AEROROZUM_DATA_ROOT']=str(Path(args.root)/'artifacts')
 credential_dir=os.environ.get('CREDENTIALS_DIRECTORY')
 os.environ['AEROROZUM_GEMINI_KEY_FILE']=str(Path(credential_dir)/'gemini-api-key') if credential_dir else '/etc/aerorozum/hackathon-key'
 os.environ.pop('GEMINI_API_KEY',None)
 prior=json.loads((Path(args.root)/'private-prompt-study/current.json').read_text());source=Path(prior['work_dir'])/'output';doc=protocol(source,out,root)
 save(out/'throughput.json',dict(at=utcnow(),per_model_initial_rpm=RPM,per_model_rpm_ceiling=RPM_STEPS[-1],adaptive_ramp_rpm=RPM_STEPS,per_model_tpm=TPM,per_model_concurrency=CONCURRENCY,model_quotas_separate=True,project_spend_cap_verified=False,actual_provider_quotas_verified=False,models=list(MODELS)))
 stop=asyncio.Event();halt={m:asyncio.Event() for m in MODELS};loop=asyncio.get_running_loop();last_429={m:time.monotonic() for m in MODELS};quota_events=[]
 for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stop.set)
 providers={};queues={(m,l):asyncio.Queue() for m in MODELS for l in LEVELS};model_limiters={m:WindowLimiter(RPM,TPM) for m in MODELS};model_semaphores={m:asyncio.Semaphore(CONCURRENCY) for m in MODELS}
 for model in MODELS:
  for level in LEVELS:
   cfg=dict(config()['gemini'],thinking_level=level,temperature=.1,reserve_output_tokens=MAX_OUTPUT,max_retries=6,concurrency=CONCURRENCY,requests_per_minute=RPM,input_tokens_per_minute=TPM,http_timeout_ms=600000)
   cfg.pop('thinking_budget',None);provider=GeminiProvider(cfg,output_dir=out/'provider'/model/level);provider.limiter=model_limiters[model];provider.semaphore=model_semaphores[model];providers[(model,level)]=provider
   async def backoff(ex,status,model=model):
    if status==429:
     limiter=model_limiters[model];limiter.rpm=max(4,int(limiter.rpm*.75));last_429[model]=time.monotonic()
     quota_events.append(dict(at=utcnow(),model=model,kind='429 backoff',rpm_after=limiter.rpm))
     save(out/'quota-events.json',dict(updated_at=utcnow(),events=quota_events))
   provider.on_request_error=backoff
   for replica in range(REPLICAS):
    for row in doc['images']:
     for prompt_version,prompt in doc['prompts'].items():queues[(model,level)].put_nowait((replica,row,prompt_version,prompt,level))
 total=sum(q.qsize() for q in queues.values());completed=0;errors=[]
 save(out/'progress.json',dict(status='running',completed=0,total=total,issued=0,issued_by_model={m:0 for m in MODELS},rate_by_model={m:RPM for m in MODELS},errors=[]))
 async def process_queue(model,level):
  nonlocal completed
  provider=providers[(model,level)];q=queues[(model,level)]
  while not stop.is_set() and not halt[model].is_set():
   try:replica,row,pv,prompt,level=q.get_nowait()
   except asyncio.QueueEmpty:return
   cfg=dict(provider.cfg,thinking_level=level);role=f'gemini4-cross-model-v1:{model}:{pv}:replica-{replica}';key=digest([doc['sha256'],model,pv,level,replica,row['id']]);path=out/'results'/(key+'.json')
   if path.exists():
    completed+=1;q.task_done();continue
   image=(root/row['image']).resolve();started=time.monotonic();record=None
   try:
    await provider.generate(role,prompt,[image],model=model,max_output_tokens=MAX_OUTPUT)
    record=provider.response_record(role,prompt,[image],model=model,max_output_tokens=MAX_OUTPUT)
    text=record.get('raw_text') or json.dumps(record.get('output') or {})
    output,predictions,valid,score=boxscore(text,row);status='completed' if valid else 'invalid_paid_response';err=None;usage=record.get('usage') or {};
   except ProviderDraining:
    q.task_done();return
   except Exception as ex:
    msg=str(ex)[:300];status='transport_error';err=msg;output=None;predictions=[];valid=False;score=tolerant_score([],row.get('boxes',[]),edge_tolerance=.75);usage={}
    if ('429' in msg or 'RESOURCE_EXHAUSTED' in msg) and last_429[model]<started:
     model_limiters[model].rpm=max(4,int(model_limiters[model].rpm*.75));last_429[model]=time.monotonic();quota_events.append(dict(at=utcnow(),model=model,kind='429 backoff after retries',rpm_after=model_limiters[model].rpm));save(out/'quota-events.json',dict(updated_at=utcnow(),events=quota_events))
    errors.append(dict(model=model,id=row['id'],error=msg,status=status))
   result=dict(id=row['id'],sha256=row['sha256'],model=model,prompt_version=pv,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),thinking_level=level,replica=replica,status=status,error=err,valid_output=valid,model_output=output,predictions=predictions,score_iou=detection_score(predictions,row.get('boxes',[])),score_tolerant=score,scoring_rule_version='aki-box-tolerant-v4-75pct',usage=usage,elapsed_s=record.get('elapsed_s',time.monotonic()-started) if record else time.monotonic()-started,role=role,request_id=record.get('id') if record else None,test_accessed=False,training_eligible=False,at=utcnow())
   save(path,result);completed+=1;q.task_done();save(out/'progress.json',dict(status='running',updated_at=utcnow(),completed=completed,total=total,issued=sum(p.issued for p in providers.values()),issued_by_model={m:sum(p.issued for (mm,_),p in providers.items() if mm==m) for m in MODELS},failed=sum(x['status']=='transport_error' for x in errors),errors=errors[-20:]))
   if completed%25==0:save(out/'report.json',summarize(doc,out))
 async def checkpoint():
  while not stop.is_set():
   now=time.monotonic()
   for m in MODELS:
    limiter=model_limiters[m]
    if limiter.rpm<RPM_STEPS[-1] and now-last_429[m]>45:
     steps=[x for x in RPM_STEPS if x>limiter.rpm]
     if steps:limiter.rpm=steps[0];last_429[m]=now;quota_events.append(dict(at=utcnow(),model=m,kind='rate ramp',rpm=limiter.rpm));save(out/'quota-events.json',dict(updated_at=utcnow(),events=quota_events))
   save(out/'report.json',summarize(doc,out));save(out/'progress.json',dict(status='running',updated_at=utcnow(),completed=completed,total=total,issued=sum(p.issued for p in providers.values()),issued_by_model={m:sum(p.issued for (mm,_),p in providers.items() if mm==m) for m in MODELS},rate_by_model={m:model_limiters[m].rpm for m in MODELS},failed=sum(x['status']=='transport_error' for x in errors),errors=errors[-20:]));await asyncio.sleep(15)
 from aerorozum.label_archive import archive_pass
 from cloud.airborne_worker import metadata_record_store
 store=metadata_record_store('aerorozum-jetson-testdata-smart-bird',prefix='research/gemini4-cross-model-v1/'+work_dir.name)
 archive_stop=asyncio.Event()
 async def archive_loop():
  while not archive_stop.is_set():
   try:await asyncio.to_thread(archive_pass,out,store,work_dir/'archive-state.json',workers=4)
   except Exception as ex:save(out/'archive-status.json',dict(at=utcnow(),error=str(ex)[:300]))
   try:await asyncio.wait_for(archive_stop.wait(),timeout=60)
   except asyncio.TimeoutError:pass
 task=asyncio.create_task(checkpoint());archiver=asyncio.create_task(archive_loop())
 try:await asyncio.gather(*(process_queue(m,l) for m in MODELS for l in LEVELS for _ in range(CONCURRENCY)))
 finally:
  stop.set();task.cancel();await asyncio.gather(task,return_exceptions=True);await asyncio.gather(*(p.close() for p in providers.values()));archive_stop.set();await archiver
  await asyncio.to_thread(archive_pass,out,store,work_dir/'archive-state.json',workers=4)
 report=summarize(doc,out);report['status']='completed' if report['completed']==total else ('halted_on_model_quota' if any(e.is_set() for e in halt.values()) else 'incomplete');report['issued_by_model']={m:sum(p.issued for (mm,_),p in providers.items() if mm==m) for m in MODELS};save(out/'report.json',report);save(out/'progress.json',dict(status=report['status'],updated_at=utcnow(),completed=report['completed'],total=total,issued=sum(p.issued for p in providers.values()),issued_by_model=report['issued_by_model'],failed=sum(x['status']=='transport_error' for x in errors),errors=errors[-50:]));print(json.dumps({k:report[k] for k in ['status','completed','total','issued_by_model']}),flush=True)

if __name__=='__main__':asyncio.run(main())
