"""Exhaustive raw thermal follow-up; frame clocks stay explicit and generation pinned."""
import asyncio,hashlib,json,re,signal,subprocess,sys,urllib.request,urllib.parse,urllib.error
from datetime import datetime,timezone
from pathlib import Path
from .common import ROOT,data_root,config,digest,save,utcnow
from .cloud_cache import GCSRecordStore
from .labelling import extract_video_samples,freeze_session_splits,label_frames
from .quarantine_recovery import receipt_records
BUCKET='aerorozum-jetson-testdata-smart-bird'

def prepare_additional(root=None):
 root=Path(root or data_root());out=root/'labelling/additional-bucket-sources';probe=json.loads((out/'native-probe.json').read_text());auth=GCSRecordStore(BUCKET)
 def get(url):
  for attempt in range(2):
   headers=auth._headers()
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as response:return response.read()
   except urllib.error.HTTPError as e:
    if e.code==401 and attempt==0:auth._invalidate(headers);continue
    raise
 def receipts(src):
  name=src['object'].split('/recordings/')[0]+'/recordings/'+src['session_id']+'/thermal-timestamps.jsonl';url='https://storage.googleapis.com/storage/v1/b/'+BUCKET+'/o/'+urllib.parse.quote(name,safe='');meta=json.loads(get(url));raw=get(url+'?alt=media&generation='+str(meta['generation']));folder=out/'receipts'/digest([name,meta['generation']]);folder.mkdir(parents=True,exist_ok=True);original=folder/'original.jsonl'
  if not original.exists():original.write_bytes(raw)
  records,transform=receipt_records(original,folder/'cleaned.jsonl');return records,folder/'cleaned.jsonl',{**transform,'object':name,'generation':str(meta['generation'])}
 original_path=root/'source-videos/labelling-sources-original-150.json';master_path=root/'source-videos/labelling-sources.json'
 if not original_path.exists():save(original_path,json.loads(master_path.read_text()))
 original=json.loads(original_path.read_text());sources=probe['sources'];dedup_path=out/'recovered-raw-dedup.json';dedup=json.loads(dedup_path.read_text()) if dedup_path.exists() else {}
 duplicate=lambda s: s['recovered_variant'] and dedup.get('identical_decoded_RGB_sequence') and s['object']==dedup.get('recovered_object') and s['generation']==dedup.get('recovered_generation') and s['sha256']==dedup.get('recovered_sha256')
 selected=[s for s in sources if not duplicate(s)]
 splits=freeze_session_splits([s['session_id'] for s in selected],root/'labelling/robotics-er2-5fps/session-splits.json',17);rows=[];reports=[];master={s['object']:s for s in original['videos']};receipt_cache={}
 for src in selected:
  if not src.get('pts_strictly_monotonic'):
   reports.append({**src,'status':'quarantined_no_monotonic_native_PTS'});continue
  sid=src['session_id'];segment=src['segment'];aligned=False;clock={};records=[];timestamp_path=None;reason=None
  try:
   if sid not in receipt_cache:receipt_cache[sid]=receipts(src)
   records,timestamp_path,clock=receipt_cache[sid];local=[x for x in records if int(x.get('segment',0))==segment];values=[int(x['received_at_ns']) for x in local];counters=all('frame' in x for x in local);contiguous=not counters or all(int(b['frame'])==int(a['frame'])+1 for a,b in zip(local,local[1:]));aligned=(not src['recovered_variant'] and len(local)==src['native_frames'] and contiguous and all(b>a for a,b in zip(values,values[1:])))
   if not aligned:reason='receipt/native count, ordering or recovered-variant alignment not verified'
  except (ValueError,KeyError,OSError,urllib.error.HTTPError) as ex:reason='receipt unavailable or invalid: '+type(ex).__name__
  timing={'mode':'original_receipt_time' if aligned else 'video_time_only','wallclock_alignment_available':aligned,'gps_alignment_available':False,'receipt':clock,'fallback_reason':reason}
  source={k:src[k] for k in ('bucket','object','generation','session_id','segment')};source.update(timing_mode=timing['mode'],wallclock_alignment_available=aligned,gps_alignment_available=False,receipt_timestamp_recovery=timing)
  if src['recovered_variant']:source['recovered_variant_of']=src['object'].replace('/recovered/','/').replace('-recovered.avi','.avi')
  folder=out/'frames'/digest(source)[:20]
  try:samples=extract_video_samples(src['path'],folder,session_id=sid,source=source,timestamps_path=timestamp_path if aligned else None,segment=segment,fps=5,split=splits[sid],sample_origin_s=int(records[0]['received_at_ns'])/1e9 if aligned else None)
  except (ValueError,OSError,subprocess.SubprocessError) as ex:
   reports.append({**src,'status':'quarantined_native_frame_decode_or_alignment','error':str(ex)[:300],'timing':timing});continue
  for row in samples:
   row.update(wallclock_alignment_available=aligned,gps_alignment_available=False,timing_resolution=timing['mode'])
   if aligned:row['receipt_timestamp_ns']=int(local[row['frame_index']]['received_at_ns'])
  rows.extend(samples);reports.append({**src,'status':'sampled','sampled_frames':len(samples),'timing':timing});master[src['object']]={**source,'path':src['path'],**({'timestamps_path':str(timestamp_path)} if aligned else {})}
 version_signature=digest(sorted((s['object'],s['generation']) for s in sources))
 signature=digest({'versions':version_signature,'sample_bindings':[(r['id'],r['sha256'],r['source'],r.get('receipt_timestamp_ns'),r['split']) for r in rows],'quarantined_sources':[(s['object'],s['generation'],s['status']) for s in reports if s['status']!='sampled']})
 doc={'metadata':{'status':'sampled_unlabelled','created_at':utcnow(),'generation_signature':version_signature,'content_signature':signature,'source_versions':len(sources),'selected_source_versions':len(selected),'samples':len(rows),'recovered_duplicate_excluded':bool(dedup.get('identical_decoded_RGB_sequence')),'paid_model_calls':0},'session_splits':splits,'sources':reports,'images':rows}
 snapshot=out/('manifest-'+signature[:16]+'.json')
 if not snapshot.exists():save(snapshot,doc)
 save(out/'manifest.json',doc);save(master_path,{'created_at':utcnow(),'videos':list(master.values()),'original_snapshot':str(original_path),'coverage_generation_signature':version_signature});return doc

async def run_additional():
 from .providers import GeminiProvider
 root=data_root();out=root/'labelling/additional-bucket-sources';out.mkdir(parents=True,exist_ok=True)
 active=lambda unit:subprocess.run(['systemctl','--user','is-active','--quiet',unit],stdout=subprocess.DEVNULL).returncode==0
 if active('aerorozum-robotics-label-bucket.service') or active('aerorozum-quarantine-labels.service'):
  save(out/'status.json',{'status':'waiting_for_previous_label_stage','at':utcnow(),'paid_model_calls':0});return
 labels=out/'labels';labels.mkdir(parents=True,exist_ok=True);main_labels=root/'labelling/robotics-er2-5fps/labels'
 for name in ('results','paid-responses','provider','checkpoints','failures','repair-index'):
  (main_labels/name).mkdir(parents=True,exist_ok=True)
  if not (labels/name).exists():(labels/name).symlink_to(main_labels/name,target_is_directory=True)
 stopped=False
 def stop(*_):
  nonlocal stopped
  stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 cfg={**config()['gemini'],**json.loads((ROOT/'configs/labelling.json').read_text())};cfg.update(concurrency=16,thinking_budget=0,api_version='v1beta');provider=GeminiProvider(cfg,output_dir=main_labels/'provider')
 async def gate():
  nonlocal stopped
  lease=json.loads((root/'pipeline/leases/label-recovery.json').read_text());deadline=datetime.fromisoformat(lease['expires_at'].replace('Z','+00:00'))
  if stopped or datetime.now(timezone.utc)>=deadline:
   stopped=True;raise RuntimeError('Recovery stop or deadline reached; no new paid request issued')
 provider.before_queue=gate;provider.before_request=gate
 ledger_path=out/'processed-versions.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'versions':[]};covered=set(ledger['versions']);source_quarantines=ledger.get('source_quarantines',[]);frame_failures=0
 try:
  while not stopped:
   # Re-list at each completed pass so late-arriving objects cannot be omitted.
   save(out/'status.json',{'status':'discovering_generation_pinned_sources','at':utcnow(),'allowance_rpm':cfg['requests_per_minute']})
   process=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'scripts/discover_thermal_sources.py'),stdout=asyncio.subprocess.DEVNULL);code=await process.wait()
   if code:raise RuntimeError('Source discovery failed; no source silently skipped')
   doc=await asyncio.to_thread(prepare_additional,root);pending=[s for s in doc['sources'] if digest([s['object'],s['generation']]) not in covered]
   if not pending:break
   pending_versions={(s['object'],s['generation']) for s in pending};images=[r for r in doc['images'] if (r['source']['object'],r['source']['generation']) in pending_versions]
   save(out/'status.json',{'status':'labeling','at':utcnow(),'source_versions':len(pending),'frames':len(images),'allowance_rpm':cfg['requests_per_minute'],'main_overlap':False})
   result=await label_frames(images,labels,provider=provider,concurrency=16)
   if stopped:break
   source_quarantines.extend({k:s[k] for k in ('object','generation','status','error') if k in s} for s in pending if s['status']!='sampled');frame_failures+=result.get('failed',0)
   covered.update(digest([s['object'],s['generation']]) for s in pending);save(ledger_path,{'versions':sorted(covered),'at':utcnow(),'last_result':result,'frames_attempted':len(images),'source_quarantines':source_quarantines,'frame_failures_observed':frame_failures})
  if not stopped:
   from .invalid_labels import recover_invalid
   repair=await recover_invalid(provider,root)
  else:repair={'status':'deferred'}
  save(out/'status.json',{'status':'paused' if stopped else 'completed_with_explicit_failures' if source_quarantines or frame_failures or any(x.get('status')=='bounded_attempt_failed_original_retained' for x in repair.get('results',[])) else 'completed','at':utcnow(),'processed_source_versions':len(covered),'source_quarantines':source_quarantines,'frame_failures_observed':frame_failures,'invalid_response_recovery':repair,'main_overlap':False,'end_rescan_completed':not stopped})
 finally:await provider.close()
