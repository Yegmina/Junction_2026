"""Native source-time labelling. Machine/user-accepted labels are not human review."""
from __future__ import annotations
import asyncio
from collections import Counter
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

import numpy as np
from PIL import Image,ImageDraw
from .common import ROOT,config as load_config,data_root,digest,redacted_error,save,utcnow
from .evaluation import valid_box
from .label_prompts import PROMPT_V1, PROMPT_V2, PROMPTS, PROMPT_VERSION, prompt_metadata, result_prompt_metadata, registry as prompt_registry
from .label_schema import normalize_thermal_output

MODEL='gemini-robotics-er-2-preview'
PROMPT=PROMPTS[os.environ.get('AEROROZUM_LABEL_PROMPT_VERSION',PROMPT_VERSION)]

def cache_gcs_object(bucket,object_name,cache_dir,generation=None):
    """Stream a generation-pinned object through authenticated gcloud into cache."""
    from .preflight import sdk_path
    if not bucket or not object_name:raise ValueError('Explicit GCS bucket/object required')
    uri=f'gs://{bucket}/{object_name}'
    requested=uri+('#'+str(generation) if generation is not None else '')
    proc=subprocess.run([sdk_path(),'storage','objects','describe',requested,'--format=json'],capture_output=True,text=True,check=True)
    metadata=json.loads(proc.stdout);pinned=str(metadata['generation'])
    if generation is not None and pinned!=str(generation):raise ValueError('GCS object generation changed')
    root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True);suffix=Path(object_name).suffix.lower()
    if suffix not in ('.avi','.mp4','.mkv','.jsonl','.json'):suffix='.bin'
    identity=digest([bucket,object_name,pinned]);path=root/(identity+suffix);sidecar=root/(identity+'.cache.json')
    if path.is_file() and sidecar.is_file():
        old=json.loads(sidecar.read_text())
        if path.stat().st_size==int(metadata['size']) and _hash(path)==old.get('sha256'):return {**old,'path':str(path),'cached':True}
    temporary=path.with_suffix(path.suffix+'.partial')
    subprocess.run([sdk_path(),'storage','cp',uri+'#'+pinned,str(temporary),'--quiet'],capture_output=True,text=True,check=True)
    if temporary.stat().st_size!=int(metadata['size']):temporary.unlink(missing_ok=True);raise ValueError('GCS download size mismatch')
    temporary.replace(path)
    record={'bucket':bucket,'object':object_name,'generation':pinned,'size':path.stat().st_size,'sha256':_hash(path),'crc32c':metadata.get('crc32c')}
    save(sidecar,record);return {**record,'path':str(path),'cached':False}

def _hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()

def freeze_session_splits(session_ids,path,seed=17,assignments=None):
    path=Path(path);doc=json.loads(path.read_text()) if path.exists() else {'seed':int(seed),'sessions':{}}
    if doc['seed']!=int(seed):raise ValueError('Session split seed is already frozen')
    for sid in sorted(set(session_ids)):
        if not sid:raise ValueError('Each video needs a stable session_id')
        fraction=int(digest([int(seed),sid])[:12],16)/16**12
        split=(assignments or {}).get(sid,'train' if fraction<.8 else 'validation' if fraction<.9 else 'test')
        if split not in ('train','validation','test'):raise ValueError('Invalid session split')
        if sid in doc['sessions']:
            if assignments and sid in assignments and doc['sessions'][sid]!=split:raise ValueError('Session split change would leak held-out data')
            continue
        doc['sessions'][sid]=split
    save(path,doc);return doc['sessions']

def _probe(video):
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames',
        '-show_entries','stream=width,height:frame=best_effort_timestamp_time,pkt_dts_time','-of','json',str(video)],capture_output=True,text=True,check=True)
    doc=json.loads(result.stdout)
    if not doc.get('streams'):raise ValueError('No video stream')
    stream=doc['streams'][0];times=[]
    for frame in doc.get('frames',[]):
        value=frame.get('best_effort_timestamp_time',frame.get('pkt_dts_time'))
        if value is None:raise ValueError('Native frame has no actual presentation timestamp')
        times.append(float(value))
    return int(stream['width']),int(stream['height']),times

def sample_indices(times,fps=5,origin=None):
    if not times or not math.isfinite(float(fps)) or fps<=0:raise ValueError('Positive rate and source timestamps required')
    values=np.asarray(times,dtype=float)
    if not np.isfinite(values).all() or np.any(np.diff(values)<=0):raise ValueError('Source timestamps must strictly increase')
    origin=values[0] if origin is None else float(origin)
    first=max(0,math.ceil((values[0]-origin)*fps-1e-6));last=math.floor((values[-1]-origin)*fps+1e-6)
    targets=origin+np.arange(first,last+1)/fps
    return sorted(set(int(i) for i in np.searchsorted(values,targets,side='left') if i<len(values)))

def _frame_times(video,pts,timestamps_path=None,segment=None):
    if timestamps_path is None:return pts,'container_pts_seconds',False
    records=[json.loads(line) for line in Path(timestamps_path).read_text().splitlines() if line.strip()]
    if segment is None:
        match=re.search(r'thermal-(\d+)\.',Path(video).name)
        if match:segment=int(match.group(1))
    if segment is None and len({r.get('segment') for r in records})>1:raise ValueError('Specify video segment for a multi-segment timestamp list')
    if segment is not None:records=[r for r in records if int(r.get('segment',0))==int(segment)]
    if len(records)!=len(pts):raise ValueError('Receipt timestamps do not align one-to-one with native frames')
    return [int(r['received_at_ns'])/1e9 for r in records],'jetson_receipt_timestamp_not_hardware_exposure',True

def extract_video_samples(video,output_dir,*,session_id,source=None,timestamps_path=None,segment=None,fps=5,split='train',max_frames=None,sample_origin_s=None):
    video=Path(video).resolve();root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    if not session_id:raise ValueError('Stable session_id is required')
    width,height,pts=_probe(video);times,basis,absolute=_frame_times(video,pts,timestamps_path,segment)
    indices=sample_indices(times,fps,sample_origin_s)
    if max_frames is not None:
        if int(max_frames)<1:raise ValueError('max_frames must be positive')
        indices=indices[:int(max_frames)]
    selected=set(indices);rows=[];source={**(source or {}),'session_id':session_id,'video_sha256':_hash(video)}
    prior=root/'sampling.json'
    if prior.is_file():
        cached=json.loads(prior.read_text());old=cached.get('images',[])
        expected=[digest([source,index,times[index]]) for index in indices]
        if (cached.get('source')==source and float(cached.get('sampling_fps',0))==float(fps)
                and [r.get('id') for r in old]==expected
                and all(Path(r['image']).is_file() and _hash(r['image'])==r.get('sha256') for r in old)):
            return old
    with tempfile.TemporaryFile() as errors:
        process=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(video),'-map','0:v:0','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=errors)
        size=width*height*3;index=0
        try:
            while True:
                raw=process.stdout.read(size)
                if not raw:break
                if len(raw)!=size:raise ValueError('Truncated native video frame')
                if index in selected:
                    identity=digest([source,index,times[index]]);image=root/(identity+'.jpg')
                    Image.frombytes('RGB',(width,height),raw).save(image,quality=95)
                    rows.append({'id':identity,'image':str(image),'sha256':_hash(image),'width':width,'height':height,
                        'split':split,'source':source,'frame_index':index,'source_pts_s':pts[index],
                        'timestamp_s':times[index] if absolute else None,'source_offset_s':times[index]-times[0],
                        'time_basis':basis,'sampling_fps':float(fps),'sample_grid_interval_s':1/float(fps)})
                index+=1
        except BaseException:
            process.terminate();process.wait();raise
        finally:process.stdout.close()
        code=process.wait(timeout=30)
        if code or index!=len(times):raise ValueError(f'Native frame count mismatch ({index}/{len(times)}); source quarantined')
    save(root/'sampling.json',{'source':source,'native_frames':len(times),'sampled_frames':len(rows),
        'sampling_fps':fps,'density_basis':'source clock, independent of inference speed','images':rows})
    return rows

def classify_response(output,threshold=.6):
    if not isinstance(output,dict) or not isinstance(output.get('detections'),list):raise ValueError('Model response lacks a valid detections array')
    if not 0<=float(threshold)<=1:raise ValueError('Threshold outside [0,1]')
    accepted=[];uncertain=[]
    for raw in output['detections']:
        if not isinstance(raw,dict) or not valid_box(raw.get('bbox')):raise ValueError('Malformed model candidate box')
        confidence=raw.get('confidence')
        if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError('Malformed model confidence')
        (accepted if confidence>=threshold else uncertain).append(copy.deepcopy(raw))
    return {'accepted':accepted,'uncertain':uncertain,'zero_candidates':len(output['detections'])==0}

async def _call(function,*args,**kwargs):
    if inspect.iscoroutinefunction(function):return await function(*args,**kwargs)
    value=await asyncio.to_thread(function,*args,**kwargs)
    return await value if inspect.isawaitable(value) else value

async def _provider_log(provider,path,model):
    record=await _call(provider.response_record,'robotics_er2_thermal_labels',PROMPT,[path],model=model) if hasattr(provider,'response_record') else None
    if record is not None:return record
    if not hasattr(provider,'root'):return None
    sha=_hash(path)
    # Legacy providers used an older key without generation settings. Preserve
    # actual existing logs rather than reconstructing supposedly raw text.
    for candidate in (Path(provider.root)/'responses').glob('*.json'):
        try:old=json.loads(candidate.read_text())
        except (OSError,ValueError):continue
        if old.get('model')==model and old.get('prompt')==PROMPT and old.get('image_hashes')==[sha]:
            return {**old,'legacy_cache_key':True,'raw_text_available':'raw_text' in old,
                    'raw_text_policy':'original retained if present; never reconstructed'}
    return None

def inference_cache_key(row,sha,provider,model=MODEL,*,prompt=None):
    cfg=getattr(provider,'cfg',{})
    identity={'model':model,'image_sha256':sha,'prompt':PROMPT if prompt is None else prompt,
        'generation_config':{'temperature':.1,'max_output_tokens':cfg.get('reserve_output_tokens',2048),
            'api_version':cfg.get('api_version','v1beta'),'thinking_budget':cfg.get('thinking_budget',0)},
        'source_generation':row.get('source',{}).get('generation')}
    # Explicit bounded repairs have separate identities. Ordinary paid-cache
    # identities remain unchanged, including every existing remote record.
    if cfg.get('inference_revision'):identity['inference_revision']=cfg['inference_revision']
    if cfg.get('thinking_level') is not None:
        identity['generation_config'].pop('thinking_budget',None)
        identity['generation_config']['thinking_level']=str(cfg['thinking_level']).lower()
    return digest(identity)


async def _previous_prompt_result(row,sha,provider,model,root,store):
    """Reuse valid paid v1 labels under the v1 key, without calling them v2."""
    if PROMPT==PROMPT_V1 or getattr(provider,'cfg',{}).get('reuse_previous_prompt',True) is False:
        return None
    keys=[inference_cache_key(row,sha,provider,model,prompt=PROMPT_V1),
          digest([row.get('id'),sha,model,.6,PROMPT_V1])]
    for key in keys:
        path=root/'results'/(key+'.json')
        result=json.loads(path.read_text()) if path.is_file() else None
        if result is None and store is not None:result=await _call(store.get,key)
        if result is None:continue
        if result.get('sha256')!=sha or result.get('model')!=model:
            raise ValueError('Previous prompt cache model or image hash mismatch; no paid request issued')
        if result.get('status')=='invalid_model_response':continue
        try:classify_response(result['model_output'],.6)
        except (ValueError,KeyError,TypeError):continue
        if result.get('raw_response') is None and store is not None:
            result['raw_response']=await _call(store.get,key,kind='logs')
        metadata=result_prompt_metadata(result,assumed_prompt=PROMPT_V1)
        if metadata['label_prompt_version']!='thermal-v1-conservative':
            raise ValueError('Previous prompt cache has conflicting prompt provenance')
        result.update(metadata);save(path,result)
        return result,key,path
    return None

def _live_preview(result,source_path,root,identity):
    """Create live evidence before cloud sync; original training pixels stay intact."""
    with Image.open(source_path) as im:
        original=im.convert('RGB');width,height=original.size
    header=24;image=Image.new('RGB',(width,height+header),'#101722');image.paste(original,(0,header));draw=ImageDraw.Draw(image)
    draw.text((6,5),'PROVISIONAL THERMAL EVIDENCE | accept >= 0.60',fill='white')
    shapes=[]
    for detection in result['detections']:
        box=detection['bbox'];accepted=detection['confidence']>=.6
        x1,y1=box['x_min']*width/1000,box['y_min']*height/1000
        x2,y2=box['x_max']*width/1000,box['y_max']*height/1000
        color='#32df6f' if accepted else '#ffb347'
        draw.rectangle((x1,y1+header,x2,y2+header),outline=color,width=2)
        draw.text((x1,max(header,y1+header-12)),('candidate ' if accepted else 'uncertain ')+f"{detection['confidence']:.2f}",fill=color)
        shapes.append({'label':'mine_candidate' if accepted else 'uncertain_thermal_signature',
            'points':[[x1,y1],[x2,y2]],'shape_type':'rectangle','group_id':None,
            'flags':{'user_accepted':accepted,'human_reviewed':False},'description':json.dumps({'model':MODEL,'confidence':detection['confidence']})})
    preview=root/'annotated'/(identity+'-'+digest(result.get('id'))[:8]+'.jpg');preview.parent.mkdir(parents=True,exist_ok=True)
    temporary=preview.with_suffix('.tmp');image.save(temporary,format='JPEG',quality=95);temporary.replace(preview)
    base=data_root().resolve();result['annotated']=str(preview.relative_to(base)) if preview.is_relative_to(base) else str(preview)
    result['labelme']={'version':'5.4.1','flags':{'human_reviewed':False,'user_accepted_model':True},
        'imagePath':Path(source_path).name,'imageHeight':height,'imageWidth':width,'imageData':None,
        'shapes':shapes,'model':MODEL,'user_threshold':.6}
    result['annotation_colors']={'accepted':'green','uncertain':'amber'}

def _failure_kind(error):
    text=str(error)
    if text.startswith('Recovery ') and ('no new paid request' in text or 'paid inference forbidden' in text):return 'recovery_deferred'
    if 'GCS cache read failed: HTTP 401' in text:return 'cloud_cache_authentication'
    if 'GCS cache read failed:' in text:return 'cloud_cache_read'
    if 'Configured Gemini request allowance reached' in text:return 'configured_request_allowance'
    if 'schema' in text or 'detections array' in text or 'Malformed model' in text or text.startswith(('Expecting ','Unterminated ')):return 'invalid_paid_response'
    if 'Source image' in text:return 'source_integrity'
    return 'other'

def _frame_checkpoint_key(row):
    return digest([row.get('id'),row.get('sha256'),row.get('source',{}).get('generation')])

def _persist_failure(root,row,error,phase,run_id,cache_key=None):
    """Commit each failed frame immediately; never wait for the run manifest."""
    reason=redacted_error(error);kind=_failure_kind(reason)
    record={'status':'failed','run_id':run_id,'at':utcnow(),'id':row.get('id'),
            'image':row.get('image'),'sha256':row.get('sha256'),'source':row.get('source',{}),
            'split':row.get('split'),'frame_index':row.get('frame_index'),
            'timestamp_s':row.get('timestamp_s'),'source_offset_s':row.get('source_offset_s'),
            'phase':phase,'category':kind,'error':reason,'cache_key':cache_key,
            'retry_policy':'cache_first_no_repeated_paid_inference',
            'retryable':kind not in ('invalid_paid_response','source_integrity'),
            'row':row}
    key=_frame_checkpoint_key(row)
    record['failure_id']=digest([run_id,key,record['at']])
    save(root/'failures'/(record['failure_id']+'.json'),record)
    # One O_APPEND write is safe across workers/processes; the separate atomic
    # JSON record remains authoritative if a process dies during journal I/O.
    payload=(json.dumps(record,allow_nan=False,default=str)+'\n').encode()
    fd=os.open(root/'failure-journal.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:os.write(fd,payload);os.fsync(fd)
    finally:os.close(fd)
    save(root/'checkpoints'/(key+'.json'),record)
    return {k:record[k] for k in ('id','error','category','phase','at','failure_id','retryable')}

async def label_frames(rows,output_dir,provider=None,model=MODEL,threshold=.6,concurrency=8,image_root=None,record_store=None,on_failure=None):
    if model!=MODEL:raise ValueError(f'Exact model {MODEL} required; fallback is forbidden')
    if float(threshold)!=.6:raise ValueError('User authorization fixes the acceptance threshold at 0.6')
    if not 1<=int(concurrency)<=128:raise ValueError('Concurrency must be between 1 and 128')
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    base=Path(image_root or data_root()).resolve();owned=provider is None
    if owned:
        from .providers import GeminiProvider
        settings={**load_config()['gemini']}
        labelling_config=ROOT/'configs/labelling.json'
        if labelling_config.is_file():settings.update(json.loads(labelling_config.read_text()))
        settings.update(thinking_budget=0,api_version='v1beta',concurrency=int(concurrency))
        provider=GeminiProvider(settings,output_dir=root/'provider')
    queue=asyncio.Queue(maxsize=max(16,int(concurrency)*2));results=[];failures=[]
    run_id=uuid.uuid4().hex;failure_reasons=Counter()
    save(root/'prompt-registry.json',prompt_registry())
    save(root/'runs'/(run_id+'.json'),{'run_id':run_id,'started_at':utcnow(),'model':model,
        **prompt_metadata(PROMPT),
        'provider_limits':{k:getattr(provider,'cfg',{}).get(k) for k in ('max_requests','requests_per_minute','concurrency')},
        'failure_journal':'failure-journal.jsonl','checkpoint_directory':'checkpoints'})
    stores={};locks={};sync={'cloud_cache_hits':0,'local_cache_hits':0,'uploaded':0,'upload_pending':0}
    recent=[]
    recent_path=root/'recent.json'
    if recent_path.is_file():
        try:recent=json.loads(recent_path.read_text()).get('images',[])[-12:]
        except (OSError,ValueError):recent=[]
    def store_for(row):
        if record_store is not None:return record_store
        bucket=row.get('source',{}).get('bucket')
        if not bucket:return None
        if bucket not in stores:
            from .cloud_cache import GCSRecordStore
            stores[bucket]=GCSRecordStore(bucket)
        return stores[bucket]
    streaming=hasattr(rows,'__aiter__');queued={'total':0 if streaming else len(rows)}
    async def feed():
        try:
            if streaming:
                async for row in rows:
                    queued['total']+=1;await queue.put(copy.deepcopy(row))
            else:
                for row in rows:await queue.put(copy.deepcopy(row))
        finally:
            for _ in range(int(concurrency)):await queue.put(None)
    async def worker():
        while True:
            row=await queue.get()
            if row is None:queue.task_done();return
            inference_lock=None;lock_acquired=False;identity=None;phase='source_validation'
            try:
                path=Path(row['image']);path=(path if path.is_absolute() else base/path).resolve()
                if not path.is_file():raise ValueError('Source image missing')
                sha=_hash(path)
                if row.get('sha256') and row['sha256']!=sha:raise ValueError('Source image checksum mismatch')
                identity=inference_cache_key(row,sha,provider,model);cache=root/'results'/(identity+'.json')
                inference_lock=locks.setdefault(identity,asyncio.Lock());await inference_lock.acquire();lock_acquired=True
                paid_cache=root/'paid-responses'/(identity+'.json');recovered_paid=None
                legacy=root/'results'/(digest([row.get('id'),sha,model,threshold,PROMPT])+'.json')
                store=store_for(row);result=None
                phase='cache_lookup'
                if cache.exists() or legacy.exists():
                    result=json.loads((cache if cache.exists() else legacy).read_text());sync['local_cache_hits']+=1
                elif paid_cache.exists():
                    recovered_paid=json.loads(paid_cache.read_text());sync['local_cache_hits']+=1
                elif store is not None:
                    result=await _call(store.get,identity)
                    legacy_hit=False
                    if result is None:
                        result=await _call(store.get,legacy.stem);legacy_hit=result is not None
                    if result is not None:
                        if result.get('sha256')!=sha or result.get('model')!=model or (not legacy_hit and result.get('cache_key')!=identity):
                            raise ValueError('Cloud inference cache identity mismatch; paid request not issued')
                        if legacy_hit:
                            result['cache_migration']={'source_key':legacy.stem,'generation_settings':'legacy_not_recorded','raw_text':'not_reconstructed'}
                            if result.get('raw_response') is None:
                                result['raw_response']=await _call(store.get,legacy.stem,kind='logs')
                        sync['cloud_cache_hits']+=1;save(cache,result)
                if result is None and recovered_paid is None:
                    previous=await _previous_prompt_result(row,sha,provider,model,root,store)
                    if previous is not None:
                        result,identity,cache=previous
                        sync['previous_prompt_cache_hits']=sync.get('previous_prompt_cache_hits',0)+1
                if result is None:
                    phase='provider_generation'
                    if recovered_paid is None and getattr(provider,'cfg',{}).get('require_airborne_verified'):
                        eligibility=row.get('eligibility',{})
                        if not (row.get('airborne_eligible') is True and eligibility.get('status')=='eligible'
                                and eligibility.get('airborne_eligible') is True
                                and eligibility.get('telemetry',{}).get('sha256')):
                            raise ValueError('Fresh airborne telemetry eligibility required; no paid request issued')
                    output=recovered_paid['model_output'] if recovered_paid else await provider.generate('robotics_er2_thermal_labels',PROMPT,[path],model=model)
                    raw_log=recovered_paid.get('raw_response') if recovered_paid else await _provider_log(provider,path,model)
                    # Persist the paid response BEFORE schema validation/upload.
                    paid={'cache_key':identity,'model':model,'sha256':sha,'source':row.get('source',{}),**prompt_metadata(PROMPT),
                          'model_output':output,'raw_response':raw_log,'status':'paid_response_saved'}
                    if row.get('paid_response_recovery'):paid['paid_response_recovery']=copy.deepcopy(row['paid_response_recovery'])
                    save(paid_cache,paid)
                    phase='paid_schema_validation'
                    try:
                        canonical_output,normalizations=normalize_thermal_output(output)
                        classification=classify_response(canonical_output,threshold)
                    except ValueError:
                        paid['status']='invalid_model_response';save(cache,paid)
                        if store is not None:
                            try:await _call(store.write_frame,identity,paid,raw_log=raw_log)
                            except Exception:sync['upload_pending']+=1
                        raise
                    image_ref=str(path.relative_to(data_root().resolve())) if path.is_relative_to(data_root().resolve()) else str(path)
                    result={**row,'image':image_ref,'sha256':sha,'model':model,'cache_key':identity,'raw_response':raw_log,**prompt_metadata(PROMPT),
                        'provenance':'user_accepted_model',
                        'user_threshold':threshold,'human_reviewed':False,'identity_verified':False,
                        'model_output':canonical_output,'detections':canonical_output['detections'],
                        'boxes':[{'label':'provisional mine-like thermal candidate','bbox':d['bbox'],'confidence':d['confidence'],'evidence':d.get('evidence','')} for d in classification['accepted']],
                        'uncertain_boxes':[{'bbox':d['bbox'],'confidence':d['confidence']} for d in classification['uncertain']],
                        'zero_candidates':classification['zero_candidates'],'labelled_at':utcnow(),
                        'training_excluded_reason':'uncertain_regions_require_ignore_support' if classification['accepted'] and classification['uncertain'] else None}
                    if getattr(provider,'cfg',{}).get('thinking_level') is not None:
                        result['inference_profile']={k:provider.cfg.get(k) for k in ('thinking_level','reserve_output_tokens','temperature','inference_revision')}
                    if normalizations:
                        result['original_model_output']=output
                        result['output_normalization']={'policy':'lossless_bbox_key_aliases_v1','changes':normalizations,'coordinates_changed':False,'confidence_changed':False}
                    if classification['zero_candidates']:
                        result['screening']={'status':'accepted','decision':'no_mine_detected','model':model,
                            'detections':[],'rule':'zero_candidates_at_any_confidence','human_reviewed':False,'verified_negative':False}
                    save(cache,result)
                # Revalidate cached outputs, never convert failed/refused JSON to backgrounds.
                phase='cache_revalidation'
                repair_path=root/'repair-index'/(identity+'.json')
                if result.get('status')=='invalid_model_response' and not repair_path.exists() and store is not None:
                    remote_repair=await _call(store.get,digest(['repair-index',identity]),kind='progress')
                    if remote_repair is not None:save(repair_path,remote_repair)
                if result.get('status')=='invalid_model_response' and repair_path.is_file():
                    repair=json.loads(repair_path.read_text());replacement=repair.get('replacement_cache_key','')
                    if not re.fullmatch('[a-f0-9]{64}',replacement):raise ValueError('Invalid paid-cache repair identity')
                    alternate=root/'results'/(replacement+'.json')
                    if not alternate.exists() and store is not None:
                        remote_alternate=await _call(store.get,replacement)
                        if remote_alternate is not None:save(alternate,remote_alternate)
                    revised=json.loads(alternate.read_text())
                    if revised.get('model')!=model or revised.get('sha256')!=sha or revised.get('status')=='invalid_model_response':raise ValueError('Paid-cache repair provenance mismatch')
                    classify_response(revised['model_output'],threshold)
                    result=revised;identity=replacement;cache=alternate
                if result.get('model')!=model or result.get('sha256')!=sha:
                    raise ValueError('Local/cloud paid cache model or image hash mismatch')
                if result.get('status')=='invalid_model_response':
                    if store is not None:
                        try:await _call(store.write_frame,identity,result,raw_log=result.get('raw_response'))
                        except Exception:sync['upload_pending']+=1
                    raise ValueError('Previously paid response has invalid schema; no new paid request issued')
                classify_response(result['model_output'],threshold)
                already_uploaded=(result.get('id')==row.get('id') and result.get('source')==row.get('source') and result.get('frame_index')==row.get('frame_index') and result.get('cloud_sync',{}).get('status')=='uploaded')
                result.update(cache_key=identity,sha256=sha,model=model,user_threshold=threshold)
                result.update(result_prompt_metadata(result,assumed_prompt=PROMPT))
                if result.get('raw_response') is None:
                    result['raw_response']=await _provider_log(provider,path,model)
                result.update({k:v for k,v in row.items() if k not in ('sha256','image','label_prompt_version','label_prompt_sha256')})
                result['image']=str(path.relative_to(data_root().resolve())) if path.is_relative_to(data_root().resolve()) else str(path)
                phase='preview_export'
                preview=Path(result.get('annotated',''));preview=preview if preview.is_absolute() else data_root()/preview
                if not already_uploaded or not preview.is_file():_live_preview(result,path,root,identity)
                save(cache,result)
                if already_uploaded:sync['resumed_verified']=sync.get('resumed_verified',0)+1
                if store is not None and not already_uploaded:
                    try:
                        uris=await _call(store.write_frame,identity,result,raw_log=result.get('raw_response'))
                        frame_key=digest([identity,row.get('id'),row.get('source',{}),row.get('frame_index')])
                        uris['frame_label']=await _call(store.put,frame_key,{**result,'raw_log_uri':uris.get('raw_log'),'inference_cache_key':identity},kind='labels')
                        result['cloud_sync']={'status':'uploaded','uris':uris};sync['uploaded']+=1
                    except Exception as exc:
                        result['cloud_sync']={'status':'upload_pending','error':redacted_error(exc)};sync['upload_pending']+=1
                    save(cache,result)
                save(root/'checkpoints'/(_frame_checkpoint_key(row)+'.json'),{
                    'status':'completed','run_id':run_id,'id':row.get('id'),'sha256':sha,
                    'cache_key':identity,'result':'results/'+identity+'.json',
                    **result_prompt_metadata(result),
                    'cloud_sync_status':result.get('cloud_sync',{}).get('status','not_requested'),'at':utcnow()})
                save(root/'prompt-audit'/(_frame_checkpoint_key(row)+'.json'),{
                    'run_id':run_id,'id':row.get('id'),'sha256':sha,'source':row.get('source',{}),
                    'frame_index':row.get('frame_index'),'cache_key':identity,
                    **result_prompt_metadata(result),'observed_at':utcnow()})
                results.append(result)
                public={key:copy.deepcopy(result[key]) for key in ('id','image','annotated','model','boxes','uncertain_boxes',
                    'source','timestamp_s','source_offset_s','frame_index','split','user_threshold','label_prompt_version','label_prompt_sha256') if key in result}
                public['cloud_sync_status']=result.get('cloud_sync',{}).get('status','not_requested')
                recent[:]=[item for item in recent if item.get('id')!=public.get('id')]
                recent.append(public);recent[:]=recent[-12:]
                save(recent_path,{'updated_at':utcnow(),'model':model,'images':recent})
            except Exception as exc:
                failure=_persist_failure(root,row,exc,phase,run_id,identity)
                failures.append(failure);failure_reasons[failure['category']]+=1
                if on_failure is not None:await _call(on_failure,row,exc,phase)
            finally:
                if lock_acquired:inference_lock.release()
                queue.task_done()
                save(root/'progress.json',{'status':'running','completed':len(results),'failed':len(failures),'failure_reasons':dict(failure_reasons),'last_failure':failures[-1] if failures else None,'run_id':run_id,'total':queued['total'],'total_known':not streaming,'model':model,'cloud_sync':sync,'sampling_rate_is_not_api_throughput':True})
    try:await asyncio.gather(feed(),*(worker() for _ in range(int(concurrency))))
    finally:
        if owned:await provider.close()
    results.sort(key=lambda row:row.get('id',''))
    positives=[r for r in results if r['boxes']];uncertain=[r for r in results if r['uncertain_boxes']]
    zero=[r for r in results if r['zero_candidates']]
    training=[r for r in results if (r['boxes'] and not r['uncertain_boxes']) or r['zero_candidates']]
    final_status='partial' if failures else 'upload_pending' if sync['upload_pending'] else 'completed'
    document={'metadata':{'status':'passed' if not failures else 'partial','model':model,'provenance':'user_accepted_model','cloud_sync':sync,'run_id':run_id,'failure_reasons':dict(failure_reasons),
        'user_threshold':threshold,'human_reviewed':False,'identity_verified':False,
        'counts':{'labelled':len(results),'positive_frames':len(positives),'uncertain_frames':len(uncertain),'zero_candidate_frames':len(zero),'training_frames':len(training),'failed':len(failures)},
        'interpretation':'Machine labels accepted by user confidence rule; no independently verified mine identities'},'images':training,'failures':failures}
    save(root/'manifest.json',document);save(root/'positives.json',{'images':positives});save(root/'uncertain.json',{'images':uncertain});save(root/'backgrounds.json',{'images':zero,'verified_negative':False})
    export_annotations(results,root/'export')
    save(root/'progress.json',{'status':final_status,**document['metadata']['counts'],'model':model,'cloud_sync':sync,'run_id':run_id,'failure_reasons':dict(failure_reasons)})
    return {'status':final_status,'manifest':str(root/'manifest.json'),**document['metadata']['counts'],'model':model,'cloud_sync':sync}

def export_annotations(rows,output_dir):
    import shutil
    root=Path(output_dir);folder=root/'labelme';folder.mkdir(parents=True,exist_ok=True)
    coco={'info':{'description':'Robotics ER 2 machine labels, user threshold 0.6; not human-reviewed'},
          'images':[],'annotations':[],'categories':[{'id':1,'name':'provisional mine-like thermal candidate'}]}
    for i,row in enumerate(rows,1):
        path=Path(row['image']);path=path if path.is_absolute() else data_root()/path
        safe_id=digest([row['id'],row['sha256']])[:32];name=safe_id+path.suffix.lower()
        shutil.copyfile(path,folder/name)
        with Image.open(path) as im:width,height=im.size;annotated=im.convert('RGB')
        draw=ImageDraw.Draw(annotated);shapes=[]
        for detection in row['detections']:
            box=detection['bbox'];accepted=detection['confidence']>=.6
            points=[[box['x_min']*width/1000,box['y_min']*height/1000],[box['x_max']*width/1000,box['y_max']*height/1000]]
            shapes.append({'label':'mine_candidate' if accepted else 'uncertain_thermal_signature','points':points,'group_id':None,
                'shape_type':'rectangle','flags':{'user_accepted':accepted,'human_reviewed':False},
                'description':json.dumps({'model':MODEL,'confidence':detection['confidence'],'evidence':detection.get('evidence','')})})
            draw.rectangle(points[0]+points[1],outline='#3ddd68' if accepted else '#f5aa42',width=2)
            draw.text((points[0][0],max(0,points[0][1]-12)),f"{detection['confidence']:.2f}",fill='white')
            if accepted:
                x,y=points[0];w=points[1][0]-x;h=points[1][1]-y
                coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':i,'category_id':1,'bbox':[x,y,w,h],
                    'area':w*h,'iscrowd':0,'model_confidence':detection['confidence'],'provenance':'user_accepted_model'})
        save(folder/(safe_id+'.json'),{'version':'5.4.1','flags':{'user_accepted_model':True,'human_reviewed':False},
            'shapes':shapes,'imagePath':name,'imageData':None,'imageHeight':height,'imageWidth':width,
            'model':MODEL,'user_threshold':.6,'split':row.get('split'),'source':row.get('source',{})})
        (root/'annotated').mkdir(exist_ok=True);annotated.save(root/'annotated'/(safe_id+'.jpg'),quality=95)
        coco['images'].append({'id':i,'file_name':'labelme/'+name,'width':width,'height':height,'split':row.get('split'),
            'source':row.get('source',{}),'sha256':row['sha256'],'uncertain_regions':[d['bbox'] for d in row['detections'] if d['confidence']<.6]})
    save(root/'coco.json',coco)
    return {'labelme_images':len(rows),'coco_annotations':len(coco['annotations'])}

async def label_video_sources(sources,output_dir=None,*,provider=None,fps=5,threshold=.6,model=MODEL,split_seed=17,session_splits=None,concurrency=8,max_frames=None):
    if model!=MODEL:raise ValueError('Requested Robotics model is mandatory; fallback forbidden')
    if float(fps)!=5:raise ValueError('User requested source-time sampling at exactly 5 FPS')
    root=Path(output_dir or data_root()/'labelling'/'robotics-er2-5fps').resolve();root.mkdir(parents=True,exist_ok=True)
    if isinstance(sources,(str,Path)):
        document=json.loads(Path(sources).read_text());sources=document.get('videos',document.get('sources',[]))
    splits=freeze_session_splits([r['session_id'] for r in sources],root/'session-splits.json',split_seed,session_splits)
    frames=[];source_records=[];source_failures=[]
    async def stream():
        for number,original_source in enumerate(sources):
            source=copy.deepcopy(original_source)
            save(root/'sampling-progress.json',{'status':'extracting','source_index':number,'total_sources':len(sources),
                'sampled_frames':len(frames),'failed_sources':len(source_failures),'current_object':source.get('object',Path(source.get('path','')).name)})
            try:
                if not source.get('path'):
                    cached=await asyncio.to_thread(cache_gcs_object,source['bucket'],source['object'],root/'source-cache',source.get('generation'))
                    source.update(path=cached['path'],generation=cached['generation'])
                origin=None
                timestamp_path=source.get('timestamps_path')
                if timestamp_path and Path(timestamp_path).is_file():
                    with Path(timestamp_path).open() as timestamps:first=json.loads(timestamps.readline())
                    origin=int(first['received_at_ns'])/1e9
                elif timestamp_path:
                    raise ValueError('Declared receipt timestamp file is missing; not silently falling back to container time')
                original={k:source[k] for k in ('bucket','object','generation','session_id','segment','timing_mode','wallclock_alignment_available','gps_alignment_available','receipt_timestamp_recovery','recovered_variant_of') if k in source}
                rows=await asyncio.to_thread(extract_video_samples,source['path'],root/'frames'/digest(original)[:20],
                    session_id=source['session_id'],source=original,timestamps_path=timestamp_path,segment=source.get('segment'),
                    fps=fps,split=splits[source['session_id']],max_frames=max_frames,sample_origin_s=origin)
                frames.extend(rows);source_records.append({**original,'samples':len(rows),'status':'sampled'})
                for row in rows:yield row
            except Exception as exc:
                source_failures.append({'source':{k:v for k,v in source.items() if k!='path'},'error':redacted_error(exc)})
            finally:
                save(root/'sampling-manifest.json',{'sampling_fps':fps,'inference_fps':None,'density_is_not_inference_throughput':True,
                    'sources':source_records,'images':frames,'session_splits':splits,'failed_sources':source_failures})
        save(root/'sampling-progress.json',{'status':'completed' if not source_failures else 'partial',
            'total_sources':len(sources),'sampled_frames':len(frames),'failed_sources':len(source_failures)})
    result=await label_frames(stream(),root/'labels',provider,model,threshold,concurrency)
    if source_failures:result.update(status='partial',failed_sources=len(source_failures))
    return result
