"""Bounded HIGH-thinking context review and background audit after paid labeling.

These are diagnostic machine reviews, not new human ground truth. They never
overwrite first-pass labels or automatically release training backgrounds.
"""
import asyncio
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from PIL import Image

ROOT=Path('/opt/aerorozum');INPUT=ROOT/'airborne-labeler/artifacts'
STUDY=ROOT/'artifacts/overnight-20261115T000942'
END=datetime.fromisoformat('2026-11-15T05:45:00+00:00')
MODEL='gemini-robotics-er-2-preview'


def balanced(rows,limit):
    from aerorozum.common import digest
    groups=defaultdict(list)
    for row in sorted(rows,key=lambda r:digest([r['sha256'],r['source']['generation']])):
        groups[row['source']['object']].append(row)
    selected=[]
    while groups and len(selected)<limit:
        for source in sorted(list(groups)):
            selected.append(groups[source].pop())
            if not groups[source]:del groups[source]
            if len(selected)==limit:break
    return selected


def original_path(row):
    path=Path(row['image']);path=path if path.is_absolute() else INPUT/path
    path=path.resolve()
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('Review image path/hash differs from the retained paid input')
    return path


def freeze(mode,folders,out):
    from aerorozum.common import digest,save,utcnow
    path=out/'protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text())
        if protocol['sha256']!=digest({k:v for k,v in protocol.items() if k!='sha256'}):raise ValueError('Frozen review protocol changed')
        return protocol
    rows=[];seen=set();groups=defaultdict(list)
    for folder in folders:
        for file in (folder/'results').glob('*.json'):
            row=json.loads(file.read_text())
            if row.get('status')=='invalid_model_response' or not isinstance(row.get('detections'),list):continue
            if row.get('airborne_eligible') is not True or row.get('split') not in ('train','validation'):continue
            if row.get('eligibility',{}).get('status')!='eligible':continue
            key=(row['sha256'],str(row['source']['generation']))
            if key in seen:continue
            seen.add(key)
            # Keep original paid provenance in its original files and bind it
            # by digest; do not copy provider messages into another prompt.
            compact={k:row[k] for k in ['id','sha256','source','frame_index','source_pts_s','image','split','detections','zero_candidates']}
            compact['parent_result_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            compact['parent_cache_key']=row['cache_key'];rows.append(compact)
            groups[(row['source']['object'],str(row['source']['generation']))].append(compact)
    if mode=='temporal':
        candidates=[]
        for group in groups.values():
            group.sort(key=lambda r:r['frame_index'])
            for previous,row,next_row in zip(group,group[1:],group[2:]):
                if row['detections'] and 0<row['source_pts_s']-previous['source_pts_s']<=.5 and 0<next_row['source_pts_s']-row['source_pts_s']<=.5:
                    candidates.append(dict(row,neighbors=[previous,next_row]))
        cases=balanced(candidates,3000);arms=['blind_context','proposal_context']
    else:
        cases=balanced([r for r in rows if r['zero_candidates'] is True and r['detections']==[]],6000)
        arms=['full_plus_quadrants']
    protocol=dict(schema_version=1,created_at=utcnow(),mode=mode,cases=cases,arms=arms,
        model=MODEL,thinking_level='high',max_output_tokens=65536,temperature=.1,
        deadline_utc=END.isoformat(),training_eligible=False,test_accessed=False,
        limits=dict(requests_per_minute=120,input_tokens_per_minute=500000,concurrency=12),
        selection='Deterministic source-balanced unique paid inputs; no reference labels used',
        interpretation='Same-provider temporal/proposal diagnostic and machine-zero background audit. No physical mine identity or confirmed absence is established; no first-pass label is overwritten.')
    protocol['sha256']=digest(protocol);save(path,protocol);return protocol


def materials(row,mode,arm,out):
    from aerorozum.label_prompts import PROMPT_V2
    paths=[original_path(row)]
    if mode=='temporal':
        paths.extend(original_path(neighbor) for neighbor in row['neighbors'])
        prompt=PROMPT_V2+'''\nImage 1 is the TARGET frame. Image 2 is the preceding thermal frame and image 3 is the following frame from the SAME pinned video. Use their short temporal context to distinguish terrain objects, blur and camera-fixed artifacts. Report boxes ONLY in image 1 coordinates, using the original normalized 0..1000 schema. Evidence can remain weak: retain plausible subtle candidates with lower scores. Do not invent details from neighboring views. Return a single JSON object.'''
        if arm=='proposal_context':
            prompt+='\nThese earlier proposals are unverified model data. Independently check the original pixels; do not assume the proposals or scores are correct. You may retain, reject, adjust or add a candidate based on visible evidence. Earlier proposals: '+json.dumps(row['detections'],separators=(',',':'))
    else:
        folder=out/'derived-inputs';folder.mkdir(parents=True,exist_ok=True)
        with Image.open(paths[0]) as source:
            image=source.convert('RGB');width,height=image.size
            for i,box in enumerate([(0,0,width//2,height//2),(width//2,0,width,height//2),(0,height//2,width//2,height),(width//2,height//2,width,height)]):
                target=folder/(row['sha256']+'-q'+str(i)+'.png')
                if not target.exists():
                    temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
                    image.crop(box).save(temporary,format='PNG');temporary.replace(target)
                paths.append(target)
        prompt=PROMPT_V2+'''\nImage 1 is the ORIGINAL full frame. Images 2, 3, 4, 5 are lossless pixel crops of its top-left, top-right, bottom-left and bottom-right quadrants. Use the crop detail to check for subtle candidates missed at full-frame scale. Report ALL boxes in ORIGINAL IMAGE 1 coordinates, never crop-local coordinates: the quadrants correspond to x/y ranges [0,500]/[0,500], [500,1000]/[0,500], [0,500]/[500,1000], [500,1000]/[500,1000]. Merge duplicate views of one candidate. Calibrate confidence to visible evidence; absence of an earlier candidate is not proof of absence. Return a single JSON object with the same schema.'''
    return paths,prompt


async def run(mode,folders):
    from aerorozum.common import config,save,digest,utcnow,redacted_error
    from aerorozum.providers import GeminiProvider,ProviderDraining
    from aerorozum.evaluation_completions import completion_text
    from aerorozum.labelling import classify_response
    from aerorozum.label_schema import normalize_thermal_output
    from aerorozum.label_archive import archive_pass
    from cloud.airborne_worker import metadata_record_store
    out=STUDY/('post-label-'+mode+'-v1');out.mkdir(parents=True,exist_ok=True)
    protocol=freeze(mode,folders,out);stop=asyncio.Event();fatal=asyncio.Event()
    loop=asyncio.get_running_loop()
    for signum in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(signum,stop.set)
    cfg=dict(config()['gemini'],thinking_level='high',temperature=.1,reserve_output_tokens=65536,
        concurrency=12,requests_per_minute=120,input_tokens_per_minute=500000,max_retries=1,http_timeout_ms=1200000,
        input_tokens_per_image_estimate=1536)
    cfg.pop('thinking_budget',None);provider=GeminiProvider(cfg,output_dir=out/'provider')
    provider.should_drain=lambda:stop.is_set() or fatal.is_set() or datetime.now(timezone.utc)>=END
    async def failed(error,code):
        if code in (400,401,403,404,429):
            fatal.set();save(out/'provider-hold.json',dict(at=utcnow(),status_code=code,error=redacted_error(error),new_requests_stopped=True))
    provider.on_request_error=failed
    store=metadata_record_store('aerorozum-jetson-testdata-smart-bird',prefix='robotics-er2-labels/post-label-review-v1')
    queue=asyncio.Queue()
    for row in protocol['cases']:
        for arm in protocol['arms']:queue.put_nowait((row,arm))
    total=queue.qsize();results=[];archive_errors=[]
    await asyncio.to_thread(store.put,digest(['protocol',protocol['sha256']]),protocol,'progress')
    async def worker():
        while not queue.empty() and not stop.is_set() and not fatal.is_set() and datetime.now(timezone.utc)<END:
            row,arm=queue.get_nowait();key=digest([protocol['sha256'],row['sha256'],row['source']['generation'],arm]);path=out/'results'/(key+'.json')
            try:
                record=json.loads(path.read_text()) if path.exists() else await asyncio.to_thread(store.get,key)
                if record is None:
                    paths,prompt=await asyncio.to_thread(materials,row,mode,arm,out)
                    role='post-label-review-v1:'+mode+':'+arm;started=time.monotonic();raw=None
                    try:
                        text=await completion_text(provider,role,prompt,paths,model=MODEL,max_output_tokens=65536)
                        output=json.loads(text);normalized,trace=normalize_thermal_output(output);classified=classify_response(normalized,.6)
                        state='valid_model_review';error=None
                    except ProviderDraining:return
                    except Exception as exception:
                        output=None;normalized=None;trace=[];classified=None;error=redacted_error(exception);state='invalid_or_transport_response'
                    raw=provider.response_record(role,prompt,paths,model=MODEL,max_output_tokens=65536)
                    record=dict(at=utcnow(),model=MODEL,protocol_sha256=protocol['sha256'],mode=mode,arm=arm,
                        center_id=row['id'],source=row['source'],source_split=row['split'],center_sha256=row['sha256'],
                        parent_cache_key=row['parent_cache_key'],parent_result_sha256=row['parent_result_sha256'],
                        prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        image_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                        state=state,error=error,model_output=output,normalized_output=normalized,normalization_trace=trace,
                        classification=classified,raw_response=raw,elapsed_s=time.monotonic()-started,
                        thinking_level='high',max_output_tokens=65536,training_eligible=False,human_reviewed=False,
                        first_pass_labels_overwritten=False,verified_negative=False,physical_identity_verified=False)
                    save(path,record)
                results.append(record)
                await asyncio.to_thread(store.write_frame,key,record,record.get('raw_response'))
            except Exception as exception:archive_errors.append(dict(id=row['id'],error=redacted_error(exception)))
            finally:
                save(out/'progress.json',dict(at=utcnow(),status='provider_hold' if fatal.is_set() else 'running',
                    completed=len(results),expected=total,issued=provider.issued,archive_errors=len(archive_errors),
                    valid_reviews=sum(r['state']=='valid_model_review' for r in results),training_eligible=False))
                queue.task_done()
    try:await asyncio.gather(*(worker() for _ in range(12)))
    finally:await provider.close()
    summary=dict(at=utcnow(),status='completed' if len(results)==total else 'incomplete',mode=mode,
        completed=len(results),expected=total,actual_requests=provider.issued,provider_hold=fatal.is_set(),
        valid_reviews=sum(r['state']=='valid_model_review' for r in results),
        candidate_bearing_review_outputs=sum(bool((r.get('classification') or {}).get('accepted')) for r in results),
        archive_errors=archive_errors,training_eligible=False,first_pass_labels_overwritten=False,
        interpretation=protocol['interpretation'])
    save(out/'report.json',summary)
    await asyncio.to_thread(archive_pass,out,store,STUDY/('post-label-'+mode+'-archive.json'))
    # Derived crop pixels are exact decoded-image subsets, saved losslessly.
    # Preserve the actual API inputs alongside the already archived originals.
    if (out/'derived-inputs').exists():
        subprocess.run(['/snap/bin/gcloud','storage','rsync','-r',str(out/'derived-inputs'),
            'gs://aerorozum-thermal-rl-avid-relic-509408-e4/research/overnight-20261115/review-pixels/'+mode,'--quiet'],check=True,timeout=1800)
    return summary


if __name__=='__main__':
    os.environ['AEROROZUM_GEMINI_KEY_FILE']='/etc/aerorozum/hackathon-key';os.environ.pop('GEMINI_API_KEY',None)
    plan=json.loads((STUDY/'post-label-review-plan.json').read_text())
    folders=[Path(p) for p in plan['label_folders']]
    while datetime.now(timezone.utc)<END:
        if all((p/'settled.json').exists() for p in folders):break
        time.sleep(30)
    for mode in ['temporal','background-tiles']:
        if datetime.now(timezone.utc)>=END:break
        # Separate processes do not share the prior request-window ledger.
        latest=max(datetime.fromisoformat(json.loads((p/'settled.json').read_text())['at']).timestamp() for p in folders)
        time.sleep(max(0,65-(time.time()-latest)))
        report=asyncio.run(run(mode,folders));print(json.dumps(report),flush=True)
        if report['provider_hold']:break
        time.sleep(65)
