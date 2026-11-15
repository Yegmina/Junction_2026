"""Versioned HIGH-thinking relabeling after development-only prompt selection.

Original labels are preserved. Every image needs its frozen fresh airborne
telemetry binding; source-clock 5 FPS is independent of inference throughput.
"""
import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/opt/aerorozum')
INPUT=ROOT/'airborne-labeler/artifacts'
OUTPUT=INPUT/'labelling/robotics-er2-high-thinking-v1'
STUDY=ROOT/'artifacts/overnight-20261115T000942/robotics-reasoning-v1'
DEADLINE=datetime.fromisoformat('2026-11-15T05:45:00+00:00')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare():
    from aerorozum.common import save,digest,utcnow
    path=OUTPUT/'frozen-inputs.json'
    if path.exists():return json.loads(path.read_text())
    manifests=sorted((INPUT/'pipeline/inputs/airborne-snapshots').glob('*/airborne-eligible.json'))
    if len(manifests)!=2:raise ValueError('Expected the two reviewed eligible snapshots; review new arrivals explicitly')
    rows=[];seen=set();missing=[];duplicates=0;excluded_test=0
    # The global catalog provides byte-identical fallback locations for frames
    # whose original extraction directory was not transferred from the laptop.
    catalog=json.loads((ROOT/'artifacts/pipeline/inputs/vm-ingestion/candidate.json').read_text())
    locations={r['sha256']:ROOT/'artifacts'/r['image'] for r in catalog['images']}
    for manifest in manifests:
        for original in json.loads(manifest.read_text())['images']:
            row=dict(original)
            if row['split'] not in ('train','validation'):
                excluded_test+=1;continue
            eligibility=row.get('eligibility',{})
            if not (row.get('airborne_eligible') is True and eligibility.get('status')=='eligible'
                    and eligibility.get('policy_version')=='airborne-only-v2-primary-state'
                    and eligibility.get('telemetry',{}).get('sha256')):
                raise ValueError('An unbound or non-airborne frame entered the reviewed snapshot')
            image=INPUT/row['image']
            if not image.is_file():image=locations.get(row['sha256'],image)
            if not image.is_file():
                missing.append(dict(id=row['id'],sha256=row['sha256'],image=row['image']));continue
            if sha(image)!=row['sha256']:raise ValueError('Source pixels changed before HIGH relabeling')
            identity=(row['sha256'],str(row['source']['generation']))
            if identity in seen:
                duplicates+=1;continue
            seen.add(identity)
            row['image']=str(image.resolve())
            rows.append(row)
    # Save missing identities for recovery, never quietly claim all-data coverage.
    result=dict(created_at=utcnow(),images=rows,missing=missing,duplicates_avoided=duplicates,
        held_out_test_frames_excluded=excluded_test,snapshots={str(p):sha(p) for p in manifests},
        original_labels_overwritten=False,thinking_level='high',max_output_tokens=65536,
        prompt_selection_rule='Highest mean archived-zone-7 reference F2 across three HIGH replicas; prefer v2 on a tie. Require all six HIGH arms complete.',
        dispatch_deadline=DEADLINE.isoformat(),unique_eligible_available=len(rows))
    result['sha256']=digest(result);save(path,result)
    return result


def select_prompt():
    from aerorozum.common import save,utcnow
    choice=OUTPUT/'prompt-selection.json'
    if choice.exists():return json.loads(choice.read_text())['prompt_version']
    protocol=json.loads((STUDY/'protocol.json').read_text())
    references=[r for r in protocol['images'] if r.get('zone')=='zone7']
    ids={r['id'] for r in references}
    groups=defaultdict(list);counts=defaultdict(int)
    for path in (STUDY/'results').glob('*.json'):
        row=json.loads(path.read_text())
        if row['thinking_level']!='high' or row['status']=='transport_error':continue
        key=(row['prompt_version'],row['replica']);counts[key]+=1
        if row['id'] in ids:groups[key].append(row)
    expected={(prompt,replica) for prompt in protocol['prompts'] for replica in range(protocol['replicas'])}
    if set(counts)!=expected or any(counts[k]!=len(protocol['images']) for k in expected):
        raise ValueError('HIGH prompt selection awaits complete balanced development arms')
    scores=defaultdict(list)
    for (prompt,replica),rows in groups.items():
        tp=sum(r['score']['tp'] for r in rows);fp=sum(r['score']['fp'] for r in rows);fn=sum(r['score']['fn'] for r in rows)
        scores[prompt].append(5*tp/max(1,5*tp+4*fn+fp))
    means={p:sum(values)/len(values) for p,values in scores.items()}
    selected=sorted(means,key=lambda p:(-means[p],p))[0]
    from collections import Counter
    save(choice,dict(at=utcnow(),prompt_version=selected,mean_archived_development_f2=means,
        per_replica_archived_development_f2=dict(scores),study_protocol_sha256=protocol['sha256'],
        reference_provenance_counts=dict(Counter(r.get('provenance') for r in references)),
        selected_before_relabeling=True,test_accessed=False,
        interpretation='Development label agreement selects this prompt; physical mine accuracy and native-flight transfer remain unverified'))
    return selected


async def run(prompt,document,deadline=DEADLINE):
    os.environ['AEROROZUM_DATA_ROOT']=str(INPUT)
    os.environ['AEROROZUM_LABEL_PROMPT_VERSION']=prompt
    os.environ['AEROROZUM_GEMINI_KEY_FILE']='/etc/aerorozum/hackathon-key'
    os.environ.pop('GEMINI_API_KEY',None)
    from aerorozum.common import config,save,utcnow
    from aerorozum.labelling import label_frames
    from aerorozum.providers import GeminiProvider
    from aerorozum.label_archive import archive_pass
    from cloud.airborne_worker import metadata_record_store
    settings=dict(config()['gemini'],thinking_level='high',temperature=.1,reserve_output_tokens=65536,
        concurrency=16,requests_per_minute=120,input_tokens_per_minute=500000,max_retries=1,
        http_timeout_ms=1200000,require_airborne_verified=True,reuse_previous_prompt=False,
        inference_revision='airborne-high-thinking-v1')
    settings.pop('thinking_budget',None)
    provider=GeminiProvider(settings,output_dir=OUTPUT/'provider')
    stop=asyncio.Event();loop=asyncio.get_running_loop()
    for signum in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(signum,stop.set)
    provider.should_drain=lambda:stop.is_set() or datetime.now(timezone.utc)>=deadline
    async def error_callback(error,code):
        if code in (400,401,403,404):
            save(OUTPUT/'configuration-error.json',dict(at=utcnow(),status_code=code))
            stop.set()
    provider.on_request_error=error_callback
    store=metadata_record_store('aerorozum-jetson-testdata-smart-bird',prefix='robotics-er2-labels/high-thinking-airborne-v1')
    archive_stop=asyncio.Event()
    async def archive():
        while not archive_stop.is_set():
            await asyncio.to_thread(archive_pass,OUTPUT,store,OUTPUT.parent/'high-thinking-archive-state.json')
            try:await asyncio.wait_for(archive_stop.wait(),timeout=60)
            except asyncio.TimeoutError:pass
    async def rows():
        for row in document['images']:
            if stop.is_set() or datetime.now(timezone.utc)>=deadline:break
            yield row
    archiver=asyncio.create_task(archive())
    try:
        report=await label_frames(rows(),OUTPUT,provider,concurrency=16,image_root=INPUT,record_store=store)
        save(OUTPUT/'settled.json',dict(at=utcnow(),prompt_version=prompt,thinking_level='high',
            report=report,actual_requests=provider.issued,available_unique_frames=len(document['images']),
            missing_source_pixels=len(document['missing']),deadline_reached=datetime.now(timezone.utc)>=deadline,
            original_labels_overwritten=False,test_accessed=False))
    finally:
        await provider.close();archive_stop.set();await archiver
        await asyncio.to_thread(archive_pass,OUTPUT,store,OUTPUT.parent/'high-thinking-archive-state.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();document=prepare()
    print(json.dumps({k:v for k,v in document.items() if k not in ('images','missing')},indent=2),flush=True)
    if not args.prepare_only:
        while datetime.now(timezone.utc)<DEADLINE:
            if (STUDY/'report.json').exists() and json.loads((STUDY/'report.json').read_text()).get('status')=='completed':
                report=json.loads((STUDY/'report.json').read_text())
                # The previous process has its own rate-window ledger. Allow
                # that last request window to clear before handing over.
                finished=datetime.fromisoformat(report['at']).timestamp()
                time.sleep(max(0,65-(time.time()-finished)))
                selected=select_prompt();asyncio.run(run(selected,document));break
            time.sleep(30)
