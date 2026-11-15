#!/usr/bin/env python3
"""Cloud-independent airborne-frame labelling; never samples raw videos."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,signal,threading,time
from datetime import datetime
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def instant(value):return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()

def metadata_record_store(bucket,prefix='robotics-er2-labels/5fps-v1'):
    """GCS uses the reviewed VM identity; Gemini uses its separate API key."""
    from aerorozum.cloud_cache import GCSRecordStore
    from google.auth.compute_engine import Credentials
    from google.auth.transport.requests import Request
    class Store(GCSRecordStore):
        def __init__(self):
            super().__init__(bucket,prefix)
            self.credentials=Credentials(service_account_email='94824551802-compute@developer.gserviceaccount.com',scopes=['https://www.googleapis.com/auth/cloud-platform'])
        def _headers(self):
            with self._lock:
                if not self.credentials.valid:self.credentials.refresh(Request())
                return {'Authorization':'Bearer '+self.credentials.token,'Content-Type':'application/json'}
        def _invalidate(self,rejected_headers):
            with self._lock:
                if rejected_headers.get('Authorization')=='Bearer '+str(self.credentials.token):self.credentials.token=None
    return Store()

def validate(plan_path):
    from aerorozum import labelling,providers,label_prompts,label_schema
    from aerorozum.common import data_root
    plan=json.loads(plan_path.read_text());root=data_root().resolve()
    if not all(plan.get(k) is True for k in ('authorized','laptop_paid_calls_settled','prompt_selection_completed','api_lease_released')):raise ValueError('Launch authorization, settled laptop calls and released prompt-selection lease required')
    if plan.get('model')!=labelling.MODEL:raise ValueError('Exact Robotics ER2 model required')
    if time.time()<instant(plan['not_before_utc']):raise ValueError('Rate-window cooldown pending')
    if time.time()>=instant(plan['deadline_utc']):raise ValueError('Worker deadline reached')
    for name,module in [('labelling',labelling),('providers',providers),('label_prompts',label_prompts),('label_schema',label_schema)]:
        if sha(module.__file__)!=plan[name+'_source_sha256']:raise ValueError('Deployed '+name+' differs from reviewed code')
    if hashlib.sha256(labelling.PROMPT.encode()).hexdigest()!=plan['prompt_sha256']:raise ValueError('Selected prompt identity differs')
    manifest=Path(plan['manifest_path']).resolve()
    if not manifest.is_relative_to(root) or sha(manifest)!=plan['manifest_sha256']:raise ValueError('Frozen airborne manifest differs')
    doc=json.loads(manifest.read_text());rows=doc['images']
    if doc.get('status')!='airborne_only' or not rows:raise ValueError('Nonempty airborne-only manifest required')
    if doc.get('subset')!='pending_only' or doc.get('policy',{}).get('version')!='airborne-only-v2-primary-state':raise ValueError('Corrected pending-only airborne policy required')
    full=(root/doc['full_manifest']).resolve()
    if not full.is_relative_to(root) or sha(full)!=doc['full_manifest_sha256']:raise ValueError('Full eligible manifest receipt differs')
    audit=Path(plan['eligibility_manifest_path']).resolve()
    if not audit.is_relative_to(root) or sha(audit)!=doc['eligibility_manifest_sha256']:raise ValueError('Eligibility audit differs')
    seen=set()
    for row in rows:
        if row.get('airborne_eligible') is not True or row.get('eligibility',{}).get('status')!='eligible':raise ValueError('Frame lacks airborne eligibility')
        if not row.get('id') or row['id'] in seen:raise ValueError('Missing or duplicate frame identity')
        seen.add(row['id']);path=Path(row['image'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Frame path escapes artifact tree')
        image=(root/path).resolve()
        if not image.is_relative_to(root) or not image.is_file() or sha(image)!=row['sha256']:raise ValueError('Staged image missing or changed')
        if not all(row.get('source',{}).get(k) is not None for k in ('bucket','object','generation')):raise ValueError('Immutable source provenance required')
    return plan,rows

async def run(plan_path):
    from aerorozum.common import config,data_root,redacted_error,save,utcnow,ROOT
    from aerorozum.cloud_cache import GCSRecordStore
    from aerorozum.label_archive import archive_pass
    from aerorozum.labelling import label_frames
    from aerorozum.providers import GeminiProvider
    root=data_root();status=root/'pipeline/label-worker.json';plan,rows=validate(plan_path)
    credential=Path(os.environ['CREDENTIALS_DIRECTORY'])/'gemini-api-key'
    # systemd presents LoadCredential files read-only as0440 inside the
    # service's private credential directory; the backing file remains0600.
    if not credential.is_file() or credential.stat().st_mode&0o137:raise ValueError('Private systemd Gemini credential required')
    for name in ('GEMINI_API_KEY','GOOGLE_API_KEY','GOOGLE_GENAI_USE_VERTEXAI'):os.environ.pop(name,None)
    os.environ['AEROROZUM_GEMINI_KEY_FILE']=str(credential)
    output=root/'labelling/robotics-er2-airborne/labels';output.mkdir(parents=True,exist_ok=True)
    cfg={**config()['gemini'],**json.loads((ROOT/'configs/labelling.json').read_text())}
    drain_path=root/'pipeline/drain-label-worker.json'
    cfg.update(requests_per_minute=120,input_tokens_per_minute=500000,concurrency=16,api_version='v1beta',thinking_budget=0,drain_path=str(drain_path),require_airborne_verified=True)
    if {row['source']['bucket'] for row in rows}!={cfg['gcs_bucket']}:raise ValueError('Source bucket differs from reviewed archive bucket')
    provider=GeminiProvider(cfg,output_dir=output/'provider')
    if not callable(getattr(provider,'request_drain',None)) or not callable(getattr(provider,'drain',None)):raise ValueError('Tested graceful provider drain support required')
    requested=asyncio.Event();loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,requested.set)
    store=metadata_record_store(cfg['gcs_bucket'])
    stopped=threading.Event();archive=metadata_record_store(cfg['gcs_bucket'],prefix='robotics-er2-labels/5fps-raw-audit')
    snapshot={'status':'running','started_at':utcnow(),'host':'google_compute_engine','model':plan['model'],'frames':len(rows),'manifest_sha256':plan['manifest_sha256'],'prompt_sha256':plan['prompt_sha256'],'paid_calls_settled':False,'laptop_independent':True,'auth':'explicit_hackathon_api_key','cloud_archive':'gs://'+cfg['gcs_bucket']+'/robotics-er2-labels/'}
    snapshot['prompt_version']=plan['label_prompt_version']
    def mirror():
        while not stopped.is_set():
            progress={}
            try:progress=json.loads((output/'progress.json').read_text())
            except (OSError,ValueError):pass
            save(status,{**snapshot,'updated_at':utcnow(),'label_progress':progress})
            try:archive_pass(output,archive,output/'archive-state.json',workers=4)
            except Exception as error:save(output/'archive-error.json',{'at':utcnow(),'error':redacted_error(error)})
            stopped.wait(30)
    thread=threading.Thread(target=mirror,daemon=True);thread.start()
    task=asyncio.create_task(label_frames(rows,output,provider=provider,concurrency=16,image_root=root,record_store=store))
    try:
        while not task.done():
            if requested.is_set() or drain_path.exists() or time.time()>=instant(plan['deadline_utc']):
                snapshot.update(status='draining');provider.request_drain();await provider.drain();task.cancel()
                try:await task
                except asyncio.CancelledError:pass
                snapshot.update(status='drained',paid_calls_settled=True);break
            await asyncio.wait({task},timeout=1)
        if task.done() and not task.cancelled():snapshot.update(status='completed',result=task.result(),paid_calls_settled=True)
    except Exception as error:snapshot.update(status='failed',error=redacted_error(error));raise
    finally:
        provider.request_drain();await provider.drain();await provider.close()
        stopped.set();await asyncio.to_thread(thread.join,180)
        try:await asyncio.to_thread(archive_pass,output,archive,output/'archive-state.json',workers=4)
        except Exception as error:snapshot['archive_error']=redacted_error(error)
        save(status,{**snapshot,'updated_at':utcnow(),'paid_calls_settled':True})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    try:
        version=json.loads(a.plan.read_text())['label_prompt_version']
        if version not in ('thermal-v1-conservative','thermal-v2-recall-calibrated'):raise ValueError('Unknown reviewed prompt version')
        os.environ['AEROROZUM_LABEL_PROMPT_VERSION']=version
        if a.validate_only:
            plan,rows=validate(a.plan);print(json.dumps({'status':'validated_no_paid_calls','frames':len(rows),'manifest_sha256':plan['manifest_sha256']}));return
        asyncio.run(run(a.plan))
    except Exception as error:
        from aerorozum.common import redacted_error
        print(json.dumps({'status':'failed','error':redacted_error(error)}),flush=True);raise SystemExit(1)

if __name__=='__main__':main()
