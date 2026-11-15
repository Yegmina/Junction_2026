"""Archive immutable inputs and terminal runs on the existing VM/bucket only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

ROOT=Path('/opt/aerorozum');ART=ROOT/'artifacts';STUDY=ART/'overnight-20261115T000942'
BUCKET='aerorozum-thermal-rl-avid-relic-509408-e4'
PREFIX='research/overnight-20261115'


def sync(path,relative):
    subprocess.run(['/snap/bin/gcloud','storage','rsync','-r',
        '--exclude=(^|/)(agent/opencode/.*|__pycache__/.*|auth\\.json|\\.env)',str(path),
        'gs://'+BUCKET+'/'+PREFIX+'/'+relative,'--quiet'],check=True,timeout=3600)


def archive_inputs():
    from aerorozum.common import save,utcnow
    from cloud.airborne_worker import metadata_record_store
    out=ROOT/'archives/overnight-airborne-inputs-v1';out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'airborne-labeler/artifacts/labelling/robotics-er2-high-thinking-v1/frozen-inputs.json').read_text())
    images={};bindings=[]
    for row in frozen['images']:
        source=Path(row['image']);name=row['sha256']+source.suffix.lower();target=out/name
        with source.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=row['sha256']:
                raise ValueError('Paid input pixels changed before archival')
        if not target.exists():os.link(source,target)
        images[name]=dict(sha256=row['sha256'],md5=base64.b64encode(hashlib.md5(source.read_bytes()).digest()).decode(),size=source.stat().st_size)
        bindings.append(dict(id=row['id'],file=name,source=row['source'],frame_index=row['frame_index'],eligibility=row['eligibility']))
    save(out/'manifest.json',dict(at=utcnow(),frozen_input_sha256=frozen['sha256'],
        originals_preserved=True,hard_links_only=True,images=images,bindings=bindings))
    relative='airborne-input-pixels-v1';sync(out,relative)
    # Verify every uploaded source pixel using GCS's server-computed digest,
    # byte count and immutable content-addressed filename. No inference calls.
    store=metadata_record_store(BUCKET);indexed={};page=None
    while True:
        query=dict(prefix=PREFIX+'/'+relative+'/',maxResults=1000,fields='items(name,size,md5Hash,generation),nextPageToken')
        if page:query['pageToken']=page
        req=urllib.request.Request('https://storage.googleapis.com/storage/v1/b/'+BUCKET+'/o?'+urllib.parse.urlencode(query),headers=store._headers())
        with urllib.request.urlopen(req,timeout=60) as response:data=json.load(response)
        for obj in data.get('items',[]):indexed[Path(obj['name']).name]=obj
        page=data.get('nextPageToken')
        if not page:break
    failures=[]
    for name,local in images.items():
        remote=indexed.get(name,{})
        if int(remote.get('size',-1))!=local['size'] or remote.get('md5Hash')!=local['md5']:
            failures.append(name)
    receipt=dict(at=utcnow(),status='verified' if not failures else 'verification_failed',
        unique_pixel_objects=len(images),frame_bindings=len(bindings),failures=failures,
        uri='gs://'+BUCKET+'/'+PREFIX+'/'+relative+'/',server_digest='MD5',local_content_identity='SHA256',
        no_local_originals_deleted=True,no_api_inference_issued=True)
    save(STUDY/'input-pixels-cloud-verification.json',receipt)
    store.put(hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest(),receipt,'progress')
    if failures:raise ValueError('Input-pixel archive verification failed')
    return receipt


def terminal_paths():
    paths=[]
    for family in ['postsubmission-visual-replication-v1','postsubmission-expanded-visual-v1']:
        for p in (ART/family).glob('*/status.json'):
            state=json.loads(p.read_text())
            if state.get('status') not in ('completed','failed'):continue
            paths.append((p.parent,'experiments/'+str(p.parent.relative_to(ART))))
            spec=json.loads((p.parent/'specification.json').read_text())
            training=ART/'training'/spec['namespace']
            if training.is_dir():paths.append((training,'training/'+spec['namespace']))
    for p in (ROOT/'harbor/jobs').glob('overnight-*/result.json'):
        result=json.loads(p.read_text());stats=result.get('stats',{})
        if not result.get('finished_at') or stats.get('n_running_trials') or stats.get('n_pending_trials'):continue
        paths.append((p.parent,'native-missions/'+p.parent.name))
    report=STUDY/'robotics-reasoning-v1/report.json'
    if report.exists() and json.loads(report.read_text()).get('status')=='completed':
        paths.append((report.parent,'robotics-development-study-v1'))
    return paths


if __name__=='__main__':
    from aerorozum.common import save,utcnow
    ledger_path=STUDY/'terminal-archive-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(completed={},errors=[])
    if not (STUDY/'input-pixels-cloud-verification.json').exists():
        print(json.dumps(archive_inputs()),flush=True)
    for name in ['reasoning-source-v1','high-thinking-source-v1','optimized-training-source-v1','expanded-visual-source-v1','expanded-visual-source-v2']:
        path=ROOT/'postsubmission'/name
        if path.exists():sync(path,'sources/'+name)
    deadline=datetime.fromisoformat('2026-11-15T06:30:00+00:00').timestamp()
    while time.time()<deadline:
        for path,relative in terminal_paths():
            if relative in ledger['completed']:continue
            try:
                sync(path,relative);ledger['completed'][relative]=dict(at=utcnow(),uri='gs://'+BUCKET+'/'+PREFIX+'/'+relative)
            except Exception as error:
                ledger['errors'].append(dict(at=utcnow(),path=relative,error=str(error)[:1000]))
            save(ledger_path,ledger)
        time.sleep(60)
