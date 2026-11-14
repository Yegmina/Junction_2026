from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .common import config, data_root, redacted_error, save, secret, utcnow
from .jobs import cost_status

SPONSORS = [
    ('codex','Codex','active','Parallel implementation, tests and development evidence'),
    ('gemini','Gemini API','not_configured','Teacher, reviewer and frontier campaign baseline'),
    ('gemma','Gemma','planned','Main trainable visual survey model'),
    ('nemotron','Nemotron','not_configured','3.5 Super VL failure analysis'),
    ('verda','Verda','not_configured','Independent compute worker'),
    ('modal','Modal','not_configured','Public GPU inference'),
    ('entire','Entire','not_configured','Development sessions and code graph'),
    ('craci','CRACI','not_configured','CI and reproducible builds'),
    ('cloudinary','Cloudinary','not_configured','Media inspection and annotated clips'),
    ('tigerdata','Tiger Data','not_configured','Research time series'),
    ('atlas','MongoDB Atlas','not_configured','Searchable evidence records'),
    ('aiven','Aiven','not_configured','Worker event transport'),
    ('context','Context.dev','not_configured','Source-grounded custom campaigns'),
    ('oxylabs','Oxylabs','not_configured','Live source verification'),
    ('elevenlabs','ElevenLabs','not_configured','Evidence-grounded narration'),
    ('vultr','Vultr','not_configured','Public demo gateway'),
]


def sdk_path():
    return shutil.which('gcloud') or str(Path.home()/'.local/google-cloud-cli-install/google-cloud-sdk/bin/gcloud')


def cloud_identity():
    try:
        p = subprocess.run([sdk_path(),'auth','list','--filter=status:ACTIVE','--format=value(account)'],capture_output=True,text=True,timeout=15)
        return dict(status='authenticated' if p.stdout.strip() else 'not_authenticated',project=config().get('cloud',{}).get('project','avid-relic-509408-e4'))
    except (OSError,subprocess.TimeoutExpired):
        return dict(status='not_configured')


def snapshot(check_models=False):
    root = data_root(); cfg = config()
    gemini=dict(id='gemini',name='Hackathon Gemini API',status='configured' if secret('GEMINI_API_KEY','AEROROZUM_GEMINI_KEY_FILE') else 'not_configured',backend='developer_api',enterprise_fallback=False,quota_verified=cfg['gemini'].get('project_quota_verified',False))
    catalog = root/'providers/gemini-models.json'
    if check_models and gemini['status']=='configured':
        from .providers import GeminiProvider
        try:
            provider=GeminiProvider();names=provider.available_models();provider.client.close()
            save(catalog,dict(status='connected',models=names,backend='developer_api',checked_at=utcnow()))
        except Exception as e:save(catalog,dict(status='failed',error=redacted_error(e),checked_at=utcnow()))
    if catalog.exists():
        d=json.loads(catalog.read_text());gemini.update(status=d['status'],models=d.get('models',[]),detail=d.get('error','Explicit API-key authentication; separate from cloud compute'))
    sponsors=[dict(id=i,name=n,status=('connected' if i=='gemini' and gemini['status']=='connected' else s),detail=d) for i,n,s,d in SPONSORS]
    physics_reports=list((root/'physics').glob('*verification*.json'))
    physics={'solver':'DuMux 3.10.0 planned / Python experimental reference','status':'development','research_accepted':False,'validation':'No accepted production field yet'}
    if physics_reports:
        physics.update(verification=json.loads(max(physics_reports,key=lambda p:p.stat().st_mtime).read_text()))
    training_file=root/'training/status.json'
    training=json.loads(training_file.read_text()) if training_file.exists() else dict(status='not_started',metrics=[])
    requests=list((root/'gemini/responses').glob('*.json'))
    usage={'completed_requests':len(requests),'total_tokens':0,'by_role':{}}
    for path in requests:
        try:
            row=json.loads(path.read_text());usage['total_tokens']+=row.get('usage',{}).get('total_token_count') or 0
            role=row.get('role','unknown');usage['by_role'][role]=usage['by_role'].get(role,0)+1
        except (OSError,ValueError):pass
    cloud_file=root/'cloud/inventory.json'
    cloud=json.loads(cloud_file.read_text()) if cloud_file.exists() else {'instances':[]}
    state=dict(updated_at=utcnow(),experiment=cfg['experiment'],counts=cfg['counts'],providers=[gemini,dict(id='google_cloud',name='Google Cloud compute',**cloud_identity())],models=[dict(id=cfg['primary_model'],role='Primary trainable Gemma',status=training['status']),dict(id=cfg['reference_model'],role='Gemma reference',status='available_for_download'),dict(id=cfg['gemini']['teacher_model'],role='Hackathon teacher',status=gemini['status']),dict(id=cfg['gemini']['bulk_model'],role='High-volume teacher/reviewer',status=gemini['status'])],physics=physics,training=training,sponsors=sponsors,budget=cost_status(),gemini_usage=usage,cloud=cloud)
    save(root/'status.json',state)
    return state
