#!/usr/bin/env python3
"""Observe label health without constructing any inference provider or retrying."""
import json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/opt/aerorozum/airborne-labeler/artifacts')
LABELS=ROOT/'labelling/robotics-er2-airborne/labels'

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def main():
    inactive=0
    while True:
        now=time.time();progress=read(LABELS/'progress.json');worker=read(ROOT/'pipeline/label-worker.json')
        active=subprocess.run(['systemctl','is-active','--quiet','aerorozum-airborne-labeler.service']).returncode==0
        raw=list((LABELS/'provider/raw').glob('*.json'));recent=0
        for path in raw:
            try:recent+=path.stat().st_mtime>=now-60
            except FileNotFoundError:pass
        health={'checked_at':datetime.now(timezone.utc).isoformat(),'status':worker.get('status','waiting'),'worker_active':active,'completed_frames':progress.get('completed',progress.get('labelled',0)),'failed_frames':progress.get('failed',0),'failure_categories':progress.get('failure_reasons',{}),'latest_failure':progress.get('last_failure'),'canonical_upload_pending':progress.get('cloud_sync',{}).get('upload_pending',0),'raw_response_files':len(raw),'model_response_files_last60s':recent,'prompt_version':worker.get('prompt_version'),'monitor_issues_model_requests':False,'automatic_repairs':False}
        LABELS.mkdir(parents=True,exist_ok=True);temporary=LABELS/'.worker-health.tmp';temporary.write_text(json.dumps(health,indent=2));os.replace(temporary,LABELS/'worker-health.json')
        print(json.dumps(health),flush=True)
        inactive=0 if active else inactive+1
        if not active and (worker.get('status') in ('completed','drained','failed') or inactive>=4):return
        time.sleep(30)

if __name__=='__main__':main()
