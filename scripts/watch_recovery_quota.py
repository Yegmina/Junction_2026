#!/usr/bin/env python3
"""Pause only the extra recovery allowance on observed quota/auth rejections."""
import json,time
from datetime import datetime
from aerorozum.common import data_root,save,utcnow
r=data_root();lease=r/'pipeline/leases/label-recovery.json';provider=r/'labelling/robotics-er2-5fps/labels/provider/errors';seen=set()
while True:
 if lease.exists():
  d=json.loads(lease.read_text());created=datetime.fromisoformat(d['created_at']).timestamp() if d.get('created_at') else 0
  if d.get('status')=='granted':
   for p in provider.glob('*.json'):
    stamp=(str(p),p.stat().st_mtime_ns)
    if stamp in seen or p.stat().st_mtime<created:continue
    seen.add(stamp)
    try:
     e=json.loads(p.read_text());text=str(e.get('error','')).strip();code=e.get('status_code')
     if code not in (401,429):code=429 if text.startswith('429 ') else 401 if text.startswith('401 ') else None
     if code:
      d.update(status='paused',paused_at=utcnow(),status_code=code,pause_reason='quota/auth rejection observed during bounded recovery probe',error_record=str(p.relative_to(r)));save(lease,d);save(r/'labelling/robotics-er2-5fps/recovery/quota-pause.json',d);print(json.dumps({'paused':True,'code':code,'at':utcnow()}),flush=True);break
    except (ValueError,OSError):continue
 time.sleep(1)
