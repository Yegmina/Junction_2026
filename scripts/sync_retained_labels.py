#!/usr/bin/env python3
import json,time
from aerorozum.common import data_root,redacted_error
from aerorozum.label_sync import retry_uploads
while True:
 for name in ('robotics-er2-5fps','robotics-er2-airborne-priority','robotics-er2-airborne'):
  try:print(json.dumps({'run':name,**retry_uploads(data_root()/'labelling'/name/'labels')}),flush=True)
  except Exception as ex:print(json.dumps({'run':name,'error':redacted_error(ex)}),flush=True)
 time.sleep(30)
