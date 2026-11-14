#!/usr/bin/env python3
import json,signal,time
from aerorozum.common import save,utcnow,redacted_error
from aerorozum.real_ingester import RealIngester
stop=False

def drain(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,drain);signal.signal(signal.SIGINT,drain)
ingester=RealIngester()
while not stop:
 try:
  result=ingester.scan();print(json.dumps({'at':result['updated_at'],'counts':result['counts'],'cloud_sync':result['cloud_sync']}),flush=True)
 except Exception as ex:save(ingester.output/'ingester-error.json',{'status':'failed_pass','at':utcnow(),'error':redacted_error(ex)});print(redacted_error(ex),flush=True)
 for _ in range(15):
  if stop:break
  time.sleep(1)
