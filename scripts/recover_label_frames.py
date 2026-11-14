#!/usr/bin/env python3
import asyncio,json,signal,subprocess
from aerorozum.common import ROOT,config,data_root,save,utcnow,redacted_error
from aerorozum.label_recovery import recover_pass,paid_permission,seed_recovery_window
from aerorozum.providers import GeminiProvider
stop=False

def drain(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,drain);signal.signal(signal.SIGINT,drain)
r=data_root();labels=r/'labelling/robotics-er2-5fps/labels';manifest=labels.parent/'recovery/manifest.json';lease=r/'pipeline/leases/label-recovery.json'
def main_active():return subprocess.run(['systemctl','--user','is-active','--quiet','aerorozum-robotics-label-bucket.service']).returncode==0
def pause(reason,code=None):
 current=json.loads(lease.read_text()) if lease.exists() else {}
 current.update(status='paused',paused_at=utcnow(),pause_reason=reason,status_code=code);save(lease,current);save(labels.parent/'recovery/quota-pause.json',current)
async def failed_frame(row,error,phase):
 if 'GCS cache read failed: HTTP 401' in str(error):pause('Repeated GCS401 after automatic token refresh; preserving row for cache-first retry',401)
async def main():
 provider=None
 async def gate():
  if stop:raise RuntimeError('Recovery drain requested; no new paid request issued')
  if not paid_permission(lease,main_active())['allowed']:raise RuntimeError('Recovery labeling allowance paused or expired; no new paid request issued')
 async def api_error(error,code):
  if code in (429,401):pause('Recovery API rejection: '+redacted_error(error),code)
 def factory(cfg,permission):
  nonlocal provider
  if provider is None:
   provider=GeminiProvider({**cfg,'concurrency':8,'requests_per_minute':permission['requests_per_minute'] or 30},output_dir=labels/'provider');provider.before_queue=gate;provider.before_request=gate;provider.on_request_error=api_error
   restored=seed_recovery_window(provider,labels,json.loads(manifest.read_text()));save(labels.parent/'recovery/window-restoration.json',{'at':utcnow(),'restored_attempts':restored,'method':'conservative recent paid response/error arrivals','rate_limit_preserved_across_handoff':True})
  elif permission['allowed']:provider.limiter.rpm=permission['requests_per_minute']
  return provider
 try:
  while not stop:
   state=await recover_pass(manifest,labels,lease,main_running=main_active(),provider_factory=factory,on_failure=failed_frame);print(json.dumps(state),flush=True)
   if state['status']=='completed':break
   for _ in range(15):
    if stop:break
    await asyncio.sleep(1)
 finally:
  if provider is not None:await provider.close()
asyncio.run(main())
