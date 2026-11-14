#!/usr/bin/env python3
"""Sequential follow-up under the freed main allowance, never overlapping main."""
import asyncio,json,signal,subprocess,socket
from datetime import datetime,timezone
from aerorozum.common import ROOT,config,data_root,save,utcnow
from aerorozum.providers import GeminiProvider
from aerorozum.labelling import label_frames
stop=False

def drain(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,drain);signal.signal(signal.SIGINT,drain)
r=data_root();root=r/'labelling/quarantine-recovery'
async def main():
 if subprocess.run(['systemctl','--user','is-active','--quiet','aerorozum-robotics-label-bucket.service']).returncode==0:
  save(root/'status.json',{'status':'waiting_for_main_completion','at':utcnow(),'paid_inference_issued':False});return
 out=root/'labels';out.mkdir(parents=True,exist_ok=True)
 for name in ('results','paid-responses','provider','checkpoints','failures','repair-index'):
  target=r/'labelling/robotics-er2-5fps/labels'/name;target.mkdir(parents=True,exist_ok=True)
  if not (out/name).exists():(out/name).symlink_to(target,target_is_directory=True)
 sampling=json.loads((r/'labelling/robotics-er2-5fps/sampling-progress.json').read_text())
 if sampling.get('status') not in ('completed','partial') or 'source_index' in sampling:
  save(root/'status.json',{'status':'waiting_for_full_main_pass','at':utcnow(),'paid_inference_issued':False});return
 doc=json.loads((root/'manifest.json').read_text());cfg={**config()['gemini'],**json.loads((ROOT/'configs/labelling.json').read_text())};cfg.update(concurrency=16,thinking_budget=0,api_version='v1beta')
 provider=GeminiProvider(cfg,output_dir=r/'labelling/robotics-er2-5fps/labels/provider')
 async def gate():
  global stop
  deadline=json.loads((r/'pipeline/leases/label-recovery.json').read_text()).get('expires_at')
  if stop or (deadline and datetime.now(timezone.utc)>=datetime.fromisoformat(deadline.replace('Z','+00:00'))):
   stop=True
   raise RuntimeError('Recovery drain or deadline reached; no new paid request issued')
 provider.before_queue=gate;provider.before_request=gate
 save(root/'status.json',{'status':'running','at':utcnow(),'total':len(doc['images']),'allowance':'main120RPM after main full-source pass','main_overlap':False})
 try:
  # Recover new durable transient failures from the full main pass first.
  while not stop:
   pending=[]
   for path in (r/'labelling/robotics-er2-5fps/labels/checkpoints').glob('*.json'):
    entry=json.loads(path.read_text())
    if entry.get('status')=='failed' and entry.get('retryable') and entry.get('row') and 'identity mismatch' not in entry.get('error',''):pending.append(entry['row'])
   if not pending:break
   try:
    await asyncio.to_thread(socket.getaddrinfo,'storage.googleapis.com',443)
    await asyncio.to_thread(socket.getaddrinfo,'generativelanguage.googleapis.com',443)
   except OSError:
    save(root/'status.json',{'status':'waiting_for_network','at':utcnow(),'retry_frames':len(pending)});await asyncio.sleep(15);continue
   save(root/'status.json',{'status':'retrying_main_transient_failures','at':utcnow(),'retry_frames':len(pending)})
   await label_frames(pending[:128],out,provider=provider,concurrency=16)
   await asyncio.sleep(5)
  if stop:
   save(root/'status.json',{'status':'paused_after_drain_or_deadline','at':utcnow(),'source_frames_still_saved':len(doc['images'])});return
  result=await label_frames(doc['images'],out,provider=provider,concurrency=16)
  save(root/'status.json',{'at':utcnow(),**result,'source_segments':8,'receipt_aligned_segments':6,'video_time_only_segments':2,'main_overlap':False})
 finally:await provider.close()
asyncio.run(main())
