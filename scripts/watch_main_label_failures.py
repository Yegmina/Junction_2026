#!/usr/bin/env python3
"""Checkpoint legacy in-memory failures on count changes; never interrupts labeling."""
import json,os,subprocess,sys,time
from pathlib import Path
from aerorozum.common import ROOT,data_root,digest,save,utcnow,redacted_error
from aerorozum.labelling import _failure_kind,_frame_checkpoint_key
r=data_root();base=r/'labelling/robotics-er2-5fps';labels=base/'labels';state_path=base/'recovery/main-failure-observer.json';last=-1
while True:
 try:
  progress=json.loads((labels/'progress.json').read_text());count=progress.get('failed',0);pid=int(subprocess.check_output(['systemctl','--user','show','aerorozum-robotics-label-bucket.service','-p','MainPID','--value'],text=True).strip() or '0')
  if pid==36477 and count!=last:
   subprocess.run(['sudo','-n',sys.executable,str(ROOT/'scripts/recover_retained_label_failures.py'),str(pid)],check=True,capture_output=True,text=True,timeout=60)
   snapshot=json.loads((base/'diagnosis'/f'retained-failures-{pid}.json').read_text());rows={x['id']:x for x in json.loads((base/'sampling-manifest.json').read_text())['images']}
   for source in (base/'frames').glob('*/sampling.json'):rows.update({x['id']:x for x in json.loads(source.read_text()).get('images',[])})
   existing={json.loads(p.read_text()).get('id') for p in (labels/'failures').glob('*.json')};new=[]
   for failed in snapshot['records']:
    if failed['id'] in existing or failed['id'] not in rows:continue
    row=rows[failed['id']];kind=_failure_kind(failed['error']);unissued=failed['error'].startswith('GCS cache read failed:')
    item={'status':'failed','id':row['id'],'sha256':row['sha256'],'source':row['source'],'row':row,'category':kind,'error':redacted_error(failed['error']),'phase':'cache_lookup' if unissued else 'provider_or_model_response','paid_request_not_issued':unissued,'retryable':unissued,'original_failure_at':None,'observed_at':utcnow(),'recovered_from_live_process':pid};item['failure_id']=digest(['legacy-retained',pid,item['id'],item['error']]);save(labels/'failures'/(item['failure_id']+'.json'),item)
    checkpoint=labels/'checkpoints'/(_frame_checkpoint_key(row)+'.json')
    if not checkpoint.exists() or json.loads(checkpoint.read_text()).get('status')!='completed':save(checkpoint,item)
    new.append(item)
   if new:
    with (labels/'failure-journal.jsonl').open('a') as f:
     for item in new:f.write(json.dumps(item)+'\n')
     f.flush();os.fsync(f.fileno())
    save(labels/f'recovered-failure-batch-{count}.json',{'at':utcnow(),'process_untouched':True,'failures':new})
    follow=base/'recovery/followup-manifest.json';old=json.loads(follow.read_text()) if follow.exists() else {'metadata':{'scope':'new pre-inference failures; kept separate from frozen5757 probe'},'images':[]};known={x['id'] for x in old['images']};old['images'] +=[item['row'] for item in new if item['paid_request_not_issued'] and item['id'] not in known];save(follow,old)
   last=count
   state={'status':'watching','observed_failed_counter':count,'retained_records_recovered':len(snapshot['records']),'new_records_persisted':len(new),'main_pid':pid,'process_modified':False}
  else:state={'status':'watching_legacy' if pid==36477 else 'legacy_worker_finished_or_replaced','observed_failed_counter':count,'main_pid':pid,'process_modified':False}
  save(state_path,{**state,'checked_at':utcnow()})
 except Exception as ex:save(state_path,{'status':'monitor_error','checked_at':utcnow(),'error':redacted_error(ex),'process_modified':False})
 time.sleep(15)
