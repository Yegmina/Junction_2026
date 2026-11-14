#!/usr/bin/env python3
"""Mirror committed cloud validation progress into the local research dashboard."""
import concurrent.futures,json,subprocess,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path
from aerorozum.common import data_root,save,utcnow
from aerorozum.preflight import sdk_path
BUCKET='aerorozum-thermal-rl-avid-relic-509408-e4';PREFIX='runs/real-transfer-seed17-v1/validation/'
def main():
 root=data_root();dest=root/'validation/real-transfer-seed17-v1';dest.mkdir(parents=True,exist_ok=True)
 job={'id':'validation-zone7-seed17-v1','kind':'held_out_validation','stage':'zone7_base_sft_grpo','status':'running','completed':0,'total':264,'created_at':utcnow(),'seed':17,'zone':'zone7','test_zones_accessed':False,'cloud_prefix':'gs://'+BUCKET+'/'+PREFIX}
 save(root/'jobs'/('validation-zone7-seed17-v1.json'),{**job,'updated_at':utcnow(),'detail':'Held-out zone-7 inference launched; awaiting next committed cloud progress snapshot.'})
 token=None;expiry=0
 def headers():
  nonlocal token,expiry
  if time.monotonic()>expiry:token=subprocess.check_output([sdk_path(),'auth','print-access-token','--quiet'],text=True).strip();expiry=time.monotonic()+2400
  return {'Authorization':'Bearer '+token}
 def get(name):
  url='https://storage.googleapis.com/storage/v1/b/'+BUCKET+'/o/'+urllib.parse.quote(name,safe='')+'?alt=media'
  with urllib.request.urlopen(urllib.request.Request(url,headers=headers()),timeout=60) as r:return r.read()
 while True:
  try:
   raw=get(PREFIX+'status.json');status=json.loads(raw);save(dest/'status.json',status)
   try:
    failure=json.loads(get(PREFIX+'failure.json'));save(dest/'failure.json',failure);job.update(status='failed',updated_at=utcnow(),error=failure['error']);save(root/'jobs'/('validation-zone7-seed17-v1.json'),job);break
   except urllib.error.HTTPError as error:
    if error.code!=404:raise
   job.update(status='completed' if status['status']=='completed' else 'running',completed=status.get('completed',0),version=status.get('version'),metrics=status.get('partial_metrics',{}),updated_at=utcnow(),remote_updated_at=status.get('at'),detail=status.get('limitation','Model loading; validation predictions are not yet available.'))
   versions={}
   for version in ('base','real_sft','real_grpo'):
    try:versions[version]=json.loads(get(PREFIX+version+'/summary.json'))
    except urllib.error.HTTPError as error:
     if error.code!=404:raise
   job['versions']=versions
   if status.get('comparison'):job['summary']=status['comparison']
   save(root/'jobs'/('validation-zone7-seed17-v1.json'),job)
   print(json.dumps({'completed':job['completed'],'total':264,'status':job['status'],'at':utcnow()}),flush=True)
   if status['status']=='completed':
    url='https://storage.googleapis.com/storage/v1/b/'+BUCKET+'/o?prefix='+urllib.parse.quote(PREFIX,safe='');names=[]
    while url:
     with urllib.request.urlopen(urllib.request.Request(url,headers=headers()),timeout=60) as r:d=json.load(r)
     names += [x['name'] for x in d.get('items',[])]
     token_page=d.get('nextPageToken');url=url.split('&pageToken=')[0]+('&pageToken='+urllib.parse.quote(token_page,safe='')) if token_page else None
    def download(name):
     relative=Path(name[len(PREFIX):]);target=(dest/relative).resolve();target.relative_to(dest.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(get(name))
    with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(download,names))
    save(dest/'mirror.json',{'status':'completed','objects':len(names),'cloud_prefix':job['cloud_prefix'],'at':utcnow()});break
  except urllib.error.HTTPError as ex:
   if ex.code!=404:print(json.dumps({'mirror_error':str(ex)}),flush=True)
  except Exception as ex:print(json.dumps({'mirror_error':str(ex)[:500]}),flush=True)
  time.sleep(30)
if __name__=='__main__':main()
