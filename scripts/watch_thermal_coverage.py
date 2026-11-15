#!/usr/bin/env python3
"""Read-only cloud inventory and safe triggering after the initial labeling chain."""
import json,re,subprocess,urllib.request,urllib.parse,urllib.error
from datetime import datetime,timezone
from aerorozum.cloud_cache import GCSRecordStore
from aerorozum.common import data_root,save,digest,utcnow
r=data_root();out=r/'labelling/additional-bucket-sources';bucket='aerorozum-jetson-testdata-smart-bird';auth=GCSRecordStore(bucket)
def get(url):
 for attempt in range(2):
  h=auth._headers()
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=60) as response:return json.load(response)
  except urllib.error.HTTPError as ex:
   if ex.code==401 and attempt==0:auth._invalidate(h);continue
   raise
original=json.loads((r/'source-videos/labelling-sources-original-150.json').read_text())['videos'];known={(s['object'],str(s['generation'])) for s in original};objects=[];url='https://storage.googleapis.com/storage/v1/b/'+bucket+'/o?prefix=jetson%2F&maxResults=1000'
while url:
 d=get(url);objects.extend(d.get('items',[]));token=d.get('nextPageToken');url=url.split('&pageToken=')[0]+('&pageToken='+urllib.parse.quote(token,safe='')) if token else None
raw=[s for s in objects if re.search(r'/thermal-\d+(?:-recovered)?\.avi$',s['name'])];normal=[s for s in raw if '/recovered/' not in s['name']];selected=[s for s in raw if (s['name'],str(s['generation'])) not in known]
ledger=out/'processed-versions.json';processed=set(json.loads(ledger.read_text()).get('versions',[])) if ledger.exists() else set();duplicate=json.loads((out/'recovered-raw-dedup.json').read_text()) if (out/'recovered-raw-dedup.json').exists() else {};pending=[s for s in selected if digest([s['name'],str(s['generation'])]) not in processed and not(s['name']==duplicate.get('recovered_object') and str(s['generation'])==duplicate.get('recovered_generation') and duplicate.get('identical_decoded_RGB_sequence'))]
active=lambda unit:subprocess.run(['systemctl','--user','is-active','--quiet',unit],stdout=subprocess.DEVNULL).returncode==0
busy=any(active(u) for u in ('aerorozum-robotics-label-bucket.service','aerorozum-quarantine-labels.service','aerorozum-additional-labels.service'))
lease=json.loads((r/'pipeline/leases/label-recovery.json').read_text());expired=datetime.now(timezone.utc)>=datetime.fromisoformat(lease['expires_at'].replace('Z','+00:00'))
state={'at':utcnow(),'normal_raw_objects':len(normal),'all_raw_objects':len(raw),'original_snapshot_objects':len(original),'additional_source_versions':len(selected),'pending_source_versions':len(pending),'busy':busy,'deadline_expired':expired,'no_inference_in_discovery':True,'pending':[{'object':s['name'],'generation':s['generation'],'size':s['size']} for s in pending]}
# Initial execution is exclusively OnSuccess. Later arrivals may restart only
# the completed additional stage; never race the main service's restart policy.
if ledger.exists() and pending and not busy and not expired and not (r/'labelling/airborne-only-drain.json').exists():
 subprocess.run(['systemctl','--user','start','--no-block','aerorozum-additional-labels.service'],check=True);state['triggered_additional']=True
state['airborne_only_handoff']=(r/'labelling/airborne-only-drain.json').exists()
save(out/'watch.json',state);print(json.dumps({k:v for k,v in state.items() if k!='pending'}))
