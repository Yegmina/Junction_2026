#!/usr/bin/env python3
"""Extract/synchronize every raw source, admitting only corroborated airborne frames."""
import base64,hashlib,json,os,urllib.request,urllib.parse,urllib.error
from collections import Counter
from pathlib import Path
from aerorozum.common import data_root,save,digest,utcnow
from aerorozum.cloud_cache import GCSRecordStore
from aerorozum.labelling import extract_video_samples,MODEL,classify_response
from aerorozum.flight_filter import read_telemetry,filter_rows
r=data_root();out=r/'labelling/flight-filter';out.mkdir(parents=True,exist_ok=True);output=r/'pipeline/inputs';sources=json.loads((r/'source-videos/labelling-sources.json').read_text())['videos'];splits=json.loads((r/'labelling/robotics-er2-5fps/session-splits.json').read_text())['sessions'];auth=GCSRecordStore('aerorozum-jetson-testdata-smart-bird');metadata_credentials=None
if os.environ.get('AEROROZUM_METADATA_AUTH')=='1':
 from google.auth.compute_engine import Credentials
 from google.auth.transport.requests import Request
 metadata_credentials=Credentials();metadata_transport=Request()
def get(url):
 for attempt in range(2):
  if metadata_credentials is not None:
   if not metadata_credentials.valid:metadata_credentials.refresh(metadata_transport)
   headers={'Authorization':'Bearer '+metadata_credentials.token}
  else:headers=auth._headers()
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:return response.read()
  except urllib.error.HTTPError as ex:
   if ex.code==401 and attempt==0:
    if metadata_credentials is not None:metadata_credentials.refresh(metadata_transport)
    else:auth._invalidate(headers)
    continue
   raise


def pinned_object(bucket,name,generation=None):
 url='https://storage.googleapis.com/storage/v1/b/'+bucket+'/o/'+urllib.parse.quote(name,safe='');meta=json.loads(get(url+('?generation='+str(generation) if generation else '')))
 if generation and str(meta['generation'])!=str(generation):raise ValueError('Pinned GCS generation mismatch')
 folder=out/'native'/digest([bucket,name,str(meta['generation'])]);folder.mkdir(parents=True,exist_ok=True);path=folder/Path(name).name
 if not path.exists():
  raw=get(url+'?alt=media&generation='+str(meta['generation']))
  if len(raw)!=int(meta['size']) or (meta.get('md5Hash') and base64.b64encode(hashlib.md5(raw).digest()).decode()!=meta['md5Hash']):raise ValueError('Native/receipt object integrity mismatch')
  temporary=path.with_suffix('.partial');temporary.write_bytes(raw);temporary.replace(path)
 save(folder/'source.json',{'bucket':bucket,'object':name,'generation':str(meta['generation']),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size});return path

def telemetry(source):
 bucket=source['bucket'];name=source['object'].split('/recordings/')[0]+'/recordings/'+source['session_id']+'/telemetry.jsonl';url='https://storage.googleapis.com/storage/v1/b/'+bucket+'/o/'+urllib.parse.quote(name,safe='');meta=json.loads(get(url));folder=out/'telemetry'/digest([bucket,name,meta['generation']]);folder.mkdir(parents=True,exist_ok=True);path=folder/'telemetry.jsonl'
 if not path.exists():
  local=Path('/home/yehormsi/Downloads/Jetson-today-onboard')/source['session_id']/'telemetry.jsonl';raw=local.read_bytes() if local.exists() else None
  if raw is None or len(raw)!=int(meta['size']) or base64.b64encode(hashlib.md5(raw).digest()).decode()!=meta.get('md5Hash'):raw=get(url+'?alt=media&generation='+str(meta['generation']))
  if len(raw)!=int(meta['size']):raise ValueError('Pinned telemetry size mismatch')
  if meta.get('md5Hash') and base64.b64encode(hashlib.md5(raw).digest()).decode()!=meta['md5Hash']:raise ValueError('Pinned telemetry MD5 mismatch')
  temporary=path.with_suffix('.partial');temporary.write_bytes(raw);temporary.replace(path)
 identity={'bucket':bucket,'object':name,'generation':str(meta['generation']),'path':str(path.relative_to(r))};parsed=read_telemetry(path,identity);save(folder/'source.json',{**identity,'sha256':parsed['source']['sha256'],'size':path.stat().st_size});return parsed

# Prefer the latest validated sampling snapshot for a source version. Complete
# clock repairs outrank a video-only preview; no timestamp is invented.
existing={};rank={}
for path in (r/'labelling').glob('*/frames/*/sampling.json'):
 try:doc=json.loads(path.read_text());rows=doc['images']
 except (ValueError,KeyError,OSError):continue
 if not rows:continue
 src=rows[0].get('source',{});key=(src.get('object'),src.get('generation'));quality=2 if rows[0].get('receipt_timestamp_ns') else 1 if rows[0].get('time_basis','').startswith('jetson') else 0;score=(quality,path.stat().st_mtime_ns)
 if key not in rank or score>rank[key]:existing[key]=rows;rank[key]=score
prior_paid={}
for group in ('robotics-er2-5fps','robotics-er2-airborne-priority','robotics-er2-airborne'):
 for directory in ('paid-responses','results'):
  for paid_path in (r/'labelling'/group/'labels'/directory).glob('*.json'):
   paid=json.loads(paid_path.read_text());src=paid.get('source',{})
   if paid.get('model')!=MODEL or not paid.get('sha256'):continue
   try:classify_response(paid['model_output']);status='prior_paid_result_preserved'
   except (ValueError,KeyError):status='invalid_paid_output_retained'
   prior_paid[(paid['sha256'],src.get('generation'))]={'status':status,'cache_key':paid.get('cache_key',paid_path.stem),'paid_record':str(paid_path.relative_to(r))}
all_rows=[];by_session={};reports=[];sources.sort(key=lambda s:(s['session_id'],-s['segment']),reverse=True)
for index,src in enumerate(sources,1):
 sid=src['session_id'];key=(src['object'],str(src['generation']));error=None
 if not Path(src['path']).is_file():src['path']=str(pinned_object(src['bucket'],src['object'],src['generation']))
 if not src.get('timestamps_path') or not Path(src['timestamps_path']).is_file():
  try:
   receipt=src.get('receipt_timestamp_recovery',{}).get('receipt',{});name=receipt.get('object') or src['object'].split('/recordings/')[0]+'/recordings/'+sid+'/thermal-timestamps.jsonl';raw_receipt=pinned_object(src['bucket'],name,receipt.get('generation'))
   from aerorozum.quarantine_recovery import receipt_records
   _,derived=receipt_records(raw_receipt,raw_receipt.parent/'cleaned.jsonl');src['timestamps_path']=str(raw_receipt.parent/'cleaned.jsonl')
  except Exception as ex:src.pop('timestamps_path',None);error='Receipt unavailable: '+type(ex).__name__
 if sid not in by_session:
  try:by_session[sid]=telemetry(src)
  except Exception as ex:by_session[sid]=None;error=type(ex).__name__+': '+str(ex)[:180]
 rows=existing.get(key)
 if rows is None:
  source={k:src[k] for k in ('bucket','object','generation','session_id','segment')};folder=out/'frames'/digest(source)[:20]
  try:rows=extract_video_samples(src['path'],folder,session_id=sid,source=source,timestamps_path=src.get('timestamps_path'),segment=src['segment'],fps=5,split=splits[sid])
  except Exception as ex:
   error=type(ex).__name__+': '+str(ex)[:180]
   try:rows=extract_video_samples(src['path'],folder,session_id=sid,source=source,segment=src['segment'],fps=5,split=splits[sid])
   except Exception as fatal:reports.append({**source,'status':'quarantined_native_decode','error':str(fatal)[:250]});rows=[]
 # Preserve precise receipt integers where a one-to-one source receipt file is
 # present. Otherwise the existing extractor's validated receipt seconds remain.
 if src.get('timestamps_path'):
  try:
   records=[json.loads(line) for line in Path(src['timestamps_path']).read_text().splitlines() if line.strip()];local=[v for v in records if int(v.get('segment',0))==src['segment']]
   for row in rows:
    if row.get('time_basis','').startswith('jetson') and row['frame_index']<len(local):row['receipt_timestamp_ns']=int(local[row['frame_index']]['received_at_ns'])
  except (OSError,ValueError,KeyError):pass
 for row in rows:
  path=Path(row['image']);row['image']=str(path.relative_to(r)) if path.is_absolute() and path.is_relative_to(r) else str(path)
 all_rows.extend(rows);reports.append({'object':src['object'],'generation':src['generation'],'session_id':sid,'sampled_frames':len(rows),'telemetry_available':by_session.get(sid) is not None,'error':error})
 # Publish an immutable transferable snapshot after each session's last segment.
 if index==len(sources) or sources[index]['session_id']!=sid:
  audit,manifest=filter_rows(all_rows,by_session,session_splits=splits,output_dir=output,source_reports=reports,prior_paid_identities=prior_paid);snapshot=output/'airborne-snapshots'/manifest['eligibility_manifest_sha256'][:16];snapshot.mkdir(parents=True,exist_ok=True);save(snapshot/'airborne-eligibility.json',audit);manifest['eligibility_manifest']=str((snapshot/'airborne-eligibility.json').relative_to(r));manifest['coverage_status']='complete' if index==len(sources) else 'partial';manifest['source_versions_processed']=index;manifest['source_versions_total']=len(sources);save(snapshot/'airborne-eligible.json',manifest);save(output/'airborne-eligible.json',manifest);state={'status':manifest['coverage_status'],'at':utcnow(),'sources_processed':index,'sources_total':len(sources),'counts':manifest['counts'],'manifest':'pipeline/inputs/airborne-eligible.json','snapshot':str((snapshot/'airborne-eligible.json').relative_to(r)),'paid_inference_issued':False};save(out/'progress.json',state);print(json.dumps(state),flush=True)
