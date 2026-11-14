"""Retry canonical paid-label/log uploads. There is no inference client here."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
from .cloud_cache import GCSRecordStore
from .common import digest,redacted_error,save,utcnow

def frame_identity(row):
 return digest([row.get('cache_key'),row.get('id'),row.get('sha256'),row.get('source',{}),row.get('frame_index')])

def sync_paid_record(path,store):
 path=Path(path);snapshot=path.read_bytes();row=json.loads(snapshot);key=row.get('cache_key') or path.stem
 if row.get('model')!='gemini-robotics-er-2-preview' or not row.get('sha256'):raise ValueError('Unrecognized paid label identity')
 raw=row.get('raw_response')
 if raw is None:
  paid=path.parent.parent/'paid-responses'/(key+'.json')
  if paid.exists():raw=json.loads(paid.read_text()).get('raw_response')
 if not isinstance(raw,dict):raise ValueError('Original provider log unavailable; no replacement inference will be issued')
 if raw.get('model')!=row['model'] or raw.get('image_hashes')!=[row['sha256']]:raise ValueError('Provider log identity differs from paid frame')
 uris=store.write_frame(key,row,raw_log=raw)
 frame_key=digest([key,row.get('id'),row.get('source',{}),row.get('frame_index')])
 uris['frame_label']=store.put(frame_key,{**row,'raw_log_uri':uris.get('raw_log'),'inference_cache_key':key},kind='labels')
 for identity,kind in ((key,'results'),(key,'labels'),(frame_key,'labels')):
  remote=store.get(identity,kind)
  if not remote or remote.get('model')!=row['model'] or remote.get('sha256')!=row['sha256']:raise ValueError('Canonical paid record failed remote readback')
 remote_raw=store.get(key,'logs')
 if not remote_raw or remote_raw.get('model')!=row['model'] or remote_raw.get('image_hashes')!=[row['sha256']]:raise ValueError('Original paid response log failed remote readback')
 receipt={'status':'uploaded','at':utcnow(),'cache_key':key,'frame_id':row.get('id'),'sha256':row['sha256'],'model':row['model'],'frame_identity':frame_identity(row),'source_json_sha256':hashlib.sha256(snapshot).hexdigest(),'uris':uris,'remote_kinds_verified':['results','labels','logs','frame_label'],'raw_text_available':'raw_text' in remote_raw,'paid_inference_issued':False,'source_record_modified':False}
 save(path.parent.parent/'upload-receipts'/(key+'.json'),receipt);return receipt

def retry_uploads(labels_root,store_factory=None,workers=8):
 root=Path(labels_root);pending=[];verified=[];failures=[];stores={}
 for path in (root/'results').glob('*.json'):
  row=json.loads(path.read_text())
  status=row.get('cloud_sync',{}).get('status','not_requested')
  if status!='upload_pending' and not(status=='not_requested' and row.get('id')):continue
  receipt=root/'upload-receipts'/((row.get('cache_key') or path.stem)+'.json')
  if receipt.exists():
   old=json.loads(receipt.read_text())
   if old.get('status')=='uploaded' and old.get('frame_identity')==frame_identity(row):verified.append(old);continue
  pending.append((path,row))
  bucket=row.get('source',{}).get('bucket')
  if not bucket:raise ValueError('Explicit source bucket required')
  if bucket not in stores:stores[bucket]=store_factory(bucket) if store_factory else GCSRecordStore(bucket)
 def upload(item):return sync_paid_record(item[0],stores[item[1]['source']['bucket']])
 previously_verified=len(verified)
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(upload,item):item for item in pending}
  for future in as_completed(futures):
   try:verified.append(future.result())
   except Exception as ex:failures.append({'cache_file':futures[future][0].name,'error':redacted_error(ex)})
 state={'status':'synced' if not failures else 'upload_pending','at':utcnow(),'pending_source_records':len(pending)+previously_verified,'newly_attempted':len(pending),'canonical_verified':len(verified),'remaining_failed':len(failures),'failures':failures,'paid_inference_issued':False,'source_records_modified':False}
 save(root/'canonical-upload-sync.json',state);return state
