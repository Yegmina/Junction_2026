import asyncio,hashlib,json
from pathlib import Path
import pytest
from PIL import Image
from aerorozum.common import save
from aerorozum.labelling import MODEL
from aerorozum.real_ingester import RealIngester
from aerorozum.label_recovery import paid_permission,recover_pass

def native(root,identity='one',split='train',confidence=None):
 labels=root/'labelling/robotics-er2-5fps/labels';labels.mkdir(parents=True,exist_ok=True)
 path=root/f'{identity}.png';im=Image.new('RGB',(128,128));im.putdata([(x*2,x*2,x*2) for y in range(128) for x in range(128)]);im.save(path)
 box={'x_min':100,'y_min':100,'x_max':200,'y_max':200};detections=[] if confidence is None else [{'bbox':box,'confidence':confidence}]
 row={'id':identity,'image':identity+'.png','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'split':split,'source':{'session_id':'session','video_sha256':'a'*64,'generation':'1'},'model':MODEL,'provenance':'user_accepted_model','user_threshold':.6,'human_reviewed':False,'model_output':{'detections':detections},'detections':detections,'boxes':[] if confidence is None or confidence<.6 else [{'bbox':box,'confidence':confidence}],'uncertain_boxes':[] if confidence is None or confidence>=.6 else [{'bbox':box,'confidence':confidence}],'zero_candidates':confidence is None,'cloud_sync':{'status':'uploaded'}}
 save(labels/'results'/f'{identity}.json',row);save(labels.parent/'session-splits.json',{'sessions':{'session':split}});return row,labels

def test_incremental_native_index_hashes_zero_policy_and_frozen_splits(tmp_path):
 row,labels=native(tmp_path);ingester=RealIngester(tmp_path);d=ingester.scan()
 assert d['session_splits']=={'session':'train'} and d['zone_splits']['zone8']=='test'
 assert d['counts']['native_train_zero_frames']==1 and d['counts']['background_bank']==1
 assert d['images'][0]['image']=='one.png' and d['images'][0]['training_eligible']
 bank=json.loads((ingester.output/'backgrounds.json').read_text());assert not bank['crops'][0]['target_free_verified']
 (tmp_path/'one.png').write_bytes(b'modified')
 again=ingester.scan();assert again['counts']['images']==0 and again['counts']['excluded_records']==1

def test_uncertain_and_heldout_do_not_enter_training_texture_pool(tmp_path):
 native(tmp_path,confidence=.2);d=RealIngester(tmp_path).scan();assert d['counts']['images']==0
 root=tmp_path/'heldout';root.mkdir();native(root,split='test');d=RealIngester(root).scan();assert len(d['images'])==1 and not d['images'][0]['training_eligible'];assert d['counts']['background_bank']==0

def test_empty_boxes_without_explicit_empty_detections_are_not_negative(tmp_path):
 row,labels=native(tmp_path);row['zero_candidates']=False;save(labels/'results/one.json',row)
 d=RealIngester(tmp_path).scan();assert not d['images'] and d['counts']['excluded_records']==1

def test_cache_only_recovery_cannot_pay_while_main_runs(tmp_path):
 row,labels=native(tmp_path);(labels/'results/one.json').unlink();manifest=tmp_path/'recovery.json';save(manifest,{'images':[row]})
 def forbidden(*args):raise AssertionError('provider should not be constructed for unresolved unpaid frames without allowance')
 state=asyncio.run(recover_pass(manifest,labels,tmp_path/'lease.json',main_running=True,provider_factory=forbidden));assert state['pending']==1 and not state['paid_permission']['allowed']
 assert paid_permission(tmp_path/'lease.json',False)['allowed']
 save(tmp_path/'lease.json',{'status':'granted','requests_per_minute':60,'expires_at':'2020-01-01T00:00:00Z'});assert not paid_permission(tmp_path/'lease.json',True)['allowed']

def test_recovery_rerun_reuses_new_paid_result_without_second_generation(tmp_path):
 row,labels=native(tmp_path);(labels/'results/one.json').unlink();manifest=tmp_path/'recovery.json';save(manifest,{'images':[row]});lease=tmp_path/'lease.json';save(lease,{'status':'granted','requests_per_minute':60})
 calls=[]
 class Provider:
  async def generate(self,*args,**kwargs):calls.append(1);return {'detections':[]}
 state=asyncio.run(recover_pass(manifest,labels,lease,provider_factory=lambda *a:Provider()))
 assert state['active_batch']==1 and len(calls)==1
 def forbidden(*_):raise AssertionError('completed paid response must not be regenerated')
 again=asyncio.run(recover_pass(manifest,labels,lease,provider_factory=forbidden))
 assert again['completed']==1 and len(calls)==1

def test_explicit_pause_stays_paused_after_main_finishes(tmp_path):
 lease=tmp_path/'lease.json';save(lease,{'status':'paused','requests_per_minute':30})
 assert not paid_permission(lease,False)['allowed']
 save(lease,{'status':'granted','requests_per_minute':30})
 assert paid_permission(lease,False)['requests_per_minute']==30

def test_recent_attempts_restore_rate_window_without_double_counting(tmp_path):
 import time
 from datetime import datetime,timezone
 from aerorozum.label_recovery import seed_recovery_window,provider_key
 from aerorozum.providers import WindowLimiter
 row,labels=native(tmp_path)
 class P:cfg={'thinking_budget':0,'api_version':'v1beta','reserve_output_tokens':2048};limiter=WindowLimiter(30,500000)
 key=provider_key(row,P.cfg);record={'created_at':datetime.now(timezone.utc).isoformat()}
 save(labels/'provider/raw'/(key+'-0.json'),record);save(labels/'provider/errors'/(key+'-0.json'),record)
 assert seed_recovery_window(P,labels,{'images':[row]})==1
 assert len(P.limiter.calls)==1 and 0<=time.monotonic()-P.limiter.calls[0][0]<60

def test_airborne_policy_excludes_ground_unknown_and_unclassified_native(tmp_path):
 row,labels=native(tmp_path);save(tmp_path/'labelling/airborne-only-drain.json',{'requested':True});ingester=RealIngester(tmp_path)
 d=ingester.scan();assert d['counts']['native_unclassified']==1 and d['counts']['background_bank']==0 and not d['images'][0]['training_eligible']
 for status,expected in [('ground',False),('unknown',False),('eligible',True)]:
  save(tmp_path/'pipeline/inputs/airborne-eligibility.json',{'policy':{'version':'test'},'counts':{'airborne_eligible':int(expected)},'frames':[{'id':'one','sha256':row['sha256'],'source':row['source'],'frame_index':None,'eligibility':{'status':status,'airborne_eligible':expected}}]})
  d=ingester.scan();assert d['images'][0]['training_eligible'] is expected
  assert d['counts']['background_bank']==int(expected)

def test_atomic_metadata_replacement_invalidates_same_size_and_mtime_cache(tmp_path):
 import os
 path=tmp_path/'eligibility.json';save(path,{'state':'ground'});ingester=RealIngester(tmp_path)
 assert ingester.read(path)['state']=='ground'
 before=path.stat();replacement=tmp_path/'replacement.json';save(replacement,{'state':'flying'})
 assert replacement.stat().st_size==before.st_size
 os.utime(replacement,ns=(before.st_atime_ns,before.st_mtime_ns));os.replace(replacement,path)
 assert ingester.read(path)['state']=='flying'
