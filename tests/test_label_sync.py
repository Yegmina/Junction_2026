import json
from pathlib import Path
from aerorozum.common import save
from aerorozum.label_sync import sync_paid_record,frame_identity

class Store:
 def __init__(self):self.records={}
 def put(self,key,value,kind='results'):self.records.setdefault((kind,key),value);return 'gs://test/'+kind+'/'+key
 def write_frame(self,key,row,raw_log=None):return {'result':self.put(key,row),'labels':self.put(key,row,'labels'),'raw_log':self.put(key,raw_log,'logs')}
 def get(self,key,kind='results'):return self.records.get((kind,key))

def record(tmp_path):
 key='a'*64;row={'id':'actual-frame','cache_key':key,'model':'gemini-robotics-er-2-preview','sha256':'b'*64,'frame_index':12,'source':{'bucket':'test-bucket','generation':'7'},'cloud_sync':{'status':'upload_pending'},'raw_response':{'model':'gemini-robotics-er-2-preview','image_hashes':['b'*64],'raw_text':'{"detections":[]}'}};path=tmp_path/'results'/(key+'.json');save(path,row);return path,row

def test_canonical_label_and_original_log_retry_never_changes_source(tmp_path):
 path,row=record(tmp_path);original=path.read_bytes();store=Store();receipt=sync_paid_record(path,store)
 assert path.read_bytes()==original and receipt['paid_inference_issued'] is False
 assert receipt['status']=='uploaded' and receipt['raw_text_available']
 assert store.get(row['cache_key'],'logs')['raw_text']=='{"detections":[]}'
 assert len(store.records)==4

def test_receipt_cannot_claim_new_frame_after_main_changes_cached_binding(tmp_path):
 path,row=record(tmp_path)
 class Racing(Store):
  def write_frame(self,*args,**kwargs):
   value=super().write_frame(*args,**kwargs);save(path,{**row,'id':'new-binding'});return value
 receipt=sync_paid_record(path,Racing());current=json.loads(path.read_text())
 assert current['id']=='new-binding' and receipt['frame_identity']!=frame_identity(current)
