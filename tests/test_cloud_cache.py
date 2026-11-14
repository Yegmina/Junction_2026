import io,json,urllib.error
import pytest
from aerorozum.cloud_cache import GCSRecordStore

def test_cache_read_401_refreshes_once_and_returns_saved_response(monkeypatch):
 tokens=iter(['expired-test-token','fresh-test-token']);issued=[];requests=[]
 def token(*args,**kwargs):issued.append(1);return next(tokens)
 monkeypatch.setattr('aerorozum.cloud_cache.subprocess.check_output',token)
 def request(req,**kwargs):
  requests.append(req.headers['Authorization'])
  if len(requests)==1:raise urllib.error.HTTPError(req.full_url,401,'expired',None,None)
  return io.BytesIO(b'{"saved_paid_response":true}')
 monkeypatch.setattr('aerorozum.cloud_cache.urllib.request.urlopen',request)
 assert GCSRecordStore('bucket-test').get('a'*64)=={'saved_paid_response':True}
 assert requests==['Bearer expired-test-token','Bearer fresh-test-token'] and len(issued)==2

def test_cache_401_retry_is_bounded(monkeypatch):
 count=[];monkeypatch.setattr('aerorozum.cloud_cache.subprocess.check_output',lambda *a,**kw:'test-token')
 def request(req,**kwargs):count.append(1);raise urllib.error.HTTPError(req.full_url,401,'expired',None,None)
 monkeypatch.setattr('aerorozum.cloud_cache.urllib.request.urlopen',request)
 with pytest.raises(RuntimeError,match='paid inference was not issued'):GCSRecordStore('bucket-test').get('a'*64)
 assert len(count)==2

def test_upload_401_refresh_preserves_immutable_precondition(monkeypatch):
 tokens=iter(['expired-test-token','fresh-test-token']);requests=[]
 monkeypatch.setattr('aerorozum.cloud_cache.subprocess.check_output',lambda *a,**kw:next(tokens))
 def request(req,**kwargs):
  requests.append(req)
  if len(requests)==1:raise urllib.error.HTTPError(req.full_url,401,'expired',None,None)
  return io.BytesIO(b'{}')
 monkeypatch.setattr('aerorozum.cloud_cache.urllib.request.urlopen',request)
 assert GCSRecordStore('bucket-test').put('a'*64,{'original':'paid'})=='gs://bucket-test/robotics-er2-labels/5fps-v1/results/'+('a'*64)+'.json'
 assert len(requests)==2 and all('ifGenerationMatch=0' in r.full_url for r in requests)
 assert requests[0].data==requests[1].data

def test_stale_401_does_not_invalidate_concurrent_new_token(monkeypatch):
 store=GCSRecordStore('bucket-test');store._token='new-token';store._expiry=100
 store._invalidate({'Authorization':'Bearer old-token'})
 assert store._token=='new-token' and store._expiry==100
