import asyncio
from types import SimpleNamespace
import pytest
from aerorozum.providers import GeminiProvider,WindowLimiter


def provider(tmp_path,generate):
 p=GeminiProvider.__new__(GeminiProvider);p.cfg={'bulk_model':'gemini-robotics-er-2-preview','max_retries':3,'reserve_output_tokens':2048};p.root=tmp_path;p.limiter=WindowLimiter(1000,1000000);p.semaphore=asyncio.Semaphore(2);p.limit=100;p.issued=0;p.requests_lock=asyncio.Lock();p.client=SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)));return p

def test_expired_lease_gate_blocks_before_any_paid_transport(tmp_path):
 calls=[]
 async def generate(**kwargs):calls.append(1);raise AssertionError('paid transport reached')
 p=provider(tmp_path,generate)
 async def expired():raise RuntimeError('recovery lease expired')
 p.before_request=expired
 with pytest.raises(RuntimeError,match='lease expired'):asyncio.run(p.generate('role','prompt'))
 assert calls==[] and p.issued==0

def test_quota_callback_stops_retry_before_second_paid_transport(tmp_path,monkeypatch):
 calls=[];rejected=[]
 class Quota(Exception):code=429
 async def generate(**kwargs):calls.append(1);raise Quota('429 RESOURCE_EXHAUSTED')
 p=provider(tmp_path,generate)
 async def reject(error,code):rejected.append(code)
 async def gate():
  if rejected:raise RuntimeError('extra allowance paused')
 async def noop(*args,**kwargs):pass
 monkeypatch.setattr('aerorozum.providers.asyncio.sleep',noop);p.on_request_error=reject;p.before_queue=gate
 with pytest.raises(RuntimeError,match='allowance paused'):asyncio.run(p.generate('role','prompt'))
 assert calls==[1] and rejected==[429] and p.issued==1
 assert len(list((tmp_path/'errors').glob('*.json')))==1
