import asyncio
from types import SimpleNamespace
import pytest
from aerorozum.providers import WindowLimiter,ProviderDraining
from test_provider_gates import provider

class Clock:
 def __init__(self):self.now=0.
 def __call__(self):return self.now
 async def sleep(self,seconds):self.now+=seconds;await asyncio.sleep(0)

def response(text='{"detections":[]}',tokens=1254):
 return SimpleNamespace(text=text,usage_metadata=SimpleNamespace(model_dump=lambda **kw:{'prompt_token_count':tokens}),candidates=[SimpleNamespace(finish_reason='STOP')])

def test_measured_usage_refund_reaches120rpm_under500k():
 async def run():
  clock=Clock();limiter=WindowLimiter(120,500000,clock=clock,sleeper=clock.sleep)
  for _ in range(120):
   t=await limiter.acquire(4367);await limiter.dispatch(t);await limiter.reconcile(t,1254)
   assert (await limiter.snapshot())['input_tokens_accounted']<=500000
  assert clock.now==0 and (await limiter.snapshot())['requests_in_window']==120
  t=await limiter.acquire(4367);assert clock.now>=60 and t[0]>=60
 asyncio.run(run())

def test_slow_inflight_reservation_survives_window_and_refund_unblocks():
 async def run():
  clock=Clock();limiter=WindowLimiter(100,100,clock=clock,sleeper=clock.sleep)
  t=await limiter.acquire(60);await limiter.dispatch(t);clock.now=120
  pending=asyncio.create_task(limiter.acquire(50));await asyncio.sleep(0)
  assert not pending.done() and (await limiter.snapshot())['inflight_reserved_tokens']==60
  await limiter.reconcile(t,20);second=await pending
  assert (await limiter.snapshot())['input_tokens_accounted']==70
  await limiter.cancel(second)
 asyncio.run(run())

def test_unknown_usage_keeps_hold_and_actual_excess_is_not_clipped():
 async def run():
  limiter=WindowLimiter(120,500000)
  for actual in (None,0,'1254'):
   t=await limiter.acquire(4367);await limiter.dispatch(t);await limiter.reconcile(t,actual);assert t[1]==4367
  t=await limiter.acquire(4367);await limiter.dispatch(t);await limiter.reconcile(t,5000);assert t[1]==5000
 asyncio.run(run())

def test_paid_invalid_json_refunds_tokens_and_retains_retry_rpm(tmp_path,monkeypatch):
 calls=[]
 async def generate(**kw):calls.append(1);return response('broken' if len(calls)==1 else '{"detections":[]}')
 async def noop(*args,**kw):pass
 monkeypatch.setattr('aerorozum.providers.asyncio.sleep',noop)
 p=provider(tmp_path/'provider',generate);image=tmp_path/'frame.jpg';image.write_bytes(b'image')
 assert asyncio.run(p.generate('role','prompt',[image]))=={'detections':[]}
 assert p.issued==2 and len(p.limiter.calls)==2 and sum(t[1] for t in p.limiter.calls)==2508
 assert len(list((p.root/'raw').glob('*.json')))==2

def test_drain_completes_paid_request_blocks_queue_and_preserves_cache(tmp_path):
 async def run():
  started=asyncio.Event();release=asyncio.Event();calls=[]
  async def generate(**kw):calls.append(1);started.set();await release.wait();return response(tokens=1)
  p=provider(tmp_path,generate);p.semaphore=asyncio.Semaphore(1)
  paid=asyncio.create_task(p.generate('first','prompt'));await started.wait()
  queued=asyncio.create_task(p.generate('second','prompt'));await asyncio.sleep(0)
  drainer=asyncio.create_task(p.drain());await asyncio.sleep(0)
  assert not drainer.done() and p.inflight_requests==1
  release.set();assert await paid=={'detections':[]};await drainer
  with pytest.raises(ProviderDraining):await queued
  assert await p.generate('first','prompt')=={'detections':[]}
  assert len(calls)==1 and p.issued==1 and p.inflight_requests==0
 asyncio.run(run())

@pytest.mark.parametrize('control',['path','callback'])
def test_drain_controls_stop_before_transport(tmp_path,control):
 async def run():
  calls=[]
  async def generate(**kw):calls.append(1);return response()
  p=provider(tmp_path/'provider',generate)
  if control=='path':
   path=tmp_path/'drain';path.touch();p.cfg['drain_path']=str(path)
  else:
   async def stop():return True
   p.should_drain=stop
  with pytest.raises(ProviderDraining):await p.generate('role','prompt')
  assert not calls and not p.issued
 asyncio.run(run())

def test_drain_interrupts_quota_wait(tmp_path):
 async def run():
  async def generate(**kw):raise AssertionError('Must not issue')
  p=provider(tmp_path,generate);p.limiter=WindowLimiter(1,500000)
  t=await p.limiter.acquire(10);await p.limiter.dispatch(t);await p.limiter.reconcile(t,10)
  queued=asyncio.create_task(p.generate('role','prompt'));await asyncio.sleep(0);p.request_drain()
  with pytest.raises(ProviderDraining):await asyncio.wait_for(queued,1)
  assert p.issued==0
 asyncio.run(run())

def test_drain_timeout_leaves_paid_transport_alive(tmp_path):
 async def run():
  started=asyncio.Event();release=asyncio.Event()
  async def generate(**kw):started.set();await release.wait();return response(tokens=1)
  p=provider(tmp_path,generate);paid=asyncio.create_task(p.generate('role','prompt'));await started.wait()
  with pytest.raises(asyncio.TimeoutError):await p.drain(timeout=.01)
  assert not paid.done();release.set();await paid;await p.drain()
 asyncio.run(run())
