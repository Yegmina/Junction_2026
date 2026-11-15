"""Executed at a CPython safe point; see drain_legacy_label_worker.py."""
import gc as _aero_gc,asyncio as _aero_asyncio,os as _aero_os,time as _aero_time,json as _aero_json
from pathlib import Path as _AeroPath
from aerorozum.providers import GeminiProvider as _AeroProvider
_aero_gate_time=_aero_time.time()
_aero_providers=[v for v in _aero_gc.get_objects() if isinstance(v,_AeroProvider)]
_aero_names=set(_AeroProvider.generate.__code__.co_names)|{x for x in _AeroProvider.generate.__code__.co_consts if isinstance(x,str)}
if not {'limit','issued'}.issubset(_aero_names):raise RuntimeError('Loaded provider lacks verified request-limit guard; no changes installed')
async def _aero_stop_unfiltered():
    raise RuntimeError('Recovery airborne-only handoff requested; no new paid request issued')
_aero_issued=[p.issued for p in _aero_providers]
for _aero_p in _aero_providers:
    _aero_p.before_queue=_aero_stop_unfiltered
    _aero_p.before_request=_aero_stop_unfiltered
    _aero_p.limit=_aero_p.issued
_aero_receipt=_AeroPath(RECEIPT_PATH)
_aero_receipt.parent.mkdir(parents=True,exist_ok=True)
def _aero_count_generators():
    count=0
    for task in _aero_asyncio.all_tasks():
        coro=task.get_coro()
        for depth in range(100):
            if coro is None:break
            code=getattr(coro,'cr_code',None)
            if code is not None and code.co_qualname=='GeminiProvider.generate':count+=1;break
            coro=getattr(coro,'cr_await',None)
    return count
async def _aero_monitor_drain():
    while True:
        n=_aero_count_generators()
        result={'pid':_aero_os.getpid(),'installed_at_epoch_s':_aero_gate_time,'checked_at_epoch_s':_aero_time.time(),'providers_gated':len(_aero_providers),'issued_at_gate':_aero_issued,'issued_now':[p.issued for p in _aero_providers],'loaded_gate_checks':{'before_queue':'before_queue' in _aero_names,'before_request':'before_request' in _aero_names,'limit_and_issued':True},'active_generation_coroutines':n,'status':'drained' if n==0 else 'draining','requests_cancelled':False}
        temp=_aero_receipt.with_suffix('.tmp');temp.write_text(_aero_json.dumps(result));temp.replace(_aero_receipt)
        if n==0:return
        await _aero_asyncio.sleep(1)
_aero_asyncio.get_running_loop().create_task(_aero_monitor_drain())
