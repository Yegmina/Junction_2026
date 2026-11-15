"""Hackathon-only Gemini provider. No ADC, Vertex, or startup fallback."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import time

from .common import config, data_root, digest, redacted_error, save, secret, utcnow

DETECTION_PROMPT = '''Inspect this thermal image for compact ground-level thermal signatures that could warrant mine-survey review. Heat or circularity alone is not evidence of a mine. Distinguish people, animals, vegetation, rocks, broad heated ground and artifacts. Return tight boxes only for visible signatures, no more than ten. Never confirm mines or invent geographic coordinates. Return JSON {"detections":[{"label":"provisional thermal candidate|person|animal|thermal clutter","confidence":0.0,"bbox":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000},"evidence":"brief visible evidence","uncertainty":"alternative explanation"}],"explanation":"concise visible evidence"}. Boxes use named normalized integer coordinates 0..1000. An empty detections list is valid. Do not provide private chain-of-thought.'''
REVIEW_PROMPT = '''Review the ORIGINAL thermal image and these proposals. Retain only evidence visibly supported in the image. Do not assume the proposal is correct, and do not interpret all bright spots as mines. Return JSON {"reviews":[{"index":0,"decision":"retain_candidate|reject|uncertain","reason":"brief visible evidence"}],"explanation":"data limitations"}. Include every proposal index; never confirm mines or invent GPS. Proposals: '''


class ProviderDraining(RuntimeError):
    """Unstarted paid work stays resumable while accepted requests finish."""


def request_identity(cfg,*,role,prompt,images,model,max_output_tokens):
    """Identity shared by strict JSON calls and retained-completion evaluation."""
    identity=dict(role=role,prompt=prompt,images=images,model=model,
        temperature=float(cfg.get('temperature',.1)),max_output_tokens=max_output_tokens,
        api_version=cfg.get('api_version','v1beta'),thinking_budget=cfg.get('thinking_budget'))
    if cfg.get('thinking_level') is not None:
        identity['thinking_level']=str(cfg['thinking_level']).lower()
    return digest(identity)


def generation_settings(cfg,limit):
    generation=dict(temperature=float(cfg.get('temperature',.1)),response_mime_type='application/json',max_output_tokens=limit)
    if cfg.get('thinking_level') is not None:
        if cfg.get('thinking_budget') is not None:raise ValueError('Choose thinking level or token budget, not both')
        level=str(cfg['thinking_level']).lower()
        if level not in ('minimal','low','medium','high','xhigh'):raise ValueError('Unknown reasoning level')
        generation['thinking_config']={'thinking_level':level.upper()}
    elif 'thinking_budget' in cfg:
        generation['thinking_config']={'thinking_budget':cfg['thinking_budget']}
    return generation


@dataclass(eq=False)
class TokenReservation:
    created_at: float
    tokens: int
    dispatched_at: float | None = None
    settled_at: float | None = None

    def __getitem__(self,index):
        return (self.created_at if self.dispatched_at is None else self.dispatched_at,self.tokens)[index]


class WindowLimiter:
    def __init__(self, rpm=60, tpm=200000, *, clock=None, sleeper=None):
        if rpm <= 0 or tpm <= 0:
            raise ValueError('Positive quota limits required')
        self.rpm, self.tpm = rpm, tpm
        self.calls = deque()
        self.lock = asyncio.Lock()
        self.clock=clock or time.monotonic
        self.sleeper=sleeper

    def _prune(self,now):
        # Legacy (timestamp,estimate) entries can still seed a process handoff.
        self.calls=deque(entry for entry in self.calls if
            (isinstance(entry,TokenReservation) and entry.settled_at is None)
            or now-(entry.settled_at if isinstance(entry,TokenReservation) else entry[0])<60
            or now-entry[0]<60)

    def _usage(self,now):
        rpm=sum((isinstance(e,TokenReservation) and e.dispatched_at is None) or now-e[0]<60 for e in self.calls)
        tokens=sum(e[1] for e in self.calls if (isinstance(e,TokenReservation) and e.settled_at is None)
            or now-(e.settled_at if isinstance(e,TokenReservation) else e[0])<60)
        return rpm,tokens

    async def acquire(self, tokens, stop_check=None):
        if type(tokens) is not int or tokens<=0 or tokens > self.tpm:
            raise ValueError('Request exceeds configured input token window')
        while True:
            if stop_check is not None:await stop_check()
            async with self.lock:
                now=self.clock();self._prune(now);rpm,used=self._usage(now)
                if rpm<self.rpm and used+tokens<=self.tpm:
                    reservation=TokenReservation(now,tokens);self.calls.append(reservation);return reservation
            # Short interruptible waits observe refunds and file-based drains.
            await (self.sleeper(.25) if self.sleeper else asyncio.sleep(.25))

    async def dispatch(self,reservation):
        async with self.lock:
            if reservation not in self.calls or reservation.settled_at is not None or reservation.dispatched_at is not None:
                raise RuntimeError('Invalid or already dispatched token reservation')
            reservation.dispatched_at=self.clock()

    async def reconcile(self,reservation,actual_tokens=None):
        async with self.lock:
            if reservation.settled_at is not None:return
            # Missing/zero/malformed usage keeps the complete conservative hold.
            # An unexpectedly larger actual count is charged in full, never clipped.
            if type(actual_tokens) is int and actual_tokens>0:reservation.tokens=actual_tokens
            reservation.settled_at=self.clock()

    async def cancel(self,reservation):
        if reservation.dispatched_at is not None:
            await self.reconcile(reservation);return
        async with self.lock:self.calls=deque(e for e in self.calls if e is not reservation)

    async def snapshot(self):
        async with self.lock:
            now=self.clock();self._prune(now);rpm,tokens=self._usage(now)
            return {'requests_in_window':rpm,'input_tokens_accounted':tokens,'rpm_limit':self.rpm,'input_tpm_limit':self.tpm,
                'inflight_reserved_tokens':sum(e.tokens for e in self.calls if isinstance(e,TokenReservation) and e.settled_at is None)}


class GeminiProvider:
    def __init__(self, cfg=None, output_dir=None):
        from google import genai
        self.cfg = cfg or config()['gemini']
        key = secret('GEMINI_API_KEY', 'AEROROZUM_GEMINI_KEY_FILE')
        if not key:
            raise RuntimeError('Hackathon GEMINI_API_KEY is not configured; startup fallback is disabled')
        # Explicit flags prevent inherited Google Cloud variables selecting enterprise.
        self.client = genai.Client(api_key=key, enterprise=False, vertexai=False, http_options={'timeout':self.cfg.get('http_timeout_ms',120000),'api_version':self.cfg.get('api_version','v1beta')})
        self.root = Path(output_dir or data_root() / 'gemini')
        self.root.mkdir(parents=True, exist_ok=True)
        self.limiter = WindowLimiter(self.cfg.get('requests_per_minute',60), self.cfg.get('input_tokens_per_minute',200000))
        self.semaphore = asyncio.Semaphore(self.cfg.get('concurrency',8))
        self.limit = int(self.cfg.get('max_requests',200000))
        self.issued = 0
        self.requests_lock = asyncio.Lock()
        self._init_drain()

    def _init_drain(self):
        if not hasattr(self,'_drain_requested'):
            self._drain_requested=False;self._inflight_requests=0;self._idle=asyncio.Event();self._idle.set()

    @property
    def inflight_requests(self):return getattr(self,'_inflight_requests',0)

    def request_drain(self):
        self._init_drain();self._drain_requested=True

    async def _check_accepting(self):
        self._init_drain();path=self.cfg.get('drain_path')
        if path and Path(path).exists():self.request_drain()
        callback=getattr(self,'should_drain',None) or self.cfg.get('drain_callback')
        if callback is not None:
            decision=callback()
            if hasattr(decision,'__await__'):decision=await decision
            if decision:self.request_drain()
        if self._drain_requested:raise ProviderDraining('Provider drain requested; no new paid request issued')

    async def drain(self,timeout=None):
        self.request_drain()
        if timeout is None:await self._idle.wait()
        else:await asyncio.wait_for(self._idle.wait(),timeout)

    def available_models(self):
        return [m.name for m in self.client.models.list() if 'generateContent' in (m.supported_actions or [])]

    async def generate(self, role, prompt, images=(), model=None, reuse=True, max_output_tokens=None):
        from google.genai import types
        model = model or self.cfg['bulk_model']
        paths = [Path(p) for p in images]
        blobs = [p.read_bytes() for p in paths]
        content_hashes = [hashlib.sha256(b).hexdigest() for b in blobs]
        limit = max_output_tokens or self.cfg.get('reserve_output_tokens',2048)
        temperature=float(self.cfg.get('temperature',.1))
        if not math.isfinite(temperature) or not 0<=temperature<=2:raise ValueError('Temperature must be finite and within 0..2')
        cache_id = request_identity(self.cfg,role=role,prompt=prompt,images=content_hashes,model=model,max_output_tokens=limit)
        cache = self.root / 'responses' / (cache_id + '.json')
        if reuse and cache.exists():
            return json.loads(cache.read_text())['output']
        await self._check_accepting()
        contents = [types.Part.from_bytes(data=b,mime_type=mimetypes.guess_type(str(p))[0] or 'image/jpeg') for p,b in zip(paths,blobs)] + [prompt]
        # Conservative input estimate; provider token usage retained for recalibration.
        image_estimate=self.cfg.get('input_tokens_per_image_estimate',4096)
        if type(image_estimate) is not int or not 1<=image_estimate<=4096:raise ValueError('Invalid input image token estimate')
        estimate = max(1,len(prompt)//3) + image_estimate * len(blobs)
        async with self.semaphore:
            for attempt in range(self.cfg.get('max_retries',5)):
                if reuse and cache.exists():return json.loads(cache.read_text())['output']
                await self._check_accepting()
                queue_gate=getattr(self,'before_queue',None)
                if queue_gate is not None:
                    decision=queue_gate()
                    if hasattr(decision,'__await__'):await decision
                reservation=await self.limiter.acquire(estimate,stop_check=self._check_accepting)
                try:
                    gate=getattr(self,'before_request',None)
                    if gate is not None:
                        decision=gate()
                        if hasattr(decision,'__await__'):await decision
                    await self._check_accepting()
                    async with self.requests_lock:
                        if self._drain_requested:raise ProviderDraining('Provider drain requested; no new paid request issued')
                        if self.issued >= self.limit:raise RuntimeError('Configured Gemini request allowance reached')
                        await self.limiter.dispatch(reservation)
                        if self._drain_requested:raise ProviderDraining('Provider drain requested; no new paid request issued')
                        self.issued+=1;self._inflight_requests+=1;self._idle.clear()
                except BaseException:
                    await self.limiter.cancel(reservation);raise
                start = time.monotonic()
                try:
                    generation=generation_settings(self.cfg,limit)
                    response = await self.client.aio.models.generate_content(model=model,contents=contents,config=types.GenerateContentConfig(**generation))
                    # Preserve every paid response before parsing, including malformed
                    # or truncated output that may still be useful for diagnostics.
                    save(self.root/'raw'/f'{cache_id}-{attempt}.json',dict(id=cache_id,role=role,model=model,backend='developer_api',created_at=utcnow(),prompt=prompt,image_hashes=content_hashes,raw_text=response.text,usage=response.usage_metadata.model_dump(mode='json') if response.usage_metadata else {},finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,attempt=attempt+1))
                    usage=response.usage_metadata.model_dump(mode='json') if response.usage_metadata else {}
                    await self.limiter.reconcile(reservation,usage.get('prompt_token_count',usage.get('promptTokenCount')))
                    output = json.loads(response.text)
                    if not isinstance(output,dict):
                        raise ValueError('Expected JSON object')
                    usage = response.usage_metadata.model_dump(mode='json') if response.usage_metadata else {}
                    record = dict(id=cache_id,role=role,backend='developer_api',model=model,created_at=utcnow(),prompt=prompt,image_hashes=content_hashes,output=output,raw_text=response.text,finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,generation_config=dict(temperature=temperature,max_output_tokens=limit,api_version=self.cfg.get('api_version','v1beta'),thinking_budget=self.cfg.get('thinking_budget')),usage=usage,elapsed_s=time.monotonic()-start,attempt=attempt+1)
                    if self.cfg.get('thinking_level') is not None:record['generation_config']['thinking_level']=str(self.cfg['thinking_level']).lower()
                    save(cache,record)
                    return output
                except Exception as ex:
                    status = getattr(ex,'code',None)
                    save(self.root/'errors'/f'{cache_id}-{attempt}.json',dict(role=role,model=model,created_at=utcnow(),error=redacted_error(ex),status_code=status,attempt=attempt+1))
                    callback=getattr(self,'on_request_error',None)
                    if callback is not None:
                        decision=callback(ex,status)
                        if hasattr(decision,'__await__'):await decision
                    if status in (400,401,403,404) or attempt+1 >= self.cfg.get('max_retries',5):
                        raise RuntimeError(redacted_error(ex)) from None
                    retry_delay=min(30,2**attempt)
                finally:
                    await self.limiter.reconcile(reservation)
                    self._inflight_requests-=1
                    if not self._inflight_requests:self._idle.set()
                await asyncio.sleep(retry_delay)

    async def analyze(self, image, model=None):
        proposal = await self.generate('thermal_teacher',DETECTION_PROMPT,[image],model=model)
        review = await self.generate('thermal_review',REVIEW_PROMPT+json.dumps(proposal),[image],model=model)
        return dict(detector=proposal,reviewer=review,ground_truth=False,review_independence='separate prompt, same provider; correlated model checks')

    async def close(self):
        await self.drain()
        await self.client.aio.aclose()

    def response_record(self,role,prompt,images=(),model=None,max_output_tokens=None):
        model=model or self.cfg['bulk_model'];limit=max_output_tokens or self.cfg.get('reserve_output_tokens',2048)
        hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in images]
        key=request_identity(self.cfg,role=role,prompt=prompt,images=hashes,model=model,max_output_tokens=limit)
        path=self.root/'responses'/(key+'.json')
        if path.exists():return json.loads(path.read_text())
        raw=sorted((self.root/'raw').glob(key+'-*.json'))
        return json.loads(raw[-1].read_text()) if raw else None


async def analyze_manifest(manifest, maximum=None, model=None):
    """Resumable high-volume workers. Teacher results never rewrite labels."""
    from .jobs import job, update
    doc = json.loads(Path(manifest).read_text())
    rows = doc.get('images',doc.get('observations',[]))
    if maximum is not None:
        rows = rows[:maximum]
    run = job('gemini_teacher',total=len(rows),completed=0,failed=0)
    provider = GeminiProvider()
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    failures = []
    async def worker():
        while not queue.empty():
            try: row=queue.get_nowait()
            except asyncio.QueueEmpty: return
            try:
                raw = Path(row['image'])
                path = raw if raw.is_absolute() else data_root()/raw
                result = await provider.analyze(path,model=model)
                # Split/provenance stays explicit and is enforced by training loader.
                output = dict(id=row['id'],split=row.get('split'),provenance='gemini_teacher',image=row['image'],image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result=result)
                save(provider.root/'evidence'/(digest({'id':row['id'],'image':output['image_sha256']})+'.json'),output)
                update(run,completed=run['completed']+1)
            except Exception as ex:
                failures.append(dict(id=row.get('id'),error=redacted_error(ex)))
                update(run,failed=run['failed']+1)
            finally: queue.task_done()
    try:
        await asyncio.gather(*(worker() for _ in range(provider.cfg.get('concurrency',8))))
    finally:
        await provider.close()
    update(run,'failed' if failures else 'completed',failures=failures)
    return run
