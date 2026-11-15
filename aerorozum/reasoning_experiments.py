"""Paired Robotics prompt/effort evaluation on frozen development images only."""
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

from .common import config, digest, redacted_error, save, utcnow
from .evaluation_completions import completion_text
from .label_prompts import PROMPT_V2, PROMPT_V3
from .label_schema import normalize_thermal_output
from .labelling import MODEL, classify_response
from .providers import GeminiProvider, WindowLimiter, request_identity, ProviderDraining
from .training import detection_score

PROMPTS = {
    'thermal-v2-recall-calibrated': PROMPT_V2,
    'thermal-v3-two-scale-review': PROMPT_V3
}
LEVELS = ('high', 'medium', 'minimal')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def freeze(root, output, replicas=3):
    output, root = Path(output), Path(root).resolve()
    path = output / 'protocol.json'
    if path.exists():
        protocol = json.loads(path.read_text())
        if protocol['sha256'] != digest({k:v for k,v in protocol.items() if k!='sha256'}):
            raise ValueError('Frozen reasoning protocol changed')
        if file_hash(root / 'pipeline/validation.json') != protocol['manifest_sha256']:
            raise ValueError('Development manifest changed after freezing')
        for row in protocol['images']:
            image = (root / row['image']).resolve()
            if not image.is_relative_to(root) or file_hash(image) != row['sha256']:
                raise ValueError('Frozen development pixels changed')
        return protocol
    manifest = root / 'pipeline/validation.json'
    rows = json.loads(manifest.read_text())['images']
    if len(rows) != 120:
        raise ValueError('Expected the established 120-image development comparison')
    for row in rows:
        image = (root / row['image']).resolve()
        if row.get('split') != 'validation' or row.get('zone') in ('zone8', 'zone9'):
            raise ValueError('Only development images may enter prompt selection')
        if not image.is_relative_to(root) or file_hash(image) != row['sha256']:
            raise ValueError('Development image hash/path changed')
    protocol = dict(schema_version=1, created_at=utcnow(), model=MODEL, images=rows,
        prompts=PROMPTS, thinking_levels=list(LEVELS), replicas=replicas,
        max_output_tokens=65536, temperature=.1, confidence_threshold=.6,
        iou_threshold=.5, manifest_sha256=file_hash(manifest), training_eligible=False,
        test_accessed=False, reasoning_control='Native thinking_level; no thinking_budget is sent',
        output_cap_scope='Thinking and answer share the same 65,536-token allowance in every arm',
        documentation='https://ai.google.dev/gemini-api/docs/thinking',
        limits=dict(requests_per_minute=120, input_tokens_per_minute=500000, concurrency=12),
        interpretation='Repeated development label agreement; negative examples include machine-screened frames, not confirmed absence')
    protocol['sha256'] = digest(protocol)
    save(path, protocol)
    return protocol


def score_output(text, row):
    try:
        output = json.loads(text)
        normalized, changes = normalize_thermal_output(output)
        classified = classify_response(normalized, .6)
        predictions = [d['bbox'] for d in classified['accepted']]
        valid, error = True, None
    except (ValueError, TypeError, KeyError) as exception:
        output, predictions, changes = None, [], []
        valid, error = False, str(exception)[:200]
    return dict(model_output=output, normalization_trace=changes, valid_output=valid,
                schema_error=error, score=detection_score(predictions, row.get('boxes', [])),
                has_gold=bool(row.get('boxes')), predictions=predictions)


def summarize(protocol, records):
    groups = defaultdict(list)
    for row in records:
        if row['status'] != 'transport_error':
            groups[(row['prompt_version'], row['thinking_level'], row['replica'])].append(row)
    arms = []
    for (prompt, level, replica), rows in sorted(groups.items()):
        tp = sum(r['score']['tp'] for r in rows)
        fp = sum(r['score']['fp'] for r in rows)
        fn = sum(r['score']['fn'] for r in rows)
        negatives = [r for r in rows if not r['has_gold']]
        usage = defaultdict(int)
        for row in rows:
            for key, value in row.get('usage', {}).items():
                if type(value) is int:
                    usage[key] += value
        arms.append(dict(prompt_version=prompt, thinking_level=level, replica=replica,
            images=len(rows), complete=len(rows)==len(protocol['images']), tp=tp, fp=fp, fn=fn,
            micro_f2=5*tp/max(1,5*tp+4*fn+fp), valid_output_fraction=sum(r['valid_output'] for r in rows)/len(rows),
            negative_frame_disagreement_rate=sum(r['score']['fp']>0 for r in negatives)/len(negatives) if negatives else None,
            total_usage=dict(usage), total_observed_latency_s=sum(r.get('elapsed_s',0) for r in rows)))
    expected = len(protocol['images'])*len(protocol['prompts'])*len(protocol['thinking_levels'])*protocol['replicas']
    return dict(status='completed' if len(records)==expected and not any(r['status']=='transport_error' for r in records) else 'incomplete',
                at=utcnow(), expected=expected, retained_results=len(records), arms=arms,
                protocol_sha256=protocol['sha256'], test_accessed=False, training_eligible=False,
                interpretation=protocol['interpretation'])


async def run(root, output, store, deadline, replicas=3, probe=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = freeze(root, output, replicas)
    stop = asyncio.Event()
    fatal = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    limiter = WindowLimiter(**dict(rpm=120, tpm=500000))
    providers = {}
    for level in LEVELS:
        cfg = dict(config()['gemini'], thinking_level=level, temperature=protocol['temperature'], reserve_output_tokens=65536,
                   max_retries=1, concurrency=12, http_timeout_ms=1200000,
                   requests_per_minute=120, input_tokens_per_minute=500000)
        cfg.pop('thinking_budget', None)
        provider = GeminiProvider(cfg, output_dir=output/'provider'/level)
        provider.limiter = limiter
        provider.should_drain = lambda: stop.is_set() or fatal.is_set() or datetime.now(timezone.utc) >= deadline
        providers[level] = provider
    queue = asyncio.Queue()
    for replica in range(protocol['replicas']):
        for index, row in enumerate(protocol['images']):
            for level in LEVELS:
                for prompt in (list(PROMPTS) if index%2==0 else list(PROMPTS)[::-1]):
                    queue.put_nowait((row, prompt, level, replica))
    records, errors = [], []
    expected = queue.qsize()
    await asyncio.to_thread(store.put, digest(['protocol',protocol['sha256']]), protocol, 'progress')

    async def worker():
        while not queue.empty() and not fatal.is_set() and not stop.is_set() and datetime.now(timezone.utc)<deadline:
            try:
                row, prompt_version, level, replica = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            key = digest([protocol['sha256'],row['id'],prompt_version,level,replica])
            path = output/'results'/(key+'.json')
            provider = providers[level]
            prompt = protocol['prompts'][prompt_version]
            role = 'robotics-reasoning-study-v1:'+prompt_version+':replica-'+str(replica)
            try:
                record = json.loads(path.read_text()) if path.exists() else await asyncio.to_thread(store.get,key)
                if record is not None and (record.get('protocol_sha256')!=protocol['sha256'] or record.get('sha256')!=row['sha256']):
                    raise ValueError('Retained study result identity differs')
                # Transport failures stay in the immutable ledger. A later
                # attempt needs a separately declared protocol, never a silent
                # overwrite or an automatic replay of a paid invalid answer.
                if record is None:
                    started = time.monotonic()
                    try:
                        text = await completion_text(provider,role,prompt,[root/row['image']],model=MODEL,max_output_tokens=65536)
                        prediction = score_output(text,row)
                        status = 'completed' if prediction['valid_output'] else 'invalid_paid_response'
                        error = None
                    except ProviderDraining:
                        return
                    except Exception as exception:
                        text, prediction, status, error = None, {}, 'transport_error', redacted_error(exception)
                        # Rejecting an unsupported profile is an infrastructure issue,
                        # never a model-quality zero. Settle other accepted requests.
                        if any(code in error for code in ['400','401','403','404']):
                            fatal.set()
                    identity = request_identity(provider.cfg,role=role,prompt=prompt,images=[row['sha256']],model=MODEL,max_output_tokens=65536)
                    raw_path = provider.root/'raw'/(identity+'-0.json')
                    raw = json.loads(raw_path.read_text()) if raw_path.exists() else None
                    record = dict(id=row['id'], sha256=row['sha256'], model=MODEL,
                        protocol_sha256=protocol['sha256'], prompt_version=prompt_version,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), thinking_level=level,
                        replica=replica, status=status, error=error, raw_response=raw,
                        usage=raw.get('usage',{}) if raw else {}, elapsed_s=time.monotonic()-started,
                        training_eligible=False, source_split='validation', at=utcnow(), **prediction)
                    save(path,record)
                records.append(record)
                await asyncio.to_thread(store.write_frame,key,record,record.get('raw_response'))
            except Exception as exception:
                errors.append(dict(id=row['id'],level=level,prompt=prompt_version,error=redacted_error(exception)))
            finally:
                save(output/'progress.json',dict(status='configuration_error' if fatal.is_set() else 'running',
                    at=utcnow(),completed=len(records),total=expected,errors=errors[-10:],
                    issued=sum(p.issued for p in providers.values()),archive_failures=len(errors)))
                queue.task_done()
            if probe:
                return
    try:
        await asyncio.gather(*(worker() for _ in range(1 if probe else 12)))
    finally:
        await asyncio.gather(*(provider.close() for provider in providers.values()))
    report = summarize(protocol,records)
    report.update(configuration_error=fatal.is_set(),archive_errors=errors,actual_requests=sum(p.issued for p in providers.values()))
    save(output/'report.json',report)
    await asyncio.to_thread(store.put,digest(['report',protocol['sha256'],digest(report)]),report,'progress')
    from .label_archive import archive_pass
    for level in LEVELS:
        audit = await asyncio.to_thread(archive_pass, output/'provider'/level, store,
                                       output/'archive-state'/(level+'.json'))
        save(output/('raw-archive-'+level+'.json'),audit)
    audit = await asyncio.to_thread(archive_pass, output, store, output/'archive-state/results.json')
    save(output/'results-archive.json',audit)
    return report
