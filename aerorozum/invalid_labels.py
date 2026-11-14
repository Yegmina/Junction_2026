"""Auditable bounded recovery of malformed paid labels; originals stay immutable."""
from __future__ import annotations
import copy,json
from pathlib import Path
from .common import data_root,digest,save,utcnow,redacted_error
from .labelling import MODEL,PROMPT,classify_response,label_frames,inference_cache_key
from .cloud_cache import GCSRecordStore


def normalize_box_aliases(output):
    """Only rename unambiguous spelling aliases. Never infer missing geometry."""
    fixed=copy.deepcopy(output);changes=[]
    if not isinstance(fixed,dict) or not isinstance(fixed.get('detections'),list):
        raise ValueError('No explicit detections array; cannot repair absence')
    for index,candidate in enumerate(fixed['detections']):
        if not isinstance(candidate,dict) or not isinstance(candidate.get('bbox'),dict):
            raise ValueError('Candidate lacks explicit box')
        box=candidate['bbox']
        for alias,canonical in {'xmin':'x_min','ymin':'y_min','xmax':'x_max','ymax':'y_max'}.items():
            if alias not in box:continue
            if canonical in box and box[canonical]!=box[alias]:
                raise ValueError('Conflicting box aliases; manual interpretation forbidden')
            box[canonical]=box.pop(alias);changes.append({'candidate':index,'from':alias,'to':canonical})
    classify_response(fixed)
    if not changes:raise ValueError('No safe alias normalization found')
    return fixed,changes


class RecoveryProvider:
    def __init__(self,provider,key,*,normalized=None,raw=None,allow_network=False):
        self.provider=provider;self.key=key;self.normalized=normalized;self.raw=raw
        self.allow_network=allow_network;self.role='robotics_er2_thermal_labels_retry1:'+key
        self.cfg={**getattr(provider,'cfg',{}),'inference_revision':('schema-alias-v1:' if normalized is not None else 'invalid-retry1:')+key}
    async def generate(self,role,prompt,images,model):
        if self.normalized is not None:return copy.deepcopy(self.normalized)
        cached=self.provider.response_record(self.role,prompt,images,model=model)
        if cached is not None and 'output' in cached:return cached['output']
        if not self.allow_network:raise RuntimeError('Bounded retry already reserved; no additional paid attempt authorized')
        # The outer provider gates/rate limiter/deadline remain authoritative.
        previous=self.provider.cfg.get('max_retries',5);self.provider.cfg['max_retries']=1
        try:return await self.provider.generate(self.role,prompt,images,model=model)
        finally:self.provider.cfg['max_retries']=previous
    def response_record(self,role,prompt,images,model):
        return self.raw if self.normalized is not None else self.provider.response_record(self.role,prompt,images,model=model)


def find_source_rows(root,identities):
    found={}
    for path in (root/'labelling').glob('*/frames/*/sampling.json'):
        try:doc=json.loads(path.read_text())
        except (OSError,ValueError):continue
        for row in doc.get('images',[]):
            identity=(row.get('sha256'),row.get('source',{}).get('generation'))
            if identity in identities:found.setdefault(identity,row)
    return found


async def recover_invalid(provider,root=None):
    root=Path(root or data_root());labels=root/'labelling/robotics-er2-5fps/labels';out=labels/'invalid-response-recovery';out.mkdir(parents=True,exist_ok=True)
    if (root/'labelling/airborne-only-drain.json').exists():
        deferred={'status':'deferred_during_airborne_only_prompt_migration','at':utcnow(),'original_records_modified':False,'paid_inference_issued':False}
        save(out/'status.json',deferred);return deferred
    invalid=[]
    for path in (labels/'results').glob('*.json'):
        row=json.loads(path.read_text())
        if row.get('status')=='invalid_model_response' and row.get('model')==MODEL and not row.get('paid_response_recovery'):invalid.append((path.stem,row))
    rows=find_source_rows(root,{(r['sha256'],r.get('source',{}).get('generation')) for _,r in invalid});results=[]
    def archive_mapping(key,receipt,bucket):
        pending=out/(key+'-archive-pending.json')
        try:
            store=GCSRecordStore(bucket);uri=store.put(digest(['repair-index',key]),receipt,kind='progress')
            remote=store.get(digest(['repair-index',key]),kind='progress')
            if not remote or remote.get('replacement_cache_key')!=receipt['replacement_cache_key']:raise ValueError('Repair mapping remote readback mismatch')
            save(out/(key+'-archive.json'),{'status':'uploaded','at':utcnow(),'uri':uri,'receipt':receipt})
            if pending.exists():pending.unlink()
        except Exception as ex:save(pending,{'error':redacted_error(ex),'receipt':receipt})
    for key,old in invalid:
        mapping=labels/'repair-index'/(key+'.json')
        if mapping.exists():
            if (out/(key+'-archive-pending.json')).exists():archive_mapping(key,json.loads(mapping.read_text()),old.get('source',{}).get('bucket',''))
            results.append({'original_cache_key':key,'status':'already_recovered'});continue
        row=rows.get((old['sha256'],old.get('source',{}).get('generation')))
        if row is None:results.append({'original_cache_key':key,'status':'quarantined_source_row_missing'});continue
        try:normalized,changes=normalize_box_aliases(old['model_output'])
        except ValueError:normalized=None;changes=[]
        reservation=out/(key+'.json');first=not reservation.exists()
        if first:save(reservation,{'original_cache_key':key,'at':utcnow(),'status':'reserved','mode':'schema_alias_no_inference' if normalized is not None else 'one_fresh_attempt_max','original_paid_record_retained':True})
        adapter=RecoveryProvider(provider,key,normalized=normalized,raw=old.get('raw_response'),allow_network=first)
        revised=copy.deepcopy(row);revised['paid_response_recovery']={'original_cache_key':key,'mode':'unambiguous_box_key_aliases' if normalized is not None else 'single_fresh_attempt','key_changes':changes,'original_paid_record_retained':True,'original_raw_text_modified':False}
        repair_labels=out/'labels';repair_labels.mkdir(exist_ok=True)
        for name in ('results','paid-responses','checkpoints','failures','repair-index'):
            (labels/name).mkdir(exist_ok=True)
            if not (repair_labels/name).exists():(repair_labels/name).symlink_to(labels/name,target_is_directory=True)
        result=await label_frames([revised],repair_labels,provider=adapter,concurrency=1)
        replacement=inference_cache_key(row,row['sha256'],adapter)
        if result['labelled']==1:
            receipt={'original_cache_key':key,'replacement_cache_key':replacement,'sha256':row['sha256'],'model':MODEL,'at':utcnow(),'original_paid_record_retained':True,'mode':revised['paid_response_recovery']['mode']}
            save(mapping,receipt);save(reservation,{**receipt,'status':'recovered'})
            archive_mapping(key,receipt,row.get('source',{}).get('bucket',''))
            results.append({**receipt,'status':'recovered'})
        else:results.append({'original_cache_key':key,'status':'bounded_attempt_failed_original_retained','result':result})
    document={'at':utcnow(),'invalid_originals':len(invalid),'results':results,'fresh_attempts_bounded_per_original':1,'original_records_modified':False}
    save(out/'status.json',document);return document
