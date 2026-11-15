"""Queue the next three-seed visual generation from completed HIGH labels.

Keeps every admissible positive, balances explicit zero frames, preserves the
manual anchors and whole-session splits, and retains the frozen development
comparison. New labels and data composition change together; no causal label-
source isolation or independent field-accuracy claim is made.
"""
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/opt/aerorozum');ART=ROOT/'artifacts';INPUT=ROOT/'airborne-labeler/artifacts'
NIGHT=ART/'overnight-20261115T000942'


def recipe_helper():
    spec=importlib.util.spec_from_file_location('frozen_visual_recipe',ROOT/'postsubmission/replicate_visual_recipe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prepare():
    from aerorozum.common import save,digest,utcnow
    from aerorozum.real_ingester import RealIngester
    destination=ART/'postsubmission-high-teacher-visual-v1';frozen=destination/'catalog.json'
    if frozen.exists():return json.loads(frozen.read_text())
    base_path=ART/'pipeline/inputs/vm-ingestion/candidate.json';base=json.loads(base_path.read_text())
    registry_path=INPUT/'labelling/robotics-er2-airborne/session-splits.json'
    registry=json.loads(registry_path.read_text());sessions=registry['sessions']
    folders=[INPUT/'labelling'/name for name in ['robotics-er2-high-thinking-v1','robotics-er2-high-thinking-requalified-v1','robotics-er2-high-thinking-recovery-v1']]
    if not all((p/'settled.json').exists() for p in folders):return None
    records=[];held=set(base['sealed_test_image_hashes']);held.update(r['sha256'] for r in base['images'] if r['split']!='train')
    for folder in folders:
        for file in (folder/'results').glob('*.json'):
            row=json.loads(file.read_text())
            if row.get('split') in ('validation','test','fast_test','direct_test') and row.get('sha256'):held.add(row['sha256'])
            records.append((file,row))
    images=[r for r in base['images'] if r.get('row_kind')=='manual_anchor' and r['split']=='train' and r.get('training_eligible')]
    private=ART/'high-teacher-inputs-v1';reader=RealIngester(private);excluded=Counter();native=[]
    for file,original in records:
        row=dict(original)
        if row.get('split')!='train':continue
        if row.get('airborne_eligible') is not True or row.get('eligibility',{}).get('status')!='eligible':
            excluded['not_qualified_airborne']+=1;continue
        if (row.get('inference_profile') or {}).get('thinking_level')!='high':
            excluded['not_exact_high_profile']+=1;continue
        if row.get('sha256') in held:excluded['held_out_pixel_hash']+=1;continue
        if not row.get('image'):excluded['invalid_paid_response']+=1;continue
        raw=Path(row['image']);raw=raw if raw.is_absolute() else INPUT/raw;raw=raw.resolve()
        if not raw.is_relative_to(ROOT) or hashlib.sha256(raw.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Paid HIGH input path/hash changed')
        target=private/'native'/(row['sha256']+raw.suffix.lower());target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(raw,target)
        row['image']=str(target)
        try:
            item=reader.native(row,sessions)
        except ValueError as error:
            excluded[str(error)]+=1;continue
        item.update(image=str(target.relative_to(ART)),training_eligible=True,flight_state='airborne',
            flight_eligibility=original['eligibility'],cross_split_duplicate=False,
            inference_profile=original['inference_profile'],parent_cache_key=original['cache_key'],
            parent_result_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            original_labels_overwritten=False)
        native.append(item)
    reader.db.commit();reader.db.close()
    unique={};conflicts=[]
    for row in images+native:
        previous=unique.get(row['sha256'])
        if previous:
            if previous['boxes']!=row['boxes']:
                conflicts.append(dict(sha256=row['sha256'],preferred='manual_anchor' if previous['row_kind']=='manual_anchor' else 'quarantined_native_conflict'))
                if previous['row_kind']!='manual_anchor':previous['training_excluded_reason']='conflicting_native_labels_same_pixels'
            continue
        unique[row['sha256']]=row
    catalog=dict(images=list(unique.values()),session_splits=sessions,sealed_test_image_hashes=base['sealed_test_image_hashes'],
        nontraining_image_hashes=sorted(held),
        created_at=utcnow(),source_catalog_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        source_label_folders=[str(p) for p in folders],manual_anchors=len(images),native_admitted=len(native),
        exclusions=dict(excluded),duplicate_label_conflicts=conflicts,
        no_reference_test_accessed=True,automatic_promotion=False,
        interpretation='New HIGH pseudo-label cohort with the same archived training anchors. No physical mine identity is independently verified.')
    catalog['dataset_signature']=digest(dict(rows=[(r['sha256'],r['boxes'],r.get('training_excluded_reason')) for r in catalog['images']],
        nontraining_image_hashes=sorted(held),session_splits=sessions))
    save(frozen,catalog);return catalog


def launch(seed,gpu,catalog):
    from aerorozum.expanded_curriculum import select
    from aerorozum.common import save
    module=recipe_helper();out=ART/f'postsubmission-high-teacher-visual-v1/seed-{seed}'
    if out.exists():raise ValueError('A high-teacher run already exists; preserve its outcome')
    validation=ART/'pipeline/validation.json';dev=json.loads(validation.read_text())
    manifest=select(catalog,[r['sha256'] for r in dev['images']],seed)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(ROOT/'models/gemma-4-12B-it')
    sizes=[len(processor.tokenizer.encode(json.dumps({'detections':[{'bbox':b.get('bbox',b)} for b in r['boxes']]},separators=(',',':')),add_special_tokens=False)) for r in manifest['images']]
    limit=max(512,128*math.ceil((max(sizes)+32)/128))
    if limit>2048:raise ValueError('Gold targets require a separately qualified longer-completion recipe; no rows have been dropped')
    source=ROOT/('postsubmission/high-teacher-visual-source-'+str(limit)+'-snapshots-v1')
    if not source.exists():
        shutil.copytree(ROOT/'postsubmission/expanded-visual-source-v2',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        target=source/'aerorozum/training.py';code=target.read_text()
        if code.count('max_completion_length=512')!=1:raise ValueError('Unexpected source completion allowance')
        code=code.replace('max_completion_length=512','max_completion_length='+str(limit))
        # The diagnostic must use a nonconstant original, otherwise a black
        # camera frame compared with black pixels cannot test image influence.
        probe="else rows[0]['image_path']"
        if code.count(probe)!=1:raise ValueError('Unexpected image-influence probe source')
        code=code.replace(probe,"else next((r['image_path'] for r in rows if r.get('image_statistics',{}).get('gray_stddev',0)>=3),rows[0]['image_path'])")
        line=next(line for line in code.splitlines() if 'trainer=SFTTrainer(' in line)
        if line.count('callbacks=[monitor]')!=1:raise ValueError('Unexpected SFT callback source')
        code=code.replace(line,"            from .snapshot_monitor import SnapshotMonitor\n"+line.replace('callbacks=[monitor]',"callbacks=[monitor,SnapshotMonitor(output/'fixed-sft-snapshots')]"))
        checked=ROOT/'postsubmission/snapshot-qualification-source-v1/aerorozum/snapshot_monitor.py'
        qualification=json.loads((NIGHT/'snapshot-monitor-qualification.json').read_text())
        if qualification.get('status')!='passed' or module.sha(checked)!=qualification['source_sha256']:
            raise ValueError('Read-only snapshot qualification differs')
        shutil.copyfile(checked,source/'aerorozum/snapshot_monitor.py')
        target.write_text(code)
        save(source/'high-teacher-source-receipt.json',dict(completion_limit=limit,
            all_admitted_positives_preserved=True,probe_uses_nonconstant_original=True,optimizer_unchanged=True,
            fixed_read_only_sft_snapshots=True,snapshot_monitor_sha256=qualification['source_sha256']))
    receipt=json.loads((source/'high-teacher-source-receipt.json').read_text())
    if not receipt.get('fixed_read_only_sft_snapshots') or module.sha(source/'aerorozum/snapshot_monitor.py')!=receipt['snapshot_monitor_sha256']:
        raise ValueError('Incomplete or changed read-only snapshot source')
    parent_path=ART/'submission-visual-projection-v2/updates-480/result.json';parent=json.loads(parent_path.read_text())
    spec=module.recipe(parent,seed)
    spec.update(kind='high_teacher_fresh_visual',sft_steps=math.ceil(len(manifest['images'])*3.2/4),grpo_steps=48,continuation_steps=0,
        root=str(ROOT),namespace=f'postsubmission-high-teacher-visual-v1/seed-{seed}',
        manifest=str(out/'train.json'),validation=str(validation),output=str(out),source_initial=str(source),
        source_continuation=str(ROOT/'harbor/grpo-only-source-v1'),parent_adapter='unused_fresh_base',
        maximum_gold_target_tokens=max(sizes),grpo_completion_limit=limit,fixed_epoch_target=3.2,
        automatic_promotion=False,test_access=False,high_teacher_dataset_signature=catalog['dataset_signature'],
        fixed_sft_snapshot_steps=sorted({math.ceil(len(manifest['images'])*epoch/4) for epoch in [.5,1.,2.]}),
        lease=dict(id='high-teacher-visual-'+str(seed),gpu_ids=[gpu],expires_at=(datetime.now(timezone.utc)+timedelta(hours=19)).isoformat(),
                   purpose='Next full-positive HIGH-teacher visual generation; frozen development evaluation'))
    module.lease(ROOT,spec['lease']);started=False
    try:
        uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        if gpu==0 or uuids[gpu] in used:raise ValueError('GPU is still occupied; preserve its current job')
        save(out/'train.json',manifest)
        files=[Path(__file__).resolve(),Path(spec['manifest']),validation,ART/'postsubmission-high-teacher-visual-v1/catalog.json',
            source/'aerorozum/training.py',source/'aerorozum/gradient_monitor.py',source/'aerorozum/snapshot_monitor.py',
            NIGHT/'snapshot-monitor-qualification.json',source/'configs/research.json',ROOT/'postsubmission/replicate_visual_recipe.py']
        spec['input_hashes']={str(p):module.sha(p) for p in files};save(out/'specification.json',spec)
        env=dict(CUDA_VISIBLE_DEVICES=str(gpu),AEROROZUM_DATA_ROOT=str(ART),AEROROZUM_LOCAL_MODEL_PATH=str(ROOT/'models/gemma-4-12B-it'),
            AEROROZUM_GPU_MEMORY_FRACTION='.88',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
            AEROROZUM_SFT_SNAPSHOT_STEPS=json.dumps(spec['fixed_sft_snapshot_steps']))
        command=['systemd-run','--unit=aerorozum-high-teacher-visual-'+str(seed),'--property=RuntimeMaxSec=64800','--property=CPUQuota=300%','--property=MemoryMax=64G',
            '--property=StandardOutput=append:'+str(out/'worker.log'),'--property=StandardError=append:'+str(out/'worker.log')]
        command.extend('--setenv='+key+'='+value for key,value in env.items())
        command.extend([str(ROOT/'venv/bin/python'),'-u',str(ROOT/'postsubmission/replicate_visual_recipe.py'),'worker',str(out/'specification.json')])
        subprocess.run(command,check=True,timeout=30);started=True
        print(json.dumps(dict(seed=seed,gpu=gpu,images=len(manifest['images']),sft_steps=spec['sft_steps'],completion_limit=limit)),flush=True)
    finally:
        if not started:module.lease(ROOT,spec['lease'],remove=True)


if __name__=='__main__':
    pending={17:1,43:2,91:3};deadline=datetime.fromisoformat('2026-11-15T07:00:00+00:00')
    while pending and datetime.now(timezone.utc)<deadline:
        catalog=prepare()
        if catalog is None:time.sleep(30);continue
        for seed,gpu in list(pending.items()):
            parent=ART/f'postsubmission-expanded-controls-v1/seed-{seed}/status.json'
            if not parent.exists() or json.loads(parent.read_text()).get('status') not in ('completed','failed'):continue
            if (ART/f'postsubmission-high-teacher-visual-v1/seed-{seed}').exists():pending.pop(seed);continue
            try:launch(seed,gpu,catalog);pending.pop(seed)
            except Exception as error:print(json.dumps(dict(seed=seed,error=str(error)[:1200])),flush=True)
        time.sleep(30)
