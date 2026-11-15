"""Freeze an expanded real-image curriculum and launch the established recipe."""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import shutil


def launch(root, gpu=1, seed=17, optimized=False):
    from aerorozum.common import save
    from aerorozum.expanded_curriculum import select
    root = Path(root).resolve()
    worker = root/'postsubmission/replicate_visual_recipe.py'
    imported = importlib.util.spec_from_file_location('frozen_replication', worker)
    helper = importlib.util.module_from_spec(imported); imported.loader.exec_module(helper)
    out = root/'artifacts/postsubmission-expanded-visual-v1'/('seed-'+str(seed))
    if out.exists():
        raise ValueError('A declared run already exists; inspect its original outcome')
    catalog_path = root/'artifacts/pipeline/inputs/vm-ingestion/candidate.json'
    catalog = json.loads(catalog_path.read_text())
    validation = root/'artifacts/pipeline/validation.json'
    curriculum = select(catalog, [r['sha256'] for r in json.loads(validation.read_text())['images']], seed)
    original_source = root/'harbor/visual-training-source'
    # Check the existing completion allowance against every gold target before
    # launching; never drop complex frames just to fit a small output limit.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(root/'models/gemma-4-12B-it')
    sizes = [len(processor.tokenizer.encode(json.dumps({'detections':[{'bbox':b.get('bbox',b)} for b in row['boxes']]},separators=(',',':')),add_special_tokens=False)) for row in curriculum['images']]
    completion_limit = max(384, 128*math.ceil((max(sizes)+32)/128))
    if completion_limit > 1024:
        raise ValueError('Expanded labels exceed the declared 1,024-token capacity bound')
    benchmark = root/'artifacts/overnight-20261115T000942/gradient-monitor-benchmark.json'
    monitor_module = Path(__file__).resolve().parents[1]/'aerorozum/gradient_monitor.py'
    if optimized:
        qualification=json.loads(benchmark.read_text())
        if not all(qualification.get(k) is True for k in ['equivalent_statistics','unchanged_gradients','identical_optimizer_trajectory','nan_and_infinity_detected']):
            raise ValueError('The read-only gradient monitor has not passed equivalence checks')
        if qualification['monitor_source_sha256']!=helper.sha(monitor_module):
            raise ValueError('The qualified gradient monitor source differs')
    source = root/('postsubmission/expanded-visual-source-v2' if optimized else 'postsubmission/expanded-visual-source-v1')
    if not source.exists():
        shutil.copytree(original_source,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        target = source/'aerorozum/training.py'
        code = target.read_text()
        old = 'max_completion_length=384'
        if code.count(old) != 1:
            raise ValueError('Expected one detector completion limit in the frozen training source')
        code=code.replace(old,'max_completion_length='+str(completion_limit))
        if optimized:
            start=code.index('                squared=0.;count=0;visual_squared=0.')
            end=code.index("                if lora_scope=='language_visual_projection'",start)
            code=code[:start]+'''                from .gradient_monitor import statistics
                observed=statistics(model.named_parameters())
                squared=observed['squared'];count=observed['count'];visual_squared=observed['visual_squared']
'''+code[end:]
            shutil.copyfile(monitor_module,source/'aerorozum/gradient_monitor.py')
        target.write_text(code)
        save(source/'expanded-source-receipt.json',dict(original=str(original_source),
            original_training_sha256=helper.sha(original_source/'aerorozum/training.py'),
            changes={'max_completion_length':completion_limit,'read_only_device_aggregated_monitor':optimized},target_capacity_checked=True,
            no_training_rows_dropped=True))
    elif json.loads((source/'expanded-source-receipt.json').read_text())['changes']['max_completion_length'] != completion_limit:
        raise ValueError('Expanded source capacity differs from the declared recipe')
    parent_path = root/'artifacts/submission-visual-projection-v2/updates-480/result.json'
    parent = json.loads(parent_path.read_text())
    spec = helper.recipe(parent, seed)
    spec.update(kind='expanded_training_set', sft_steps=math.ceil(len(curriculum['images'])*3.2/4),
                grpo_steps=48, continuation_steps=0,
                root=str(root),namespace='postsubmission-expanded-visual-v1/seed-'+str(seed),
                manifest=str(out/'train.json'),validation=str(validation),output=str(out),
                source_initial=str(source),source_continuation=str(root/'harbor/grpo-only-source-v1'),
                parent_adapter='unused_fresh_base',automatic_promotion=False,
                fixed_epoch_target=3.2,maximum_gold_target_tokens=max(sizes),
                grpo_completion_limit=completion_limit,
                read_only_monitor_optimized=optimized,
                interpretation='Expanded dataset and longer training are changed together; this is not a causal RL ablation',
                lease=dict(id='expanded-visual-'+str(seed),gpu_ids=[gpu],
                    expires_at=(datetime.now(timezone.utc)+timedelta(hours=6)).isoformat(),
                    purpose='Expanded balanced visual curriculum with frozen development evaluation'))
    helper.lease(root,spec['lease'])
    started = False
    try:
        uuids = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        used = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        if uuids[gpu] in used or gpu == 0:
            raise ValueError('Preserve occupied GPUs and the production inference GPU')
        save(out/'train.json',curriculum)
        files = [worker,Path(__file__).resolve(),catalog_path,out/'train.json',validation,parent_path]
        if optimized:files.extend([benchmark,monitor_module,source/'aerorozum/gradient_monitor.py'])
        for src in [source,root/'harbor/grpo-only-source-v1']:
            files.extend(src/name for name in ['aerorozum/training.py','aerorozum/stream_training.py','configs/research.json'])
        spec['input_hashes'] = {str(p):helper.sha(p) for p in files}
        # Validate all image hashes, flight provenance, confidence and session
        # boundaries using the exact training source before allocating a model.
        script = 'from aerorozum.training import load_stream_manifest; import sys; rows,meta=load_stream_manifest(sys.argv[1],"stream_sft"); print(len(rows))'
        import os
        env = dict(os.environ,PYTHONPATH=str(source),AEROROZUM_DATA_ROOT=str(root/'artifacts'))
        subprocess.run([str(root/'venv/bin/python'),'-c',script,str(out/'train.json')],env=env,check=True,timeout=180)
        save(out/'specification.json',spec)
        environment = dict(CUDA_VISIBLE_DEVICES=str(gpu),AEROROZUM_DATA_ROOT=str(root/'artifacts'),
            AEROROZUM_LOCAL_MODEL_PATH=str(root/'models/gemma-4-12B-it'),AEROROZUM_GPU_MEMORY_FRACTION='.88',
            HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        command = ['systemd-run','--unit=aerorozum-expanded-visual-'+str(seed),
            '--property=RuntimeMaxSec=19800','--property=CPUQuota=300%','--property=MemoryMax=64G',
            '--property=StandardOutput=append:'+str(out/'worker.log'),
            '--property=StandardError=append:'+str(out/'worker.log')]
        command.extend('--setenv='+key+'='+value for key,value in environment.items())
        command.extend([str(root/'venv/bin/python'),'-u',str(worker),'worker',str(out/'specification.json')])
        subprocess.run(command,check=True,timeout=30); started=True
        print(json.dumps(dict(output=str(out),gpu=gpu,images=len(curriculum['images']),sft_steps=spec['sft_steps'],max_target_tokens=max(sizes))))
    finally:
        if not started:
            helper.lease(root,spec['lease'],remove=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/opt/aerorozum')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--seed',type=int,choices=[17,43,91],default=17)
    parser.add_argument('--optimized-monitor',action='store_true')
    args=parser.parse_args()
    launch(args.root,args.gpu,args.seed,args.optimized_monitor)
