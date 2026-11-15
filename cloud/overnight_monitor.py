"""Publish measured overnight status and preserve a compact resource history."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/opt/aerorozum')
ART=ROOT/'artifacts'
STUDY=ART/'overnight-20261115T000942'
PUBLIC=ROOT/'deployment/releases/coordinate-agent-v3/docs/presentation/research-overnight'


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {} if default is None else default


def cpu_times():
    fields=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(fields),fields[3]+fields[4]


def sample(previous):
    from aerorozum.common import save,utcnow
    total,idle=cpu_times();difference=total-previous[0]
    cpu=100*(1-(idle-previous[1])/difference) if difference else None
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    gpus=[]
    for line in output.splitlines():
        index,util,used,size,power=[v.strip() for v in line.split(',')]
        gpus.append(dict(id=int(index),utilization_percent=float(util),memory_used_mib=float(used),memory_total_mib=float(size),power_w=float(power)))
    reasoning=read(STUDY/'robotics-reasoning-v1/progress.json')
    high=read(ROOT/'airborne-labeler/artifacts/labelling/robotics-er2-high-thinking-v1/progress.json')
    declared=read(ROOT/'airborne-labeler/artifacts/labelling/robotics-er2-high-thinking-v1/frozen-inputs.json')
    training=[]
    for family in ['postsubmission-visual-replication-v1','postsubmission-expanded-visual-v1']:
        for spec_path in (ART/family).glob('*/specification.json'):
            spec=read(spec_path);folder=spec_path.parent;state=read(folder/'status.json')
            phases=[]
            for name in ['initial','continuation']:
                status=read(folder/name/'status.json');result=read(folder/name/'result.json')
                if status or result:
                    evaluations={k:v for k,v in result.get('results',{}).items() if k.endswith('_evaluation')}
                    phases.append(dict(phase=name,status=status,evaluations={k:{n:v.get(n) for n in ['micro_f2','tp','fp','fn','valid_json_fraction','negative_false_alarm_rate']} for k,v in evaluations.items()}))
            namespace=spec['namespace']
            metrics=[]
            for file in (ART/'training'/namespace).glob('**/metrics.jsonl'):
                try:
                    lines=file.read_text().splitlines();last=json.loads(lines[-1]) if lines else {}
                    metrics.append(dict(stage=file.parent.parent.name,last=last))
                except (OSError,ValueError):pass
            row=dict(id=family+'-'+str(spec['seed']),seed=spec['seed'],gpu_ids=spec['lease']['gpu_ids'],
                state=state,phases=phases,metrics=metrics,sft_steps=spec['sft_steps'],grpo_steps=spec['grpo_steps'])
            training.append(row)
            detail='; '.join(p['phase']+': '+p['status'].get('stage',p['status'].get('status','unknown')) for p in phases)
            save(ART/'jobs'/(row['id']+'.json'),dict(id=row['id'],kind='visual_training_replication',name=row['id'],
                model='Gemma 4 12B',status=state.get('status','unknown'),updated_at=utcnow(),
                detail=detail+'; development only; automatic promotion disabled',research_result=True))
    native=[]
    for path in (ROOT/'harbor/jobs').glob('overnight-*/result.json'):
        value=read(path);stats=value.get('stats',{})
        if not value.get('finished_at') or stats.get('n_running_trials') or stats.get('n_pending_trials'):continue
        scores=[m['mean'] for e in stats.get('evals',{}).values() for m in e.get('metrics',[]) if isinstance(m.get('mean'),(int,float))]
        native.append(dict(job=path.parent.name,finished_at=value['finished_at'],rewards=scores,
                           errors=stats.get('n_errored_trials',0),cost_usd=value.get('cost_usd')))
    physics=read(ART/'pipeline/physics.json')
    physical=[{k:r.get(k) for k in ['case_id','status','progress','remaining_wall_s_estimate','virtual_time_s','duration_s','research_accepted','linear_backend']}
              for r in physics.get('cases',[]) if r.get('case_id') in ['object-h4-local','object-h4-domain2']]
    receipt=dict(at=utcnow(),cpu_utilization_percent=cpu,host_cores=48,core_quota=43,
        disk_free_gib=shutil.disk_usage(ROOT).free/1024**3,gpus=gpus,reasoning=reasoning,
        high_thinking=dict(progress=high,available_unique_frames=declared.get('unique_eligible_available'),
                           missing_source_pixels=len(declared.get('missing',[]))),
        training=training,native=native,physics=physical,accepted_physical_pairs=physics.get('accepted_pairs'),
        submission_unchanged=True,opened_test_reused_for_selection=False,
        interpretation='Development research. Model labels are not human-confirmed physical mine locations.')
    save(STUDY/'live-status.json',receipt)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    public_temporary=PUBLIC/'.status-public.json'
    save(public_temporary,receipt);public_temporary.chmod(0o644)
    public_temporary.replace(PUBLIC/'status.json')
    with (STUDY/'resource-history.jsonl').open('a') as stream:
        stream.write(json.dumps({k:receipt[k] for k in ['at','cpu_utilization_percent','disk_free_gib','gpus']})+'\n')
    save(ART/'jobs/robotics-reasoning-study-v1.json',dict(id='robotics-reasoning-study-v1',
        kind='paired_prompt_reasoning_evaluation',name='Robotics prompt × reasoning × repeat',status=read(STUDY/'robotics-reasoning-v1/report.json').get('status','running') if reasoning.get('completed')==2160 else 'running',
        completed=reasoning.get('completed',0),total=2160,updated_at=utcnow(),
        detail='Two prompts × three native reasoning levels × three repeats × 120 development images. Paid raw logs archived; test excluded.'))
    return total,idle


if __name__=='__main__':
    previous=cpu_times();time.sleep(1)
    while True:
        try:previous=sample(previous)
        except Exception as error:print(type(error).__name__+': '+str(error)[:800],flush=True)
        time.sleep(20)
