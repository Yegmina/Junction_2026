"""Durable, bounded research DAG. Missing scientific components stay blocked.

No command provisions cloud machines or promotes experimental fields. Callback
workers must produce the declared artifact, which is independently validated
before the task can complete. Run with ``python -m aerorozum.orchestrator``.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import threading

from .common import config as load_config,data_root,digest,save,utcnow,redacted_error


@dataclass(frozen=True)
class TaskContext:
    """Callback(ctx) writes ctx.artifact and returns JSON metadata, not success."""
    task_id:str
    kind:str
    config:dict
    parameters:dict
    artifact:Path
    root:Path


def build_plan(cfg):
    nodes=[]
    def add(name,kind,dependencies=(),artifact=None,**parameters):
        heavy=kind in ('physics','compatibility','sft','grpo','policy_compatibility','policy_sft','policy_grpo','fast_generator','evaluation')
        gpu=2 if kind in ('compatibility','sft','grpo','policy_compatibility','policy_sft','policy_grpo') else 0
        memory=48 if gpu else 8 if heavy else 1
        nodes.append({'id':name,'kind':kind,'dependencies':list(dependencies),'artifact':artifact or name.replace('/','_')+'.json',
                      'parameters':parameters,'heavy':heavy,'api':kind in ('image_teacher','policy_teacher'),
                      'resources':{'memory_gb':memory,'gpu_slots':gpu},'status':'planned','attempts':0})
    add('preflight','preflight')
    add('audit','audit',('preflight',))
    add('freeze','freeze',('audit',))
    add('kernel_acceptance','kernel_acceptance',('freeze',))
    physics=[];counts=cfg['counts'];base=cfg['seed']
    for split,key in (('train','physical_pairs'),('validation','validation_pairs')):
        for i in range(counts[key]):
            for present in (True,False):
                identity=digest([split,i,present,base])[:24]
                name=f'physics/{identity}';physics.append(name)
                add(name,'physics',('kernel_acceptance',),artifact=f'physics/{identity}/field.npz',
                    split=split,pair_id=f'{split}-{i:03d}',targets_present=present,
                    seed=base+i+(10000 if split=='validation' else 0),physical_id=identity)
    for i in range(counts['direct_scenes']):
        group,layout=divmod(i,3);identity=digest([base,'direct_test',group,layout])[:24]
        name=f'physics/{identity}';physics.append(name)
        add(name,'physics',('kernel_acceptance',),artifact=f'physics/{identity}/field.npz',split='direct_test',
            layout=('A','relocated_B','target_free')[layout],seed=base+2000000+group,physical_id=identity)
    add('field_acceptance','field_acceptance',physics)
    add('fast_generator','fast_generator',('field_acceptance',))
    add('field_registry','field_registry',('fast_generator','field_acceptance'))
    add('generate','generate',('field_registry',),artifact='datasets/synthetic/manifest.json')
    add('training_manifest','training_manifest',('generate','audit'),artifact='datasets/combined/manifest.json')
    add('image_teacher','image_teacher',('generate',))
    add('policy_teacher','policy_teacher',('field_registry',))
    add('policy_manifest','policy_manifest',('policy_teacher',),artifact='datasets/policy/train.jsonl')
    add('compatibility','compatibility',('audit',),seed=counts['training_seeds'][0],stage='compatibility')
    add('policy_compatibility','policy_compatibility',('policy_manifest',),seed=counts['training_seeds'][0],stage='policy_compatibility')
    trained=[]
    for seed in counts['training_seeds']:
        for prefix,manifest,compat in (('detector','training_manifest','compatibility'),('policy','policy_manifest','policy_compatibility')):
            stage='sft' if prefix=='detector' else 'policy_sft';name=f'training/{prefix}/sft/{seed}'
            add(name,stage,(manifest,compat),seed=seed,stage=stage);trained.append(name)
            stage='grpo' if prefix=='detector' else 'policy_grpo';child=f'training/{prefix}/grpo/{seed}'
            add(child,stage,(name,),seed=seed,stage=stage);trained.append(child)
    for split in ('fast_test','direct_test'):
        add(f'evaluate/{split}','evaluation',tuple(trained)+('field_registry',),split=split)
    add('report','report',('evaluate/fast_test','evaluate/direct_test','image_teacher'))
    return {'schema_version':1,'config_hash':digest(cfg),'created_at':utcnow(),'status':'planned','tasks':nodes,
            'expected':{'physical_tasks':len(physics),'campaigns':sum(counts[k] for k in ('train_campaigns','validation_campaigns','fast_test_campaigns','direct_test_campaigns')),
                        'images':counts['images'],'training_seeds':counts['training_seeds']},
            'scientific_gaps':['No accepted production physics kernel/fields are inferred from upstream regression alone.',
                'A trained and validated fast field generator is not implemented by the image renderer.',
                'A field registry coordinator must build full campaign domains, paired counterfactuals and frozen observability certificates.',
                'Native trained-model campaign evaluation and the complete ablation runner require explicit implementations.']}


class Orchestrator:
    def __init__(self,cfg=None,root=None,callbacks=None,max_workers=2,memory_gb=None,gpu_slots=0,allow_heavy=False,allow_api=False):
        self.cfg=copy.deepcopy(cfg or load_config());self.root=Path(root or data_root()/'orchestration'/self.cfg['experiment']).resolve()
        self.root.mkdir(parents=True,exist_ok=True);self.path=self.root/'state.json';self.callbacks=callbacks or {}
        self.max_workers=int(max_workers);self.gpu_slots=int(gpu_slots);self.allow_heavy=allow_heavy;self.allow_api=allow_api
        available=os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')/1024**3
        self.memory_gb=min(float(memory_gb if memory_gb is not None else available*.8),available*.9)
        if self.max_workers<1 or self.memory_gb<=0 or self.gpu_slots<0:raise ValueError('Invalid resource limits')
        self._mutex=threading.RLock()
        self.state=json.loads(self.path.read_text()) if self.path.exists() else build_plan(self.cfg)
        if self.state['config_hash']!=digest(self.cfg):raise ValueError('Configuration changed; use a new orchestration root')
        self._save()

    def _save(self):
        with self._mutex:self.state['updated_at']=utcnow();save(self.path,self.state)

    @contextmanager
    def _lease(self):
        with (self.root/'.run.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another orchestrator is executing this run') from None
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def task(self,name):
        for task in self.state['tasks']:
            if task['id']==name:return task
        raise KeyError(f'Unknown step {name}')

    def artifact(self,task):
        path=(self.root/task['artifact']).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Task artifact escapes run directory')
        return path

    def _document(self,task):return json.loads(self.artifact(task).read_text())

    def _validate(self,task):
        path=self.artifact(task);kind=task['kind']
        if not path.is_file():return False,'Declared artifact is missing'
        try:
            if kind=='physics':
                import numpy as np
                with np.load(path,allow_pickle=False) as f:
                    shape=(len(f['times_s']),len(f['y_m']),len(f['x_m']))
                    if f['temperature_k'].shape!=shape or f['moisture'].shape!=shape:raise ValueError('Invalid field shape')
                    if not np.isfinite(f['temperature_k']).all() or not np.isfinite(f['moisture']).all():raise ValueError('Nonfinite physical field')
                meta=json.loads(path.with_suffix('.json').read_text());json.loads(path.with_suffix('.truth.json').read_text())
                if meta.get('backend') not in ('dumux','dumux_3.10.0','dumux_richards_energy'):raise ValueError('Expected actual DuMux field backend; Python demonstration is not interchangeable')
                return True,None
            if kind=='policy_manifest':
                from .training import load_policy_manifest
                load_policy_manifest(path,'policy_sft');return True,None
            d=self._document(task)
            if kind=='preflight':
                if not all(k in d for k in ('providers','physics','counts')):raise ValueError('Incomplete actual preflight snapshot')
            elif kind=='audit':
                if d.get('status')!='passed' or d.get('errors') or d.get('counts',{}).get('images',0)<1:raise ValueError('Dataset audit has not passed')
            elif kind=='freeze':
                expected=self.state['expected'];counts=self.cfg['counts']
                if d.get('config_hash')!=digest(self.cfg) or len(d.get('physics',[]))!=expected['physical_tasks'] or len(d.get('campaigns',[]))!=expected['campaigns']:raise ValueError('Frozen experiment count/hash mismatch')
                for split,key in (('train','train_campaigns'),('validation','validation_campaigns'),('fast_test','fast_test_campaigns'),('direct_test','direct_test_campaigns')):
                    if sum(s['split']==split for s in d['campaigns'])!=counts[key]:raise ValueError('Frozen split count mismatch')
            elif kind in ('kernel_acceptance','field_acceptance','fast_generator'):
                if d.get('research_accepted') is not True or d.get('status') not in ('passed','accepted'):raise ValueError('Explicit scientific acceptance artifact required')
                if d.get('config_hash')!=digest(self.cfg):raise ValueError('Acceptance is not bound to experiment configuration')
                evidence=d.get('evidence',[])
                if not evidence or not all(Path(p).is_file() for p in evidence):raise ValueError('Acceptance needs inspectable numerical evidence artifacts')
                if kind=='fast_generator' and not d.get('implementation'):raise ValueError('Fast generator implementation is missing')
            elif kind=='field_registry':
                from .generation import plan_generation
                result=plan_generation(self.artifact(self.task('freeze')),d)
                if result['status']!='ready':raise ValueError('; '.join(result['problems'][:3]))
            elif kind=='generate':
                rows=d.get('images',[])
                if d.get('research_accepted') is not True or d.get('status')!='completed' or len(rows)!=self.cfg['counts']['images']:raise ValueError('Exact accepted synthetic image count has not completed')
                for row in rows:
                    p=Path(row['image']);p=p if p.is_absolute() else data_root()/p
                    if not row.get('research_accepted') or not p.is_file():raise ValueError('Generated image/evidence missing')
            elif kind=='training_manifest':
                from .training import load_training_manifest
                rows,_=load_training_manifest(path,'sft')
                if not any(row.get('provenance') in ('synthetic_verified','direct_physics','validated_surrogate') for row in rows):raise ValueError('No verified synthetic training records')
            elif kind in ('compatibility','sft','grpo','policy_compatibility','policy_sft','policy_grpo'):
                steps=self.cfg['training']['compatibility_updates'] if 'compatibility' in kind else self.cfg['training'].get(kind+'_steps')
                if steps is None:raise ValueError('Research optimizer-update schedule is not configured')
                if d.get('status')!='completed' or d.get('stage')!=kind or d.get('seed')!=task['parameters']['seed'] or d.get('optimizer_updates',0)<steps:raise ValueError('Training optimizer/checkpoint protocol incomplete')
                if d.get('nonzero_gradient_checks',0)<1 or not d.get('image_influence',{}).get('passed'):raise ValueError('Verified image influence and nonzero gradients required')
                adapter=Path(d.get('adapter',''));adapter=adapter if adapter.is_absolute() else data_root()/adapter
                if not (adapter/'adapter_config.json').is_file():raise ValueError('Actual trained adapter missing')
            elif kind=='image_teacher':
                if d.get('status')!='completed' or d.get('completed',0)!=self.cfg['counts']['images'] or d.get('failed',0):raise ValueError('Image teacher did not complete every frozen image')
                if not d.get('run_artifact') or not Path(d['run_artifact']).is_file():raise ValueError('Teacher response evidence missing')
            elif kind=='policy_teacher':
                if d.get('status')!='completed' or d.get('completed_campaigns')!=self.cfg['counts']['train_campaigns'] or d.get('research_accepted') is not True:raise ValueError('Accepted teacher campaigns incomplete')
                for row in d.get('results',[]):
                    if row.get('status')!='completed' or not (Path(row['run_dir'])/'trajectory.jsonl').is_file():raise ValueError('Policy trajectory evidence missing')
                if len(d.get('results',[]))!=self.cfg['counts']['train_campaigns']:raise ValueError('Teacher run count differs from manifest')
            elif kind=='evaluation':
                split=task['parameters']['split'];n=self.cfg['counts'][split+'_campaigns']
                if d.get('status')!='completed' or d.get('research_accepted') is not True or d.get('campaigns')!=n or d.get('seeds')!=self.cfg['counts']['training_seeds'] or d.get('frontier_repetitions')!=self.cfg['counts']['frontier_repetitions']:raise ValueError('Full paired evaluation/seeds/repetitions are incomplete')
                if not d.get('records') or not all(Path(p).is_file() for p in d['records']):raise ValueError('Evaluation trajectories are missing')
            elif kind=='report':
                if d.get('status')!='completed' or not d.get('evaluations'):raise ValueError('Report lacks completed evaluations')
            else:raise ValueError('No artifact validator registered')
            return True,None
        except (OSError,ValueError,KeyError,TypeError) as exc:return False,str(exc)

    def _handler(self,task):
        return self.callbacks.get(task['id']) or self.callbacks.get(task['kind']) or getattr(self,'_do_'+task['kind'],None)

    def _reason(self,task):
        pending=[name for name in task['dependencies'] if self.task(name)['status']!='completed']
        if pending:return 'Prerequisites not complete: '+', '.join(pending[:4])+(' ...' if len(pending)>4 else '')
        if task['heavy'] and not self.allow_heavy:return 'Heavy execution disabled; no cloud machines are provisioned by this runner'
        if task['api'] and not self.allow_api:return 'API execution paused; --allow-api must be explicit'
        if task['api'] and not self.cfg.get('gemini',{}).get('project_quota_verified',False):return 'Full-volume Gemini quota is not verified'
        if task['resources']['memory_gb']>self.memory_gb or task['resources']['gpu_slots']>self.gpu_slots:return 'Task exceeds host memory/GPU limits'
        if task['kind'] in ('sft','grpo','policy_sft','policy_grpo') and not self.cfg['training'].get(task['kind']+'_steps'):return 'Research optimizer schedule is not configured'
        if not self._handler(task):
            reasons={'kernel_acceptance':'Awaiting external numerical acceptance; runner never self-accepts kernels',
                'physics':'No accepted DuMux worker callback; experimental Python is not a fallback',
                'field_acceptance':'Awaiting external refinement/conservation/counterfactual acceptance',
                'fast_generator':'Validated field interpolation/composition worker is not registered',
                'field_registry':'Accepted domain/counterfactual/certificate coordinator is not registered',
                'evaluation':'Trained-model campaign ablation evaluator is not registered'}
            return reasons.get(task['kind'],'No concrete worker registered')
        if task['heavy'] or task['api']:
            from .jobs import cost_status
            used=sum(row['spent_eur']+row['reserved_eur'] for row in cost_status() if row['provider']=='google_cloud')
            if used>=self.cfg.get('google_cloud_budget_eur',1000):return 'Google Cloud committed-cost cap reached'
        return None

    def _verify_unlocked(self):
        for task in self.state['tasks']:
            if task['status']=='running':task.update(status='planned',reason='Interrupted worker; validating artifacts')
            valid,reason=self._validate(task)
            if valid and all(self.task(d)['status']=='completed' for d in task['dependencies']):task.update(status='completed',reason=None,verified_at=utcnow())
            elif task['status']=='completed':task.update(status='blocked',reason='Artifact verification failed: '+str(reason))
        for task in self.state['tasks']:
            if task['status']!='completed':
                reason=self._reason(task)
                if reason:task.update(status='blocked',reason=reason)
                elif task['status']!='failed':task.update(status='planned',reason=None)
        self._save()

    def verify_resume(self):
        with self._lease():self._verify_unlocked()
        return self.summary()

    def _execute(self,task):
        with self._mutex:
            task.update(status='running',reason=None,started_at=utcnow(),attempts=task['attempts']+1);self._save()
        ctx=TaskContext(task['id'],task['kind'],copy.deepcopy(self.cfg),copy.deepcopy(task['parameters']),self.artifact(task),self.root)
        ctx.artifact.parent.mkdir(parents=True,exist_ok=True)
        try:
            result=self._handler(task)(ctx)
            if asyncio.iscoroutine(result):result=asyncio.run(result)
            valid,reason=self._validate(task)
            if not valid:raise ValueError('Worker artifact failed validation: '+str(reason))
            with self._mutex:task.update(status='completed',finished_at=utcnow(),reason=None,result=result if isinstance(result,dict) else None)
        except Exception as exc:
            with self._mutex:task.update(status='failed',finished_at=utcnow(),reason=redacted_error(exc))
        self._save()

    def run_step(self,name):
        with self._lease():
            self._verify_unlocked();task=self.task(name)
            if task['status']=='completed':return copy.deepcopy(task)
            reason=self._reason(task)
            if reason:task.update(status='blocked',reason=reason);self._save()
            else:self._execute(task)
            return copy.deepcopy(task)

    def run(self):
        with self._lease(),ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            self._verify_unlocked();active={};attempted=set()
            while True:
                used_memory=sum(t['resources']['memory_gb'] for t in active.values());used_gpus=sum(t['resources']['gpu_slots'] for t in active.values())
                for task in self.state['tasks']:
                    if task['status'] in ('completed','running','failed') or task['id'] in attempted:continue
                    reason=self._reason(task)
                    if reason:task.update(status='blocked',reason=reason);continue
                    if len(active)>=self.max_workers:break
                    if used_memory+task['resources']['memory_gb']>self.memory_gb or used_gpus+task['resources']['gpu_slots']>self.gpu_slots:continue
                    task.update(status='running',reason=None);attempted.add(task['id'])
                    active[pool.submit(self._execute,task)]=task
                    used_memory+=task['resources']['memory_gb'];used_gpus+=task['resources']['gpu_slots']
                self._save()
                if not active:break
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:future.result();active.pop(future)
            self.state['status']='completed' if all(t['status']=='completed' for t in self.state['tasks']) else 'blocked';self._save()
        return self.summary()

    def summary(self):
        counts={status:sum(t['status']==status for t in self.state['tasks']) for status in ('planned','blocked','running','completed','failed')}
        return {'status':self.state['status'],'state':str(self.path),'counts':counts,'expected':self.state['expected'],
                'blocked':[{'id':t['id'],'reason':t.get('reason')} for t in self.state['tasks'] if t['status'] in ('blocked','failed')],
                'resource_limits':{'max_workers':self.max_workers,'memory_gb':self.memory_gb,'gpu_slots':self.gpu_slots}}

    def _do_preflight(self,ctx):
        from .preflight import snapshot
        save(ctx.artifact,snapshot(check_models=False));return {'network_model_checks':False}

    def _do_audit(self,ctx):
        from .dataset import audit
        result=audit(self.cfg['dataset_zip'],export=True);save(ctx.artifact,result)
        return {'images':result['counts'].get('images',0)}

    def _do_freeze(self,ctx):
        from .experiment import freeze
        result=freeze();save(ctx.artifact,result)
        return {'campaigns':len(result['campaigns']),'physics':len(result['physics'])}

    def _do_generate(self,ctx):
        from .generation import generate_dataset
        result=generate_dataset(self.artifact(self.task('freeze')),self.artifact(self.task('field_registry')),output_dir=ctx.artifact.parent,cfg=self.cfg)
        if result['status']!='completed':raise ValueError('; '.join(result.get('problems',[])[:3]))
        return result

    def _do_training_manifest(self,ctx):
        synthetic=self._document(self.task('generate'));real=json.loads((data_root()/'datasets/real/manifest.json').read_text())
        rows=copy.deepcopy(real['images']);base=data_root().resolve()
        for source in synthetic['images']:
            row=copy.deepcopy(source);path=Path(row['image']);path=(path if path.is_absolute() else base/path).resolve()
            if not path.is_relative_to(base):raise ValueError('Generated image is outside artifact root')
            if not row.get('research_accepted'):raise ValueError('Unaccepted synthetic label')
            row.update(image=str(path.relative_to(base)),sha256=row.get('image_sha256'),source_provenance=row['provenance'])
            if row['provenance']=='accepted_physics_counterfactual':row['provenance']='synthetic_verified'
            rows.append(row)
        save(ctx.artifact,{'metadata':{'status':'passed','research_accepted':True},'images':rows})
        return {'images':len(rows),'synthetic_images':len(synthetic['images'])}

    def _do_image_teacher(self,ctx):
        from .providers import analyze_manifest
        result=asyncio.run(analyze_manifest(self.artifact(self.task('generate')),model=self.cfg['gemini']['bulk_model']))
        result={**result,'run_artifact':str(data_root()/'jobs'/(result['id']+'.json'))};save(ctx.artifact,result)
        return {'completed':result.get('completed',0)}

    def _do_policy_teacher(self,ctx):
        from .campaigns import run_campaign_batch
        result=asyncio.run(run_campaign_batch(self.artifact(self.task('freeze')),self.artifact(self.task('field_registry')),cfg=self.cfg,run_root=self.root/'teacher-campaigns',split='train'))
        save(ctx.artifact,result);return {'completed_campaigns':result.get('completed_campaigns',0)}

    def _do_policy_manifest(self,ctx):
        from .campaigns import export_policy_trajectories
        teacher=self._document(self.task('policy_teacher'))
        return export_policy_trajectories([row['run_dir'] for row in teacher['results']],ctx.artifact,split='train')

    def _training(self,ctx):
        from .training import train
        stage=ctx.parameters['stage'];seed=ctx.parameters['seed']
        if stage=='compatibility':manifest=data_root()/'datasets/real/manifest.json'
        elif stage.startswith('policy_'):manifest=self.artifact(self.task('policy_manifest'))
        else:manifest=self.artifact(self.task('training_manifest'))
        steps=self.cfg['training']['compatibility_updates'] if 'compatibility' in stage else self.cfg['training'][stage+'_steps']
        result=train(manifest,stage,seed,steps,self.cfg['primary_model']);save(ctx.artifact,result)
        return {'optimizer_updates':result['optimizer_updates']}

    _do_compatibility=_training
    _do_sft=_training
    _do_grpo=_training
    _do_policy_compatibility=_training
    _do_policy_sft=_training
    _do_policy_grpo=_training

    def _do_report(self,ctx):
        evaluations=[self._document(self.task('evaluate/'+split)) for split in ('fast_test','direct_test')]
        save(ctx.artifact,{'status':'completed','created_at':utcnow(),'config_hash':digest(self.cfg),'evaluations':evaluations,
            'counts':self.cfg['counts'],'interpretation':'Simulation evaluation; physical validation and real mine confirmation are separate claims'})
        return {'evaluations':2}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','run','run-step','verify'))
    parser.add_argument('--step');parser.add_argument('--config');parser.add_argument('--root')
    parser.add_argument('--max-workers',type=int,default=2);parser.add_argument('--memory-gb',type=float)
    parser.add_argument('--gpu-slots',type=int,default=0);parser.add_argument('--allow-heavy',action='store_true')
    parser.add_argument('--allow-api',action='store_true');parser.add_argument('--callbacks-module')
    args=parser.parse_args();callbacks={}
    if args.callbacks_module:
        import importlib
        callbacks=importlib.import_module(args.callbacks_module).CALLBACKS
    runner=Orchestrator(load_config(args.config),args.root,callbacks,args.max_workers,args.memory_gb,args.gpu_slots,args.allow_heavy,args.allow_api)
    if args.command=='run':result=runner.run()
    elif args.command=='verify':result=runner.verify_resume()
    elif args.command=='run-step':
        if not args.step:parser.error('--step required for run-step')
        result=runner.run_step(args.step)
    else:result=runner.summary()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
