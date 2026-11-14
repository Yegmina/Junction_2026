"""Auditable Gemma vision LoRA SFT/GRPO training; heavy dependencies load on demand."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
from typing import Any
from .common import config, data_root, digest, save, utcnow, redacted_error
from .jobs import job, update

PROMPT = ('Inspect this thermal image for provisional mine-like ground signatures. '
          'Return ONLY JSON {"detections":[{"bbox":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000}}]}. '
          'Boxes must tightly bound visible evidence, normalized to 0..1000. '
          'Return an empty detections list when no supported candidate is visible. '
          'Thermal appearance does not confirm a physical mine.')
APPROVED_REAL = {'original_manual', 'reviewed_ai'}
APPROVED_SYNTHETIC = {'synthetic_verified','direct_physics','validated_surrogate'}
STAGES = {'compatibility','sft','grpo','policy_compatibility','policy_sft','policy_grpo','real_sft','real_grpo','stream_sft','stream_grpo'}
BOX_KEYS = ('x_min','y_min','x_max','y_max')


def checked_box(value: dict) -> dict:
    if not isinstance(value,dict) or any(k not in value for k in BOX_KEYS): raise ValueError('Missing named box coordinate')
    box = {k:float(value[k]) for k in BOX_KEYS}
    if not all(math.isfinite(v) for v in box.values()): raise ValueError('Non-finite box coordinate')
    if not (0 <= box['x_min'] < box['x_max'] <= 1000 and 0 <= box['y_min'] < box['y_max'] <= 1000):
        raise ValueError('Invalid normalized box bounds')
    return box


def iou(a: dict, b: dict) -> float:
    a,b=checked_box(a),checked_box(b)
    intersection=max(0,min(a['x_max'],b['x_max'])-max(a['x_min'],b['x_min']))*max(0,min(a['y_max'],b['y_max'])-max(a['y_min'],b['y_min']))
    area=lambda x:(x['x_max']-x['x_min'])*(x['y_max']-x['y_min'])
    return intersection/max(1e-12,area(a)+area(b)-intersection)


def parse_prediction(completion: Any) -> list[dict]:
    if isinstance(completion,list):
        completion=completion[-1].get('content','') if completion else ''
        if isinstance(completion,list): completion=''.join(x.get('text','') for x in completion if isinstance(x,dict))
    if not isinstance(completion,str): raise ValueError('Completion must be text')
    text=completion.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    value=json.loads(text)
    if not isinstance(value,dict) or not isinstance(value.get('detections'),list): raise ValueError('Missing detections array')
    if len(value['detections'])>100: raise ValueError('Excessive predictions')
    return [checked_box(d['bbox']) for d in value['detections']]


def detection_score(predicted: list[dict], gold: list[dict], threshold=.5) -> dict:
    """Maximum-cardinality matching; duplicate/oversized boxes cannot add credit."""
    predicted=[checked_box(b.get('bbox',b)) for b in predicted]
    gold=[checked_box(b.get('bbox',b)) for b in gold]
    edges={i:sorted([j for j,g in enumerate(gold) if iou(p,g)>=threshold],key=lambda j:iou(p,gold[j]),reverse=True) for i,p in enumerate(predicted)}
    matched={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen: continue
            seen.add(j)
            if j not in matched or augment(matched[j],seen): matched[j]=i;return True
        return False
    for i in range(len(predicted)): augment(i,set())
    tp=len(matched);fp=len(predicted)-tp;fn=len(gold)-tp
    f2=5*tp/(5*tp+4*fn+fp) if gold else float(not predicted)
    return dict(f2=f2,tp=tp,fp=fp,fn=fn,matched_iou=sum(iou(predicted[i],gold[j]) for j,i in matched.items())/max(1,tp))


def detection_reward(completions, solution=None, **kwargs):
    """Verifiable TRL reward; grammar, verbosity and confidence earn no points."""
    solutions=solution if solution is not None else kwargs.get('gold_boxes')
    if solutions is None or len(solutions)!=len(completions): raise ValueError('Missing aligned ground truth')
    rewards=[]
    for completion, target in zip(completions,solutions):
        try:
            target=json.loads(target) if isinstance(target,str) else target
            rewards.append(float(detection_score(parse_prediction(completion),target)['f2']))
        except (ValueError,KeyError,TypeError,OverflowError): rewards.append(-1.)
    return rewards

def dense_detection_reward(completions, solution=None, **kwargs):
    """Continuous one-to-one IoU credit for explicitly versioned stream batches."""
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    solutions=solution if solution is not None else kwargs.get('gold_boxes')
    if solutions is None or len(solutions)!=len(completions):raise ValueError('Missing aligned ground truth')
    rewards=[]
    for completion,target in zip(completions,solutions):
        try:
            predicted=parse_prediction(completion);target=json.loads(target) if isinstance(target,str) else target
            gold=[checked_box(b.get('bbox',b)) for b in target]
            if not gold:rewards.append(float(not predicted));continue
            if not predicted:rewards.append(0.);continue
            overlaps=np.asarray([[iou(p,g) for g in gold] for p in predicted]);rows,cols=linear_sum_assignment(-overlaps)
            rewards.append(float(5*overlaps[rows,cols].sum()/(4*len(gold)+len(predicted))))
        except (ValueError,KeyError,TypeError,OverflowError):rewards.append(-1.)
    return rewards


def load_training_manifest(manifest: str | Path, stage: str, root: str | Path | None=None) -> tuple[list[dict],dict]:
    if stage not in STAGES: raise ValueError('Unknown training stage')
    root=Path(root or data_root()).resolve();manifest=Path(manifest)
    document=json.loads(manifest.read_text());metadata=document.get('metadata',{})
    if metadata.get('status') in ('failed','rejected'): raise ValueError('Dataset audit failed')
    rows=document.get('images',[])
    if not isinstance(rows,list): raise ValueError('Manifest images must be an array')
    groups={}; selected=[];synthetic=0
    for row in rows:
        split=row.get('split');identity=row.get('sha256') or row.get('id') or row.get('image')
        if not identity: raise ValueError('Missing sample identity')
        if identity in groups and groups[identity]!=split: raise ValueError('Image identity crosses dataset splits')
        groups[identity]=split
        if row.get('zone') in ('zone8','zone9') and split!='test': raise ValueError('Held-out test zone leaked')
        if row.get('zone')=='zone7' and split!='validation': raise ValueError('Held-out validation zone leaked')
        if split!='train': continue
        provenance=row.get('provenance','unknown');fidelity=str(row.get('fidelity',metadata.get('fidelity','')))
        if any(x in fidelity.lower() for x in ('demo','analytic','unvalidated')): raise ValueError('Demonstration images cannot be training evidence')
        if provenance in APPROVED_REAL: pass
        elif provenance in APPROVED_SYNTHETIC:
            if not row.get('research_accepted',metadata.get('research_accepted',False)): raise ValueError('Synthetic physics has not passed acceptance')
            synthetic+=1
        else: raise ValueError(f'Unapproved training provenance: {provenance}')
        image=Path(row.get('image',''))
        if not row.get('image') or image.is_absolute(): raise ValueError('Images must use artifact-relative paths')
        image=(root/image).resolve()
        try: image.relative_to(root)
        except ValueError: raise ValueError('Image escapes artifact root') from None
        if not image.is_file(): raise ValueError(f'Missing training image: {row.get("id")}')
        if row.get('sha256') and hashlib.sha256(image.read_bytes()).hexdigest()!=row['sha256']: raise ValueError('Training image checksum mismatch')
        boxes=[checked_box(b.get('bbox',b)) for b in row.get('boxes',[])]
        selected.append({**row,'image_path':str(image),'gold_boxes':boxes})
    if not selected: raise ValueError('No approved training rows')
    if stage!='compatibility' and synthetic==0: raise ValueError('Research training requires accepted synthetic physics; use compatibility for real-only pilots')
    if stage=='compatibility': selected=selected[:16]
    return selected,dict(manifest_hash=digest(document),training_images=len(selected),synthetic_images=synthetic,provenance={p:sum(r['provenance']==p for r in selected) for p in sorted({r['provenance'] for r in selected})},stage=stage,research_result=stage not in ('compatibility','policy_compatibility'))


def load_stream_manifest(manifest,stage,root=None):
    """Immutable incremental batches; native labels and physical acceptance differ."""
    root=Path(root or data_root()).resolve();document=json.loads(Path(manifest).read_text());meta=document.get('metadata',{})
    if stage not in ('stream_sft','stream_grpo') or meta.get('frozen_stream_batch') is not True:raise ValueError('Frozen streaming batch required')
    if meta.get('namespace')!='incremental-stream-v1':raise ValueError('Incremental namespace must remain separate from full research protocol')
    sessions=document.get('session_splits',{});rows=[];identities=set();certificates={}
    for row in document.get('images',[]):
        if row.get('split')!='train':raise ValueError('Held-out image leaked into incremental training batch')
        if row.get('training_eligible') is False or row.get('cross_split_duplicate') is True:raise ValueError('Explicitly excluded source cannot train')
        zone=row.get('zone') or row.get('parent_zone') or row.get('source',{}).get('zone')
        if zone in ('zone7','zone8','zone9'):raise ValueError('Held-out manual zone leaked into stream')
        sha=row.get('sha256')
        if not isinstance(sha,str) or not re.fullmatch(r'[a-f0-9]{64}',sha) or sha in identities:raise ValueError('Missing/duplicate image identity')
        identities.add(sha);image=Path(row.get('image',''))
        if image.is_absolute() or '..' in image.parts:raise ValueError('Stream image must use contained artifact-relative path')
        image=(root/image).resolve()
        if not image.is_relative_to(root) or not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest()!=sha:raise ValueError('Stream image hash/path mismatch')
        boxes=[checked_box(x.get('bbox',x)) for x in row.get('boxes',[])];source=row.get('source',{});provenance=row.get('provenance')
        if provenance=='user_accepted_model':
            if row.get('model')!='gemini-robotics-er-2-preview' or row.get('user_threshold')!=.6 or row.get('human_reviewed') is not False:raise ValueError('Exact user-accepted Robotics provenance required')
            session=source.get('session_id')
            if session and sessions.get(session)!='train':raise ValueError('Native source session is not frozen as train')
            detections=row.get('detections')
            if not isinstance(detections,list) or row.get('uncertain_boxes') or row.get('training_excluded_reason'):raise ValueError('Uncertain model labels excluded from training')
            if detections:
                if len(boxes)!=len(detections) or any(d.get('confidence',0)<.6 or checked_box(d['bbox'])!=box for d,box in zip(detections,boxes)):raise ValueError('Accepted detection boxes disagree')
            elif boxes or row.get('zero_candidates') is not True:raise ValueError('Empty boxes require explicit zero-candidate model response')
        elif provenance in APPROVED_REAL:
            if zone not in ('zone1','zone2','zone3','zone4','zone5'):raise ValueError('Approved manual train-zone provenance required')
        elif provenance in APPROVED_SYNTHETIC:
            cert_hash=row.get('acceptance_certificate_sha256');cert_path=meta.get('accepted_pair_certificates',{}).get(cert_hash)
            if not row.get('research_accepted') or not cert_path:raise ValueError('Synthetic stream row lacks accepted numerical pair lineage')
            if cert_hash not in certificates:
                data=Path(cert_path).read_bytes()
                if hashlib.sha256(data).hexdigest()!=cert_hash:raise ValueError('Accepted pair certificate changed')
                cert=json.loads(data)
                if cert.get('accepted') is not True or cert.get('research_accepted') is not True or cert.get('status')!='numerically_accepted_pair':raise ValueError('Experimental synthetic fields cannot train')
                certificates[cert_hash]=cert
            if row.get('physical_field_sha256') not in (certificates[cert_hash].get('present_sha256'),certificates[cert_hash].get('absent_sha256')):raise ValueError('Synthetic field hash not in accepted pair')
        else:raise ValueError('Unsupported streaming provenance')
        rows.append({**row,'image_path':str(image),'gold_boxes':boxes})
    if not rows or not any(r['gold_boxes'] for r in rows):raise ValueError('Incremental batch has no positive evidence')
    reward_protocol=meta.get('reward_protocol','f2-threshold-v1')
    if reward_protocol not in ('f2-threshold-v1','soft-iou-f2-v2'):raise ValueError('Unknown frozen stream reward protocol')
    return rows,{'manifest_hash':digest(document),'stage':stage,'training_images':len(rows),'namespace':'incremental-stream-v1','reward_protocol':reward_protocol,
        'provenance':dict((p,sum(r['provenance']==p for r in rows)) for p in sorted({r['provenance'] for r in rows})),
        'research_result':True,'physics_acceptance_claimed':bool(certificates),'human_reviewed':False,'batch_id':meta.get('batch_id')}


def language_lora_targets(model) -> list[str]:
    names=[]
    for name,module in model.named_modules():
        low=name.lower()
        if name.endswith(('.q_proj','.v_proj')) and 'self_attn' in low and not any(x in low for x in ('vision','audio','projector','embedder')):
            names.append(name)
    if not names: raise ValueError('No language-attention LoRA targets found; refusing broad fallback')
    return names


def build_dataset(rows: list[dict], stage: str):
    from datasets import Dataset, Image as DatasetImage
    examples=[]
    for row in rows:
        prompt=[{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT}]}]
        item=dict(prompt=prompt,image=row['image_path'])
        if stage in ('grpo','real_grpo','stream_grpo'): item['solution']=json.dumps(row['gold_boxes'])
        else: item['completion']=[{'role':'assistant','content':[{'type':'text','text':json.dumps({'detections':[{'bbox':b} for b in row['gold_boxes']]},separators=(',',':'))}]}]
        examples.append(item)
    return Dataset.from_list(examples).cast_column('image',DatasetImage())


def image_influence_probe(model, processor, image_path):
    import torch
    from PIL import Image
    original=Image.open(image_path).convert('RGB');blank=Image.new('RGB',original.size,(127,127,127))
    outputs=[];was_training=model.training;model.eval()
    try:
        for image in (original,blank):
            messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
            inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
            device=model.get_input_embeddings().weight.device
            inputs={k:(v.to(device=device,dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(device)) if torch.is_tensor(v) else v for k,v in inputs.items()}
            with torch.no_grad(): outputs.append(model(**inputs,logits_to_keep=1).logits[:,-1,:].float().cpu())
        difference=float((outputs[0]-outputs[1]).abs().mean())
        if not math.isfinite(difference) or difference<=1e-8: raise ValueError('Image influence probe failed')
        return {'mean_absolute_logit_change':difference,'passed':True}
    finally: model.train(was_training)


def train(manifest, stage='compatibility', seed=17, steps=10, model=None):
    if stage not in STAGES: raise ValueError('Unknown training stage')
    if not isinstance(steps,int) or steps<1: raise ValueError('Training steps must be a positive integer')
    settings=config();model_name=model or settings['primary_model']
    if model_name not in (settings['primary_model'],settings['reference_model']): raise ValueError('Only explicitly configured Gemma models may train; no small-model fallback')
    if stage in ('compatibility','policy_compatibility') and steps!=settings['training']['compatibility_updates']: raise ValueError('Compatibility protocol requires exactly ten optimizer updates')
    policy_stage=stage.startswith('policy_');real_stage=stage in ('real_sft','real_grpo');stream_stage=stage in ('stream_sft','stream_grpo')
    if policy_stage:rows,protocol=load_policy_manifest(manifest,stage)
    elif stream_stage:rows,protocol=load_stream_manifest(manifest,stage)
    elif real_stage:rows,protocol=load_real_transfer_manifest(manifest,stage,seed=seed)
    else:rows,protocol=load_training_manifest(manifest,stage)
    memory_probe=os.environ.get('AEROROZUM_MEMORY_PROBE')=='1'
    if memory_probe:protocol.update(research_result=False,scope='GPU memory compatibility probe only')
    record=job('training',stage=stage,seed=seed,model=model_name,completed=0,total=steps,research_result=not memory_probe and stage not in ('compatibility','policy_compatibility'))
    slug=re.sub(r'[^a-zA-Z0-9._-]','_',model_name)
    namespace=os.environ.get('AEROROZUM_TRAINING_NAMESPACE','')
    if namespace and (Path(namespace).is_absolute() or '..' in Path(namespace).parts):raise ValueError('Unsafe training namespace')
    training_root=data_root()/'training'/(namespace or ('real-transfer-machine-labels-v1' if real_stage else ''))
    output=training_root/slug/stage/f'seed-{seed}';output.mkdir(parents=True,exist_ok=True)
    prior=output/'protocol.json'
    if prior.exists() and json.loads(prior.read_text()).get('manifest_hash')!=protocol['manifest_hash']:
        update(record,'failed',error='Refusing checkpoint recovery with a different dataset');raise ValueError('Checkpoint dataset differs')
    completed=output/'summary.json'
    if completed.is_file():
        previous=json.loads(completed.read_text())
        same=(previous.get('status')=='completed' and previous.get('manifest_hash')==protocol['manifest_hash']
            and previous.get('model')==model_name and previous.get('seed')==seed and previous.get('stage')==stage
            and (previous.get('optimizer_updates',0)>=steps or (previous.get('early_stopped') is True and previous.get('requested_optimizer_updates')==steps))
            and previous.get('nonzero_gradient_checks',0)>0)
        adapter=output/'adapter'
        if same and (adapter/'adapter_config.json').is_file() and (adapter/'adapter_model.safetensors').is_file():
            update(record,'completed',completed=previous['optimizer_updates'],summary=previous);return previous
    save(prior,{**protocol,'model':model_name,'seed':seed,'steps':steps,'created_at':utcnow()})
    try:
        parent=None
        if stage in ('grpo','policy_grpo','real_grpo','stream_grpo'):
            parent_stage='policy_sft' if policy_stage else 'real_sft' if real_stage else 'stream_sft' if stream_stage else 'sft'
            parent=Path(os.environ.get('AEROROZUM_SFT_CHECKPOINT',training_root/slug/parent_stage/f'seed-{seed}'/'adapter'))
            validate_parent_adapter(parent,parent_stage,model_name,seed,protocol['manifest_hash'])
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, TrainerCallback, set_seed
        from transformers.trainer_utils import get_last_checkpoint
        from peft import LoraConfig, get_peft_model, PeftModel
        from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU is required; training has not started')
        set_seed(seed);torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')));torch.cuda.reset_peak_memory_stats()
        memory_fraction=os.environ.get('AEROROZUM_GPU_MEMORY_FRACTION')
        if memory_fraction:
            fraction=float(memory_fraction)
            if not 0<fraction<=.9:raise ValueError('GPU allocator must preserve10% headroom')
            torch.cuda.set_per_process_memory_fraction(fraction)
        local_source=os.environ.get('AEROROZUM_LOCAL_MODEL_PATH')
        if local_source:
            local_config=json.loads((Path(local_source)/'config.json').read_text())
            if local_config.get('model_type')!='gemma4_unified' or 'Gemma4UnifiedForConditionalGeneration' not in local_config.get('architectures',[]):
                raise ValueError('Local model must be the expected Gemma4 Unified architecture')
            source=local_source
            processor=AutoProcessor.from_pretrained(source)
        else:
            source=model_name
            processor=AutoProcessor.from_pretrained(source)
        placement=os.environ.get('AEROROZUM_DEVICE_MAP')
        if placement and placement not in ('balanced','auto','balanced_low_0','sequential'):raise ValueError('Unsupported model placement')
        load_kwargs={'device_map':placement} if placement else {}
        base=AutoModelForImageTextToText.from_pretrained(source,dtype=torch.bfloat16,attn_implementation=os.environ.get('AEROROZUM_ATTN_IMPL','sdpa'),low_cpu_mem_usage=True,**load_kwargs)
        if not placement:base.to(torch.device('cuda',torch.cuda.current_device()))
        base.config.use_cache=False
        warm=os.environ.get('AEROROZUM_WARMSTART_ADAPTER') if stage=='stream_sft' else None
        if warm:
            previous=json.loads((Path(warm).parent/'summary.json').read_text())
            if previous.get('status')!='completed' or previous.get('model')!=model_name or previous.get('seed')!=seed:raise ValueError('Warm-start lineage does not match this model and seed')
            trained=PeftModel.from_pretrained(base,warm,is_trainable=True)
        elif parent is not None:
            trained=PeftModel.from_pretrained(base,parent,is_trainable=True)
        else:
            targets=language_lora_targets(base)
            trained=get_peft_model(base,LoraConfig(r=settings['training']['lora_rank'],lora_alpha=32,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
        trained.enable_input_require_grads()
        probe_image=next((r['image_paths'][0] for r in rows if r.get('image_paths')),None) if policy_stage else rows[0]['image_path']
        if not probe_image:raise ValueError('At least one image-bearing training example is required')
        probe=image_influence_probe(trained,processor,probe_image);save(output/'image_influence.json',probe)
        dataset=build_policy_dataset(rows,stage) if policy_stage else build_dataset(rows,stage)
        class Monitor(TrainerCallback):
            checks=0
            nonzero_checks=0
            zero_variance_steps=0
            early_stop_reason=None
            def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
                squared=0.;count=0
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        if not torch.isfinite(p.grad).all(): raise FloatingPointError('Non-finite training gradient')
                        squared+=float(p.grad.float().square().sum());count+=p.grad.numel()
                if not count or not math.isfinite(squared): raise FloatingPointError('No finite LoRA gradients')
                self.checks+=1
                self.nonzero_checks+=int(squared>0)
                save(output/'gradient_checks.json',{'checks':self.checks,'nonzero_checks':self.nonzero_checks,'last_gradient_l2':math.sqrt(squared),'finite':True})
            def on_log(self,args,state,control,logs=None,**kwargs):
                metrics={k:float(v) for k,v in (logs or {}).items() if isinstance(v,(int,float)) and math.isfinite(float(v))}
                if logs and 'loss' in logs and not math.isfinite(float(logs['loss'])): raise FloatingPointError('Non-finite training loss')
                with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':state.global_step,'at':utcnow(),**metrics})+'\n')
                if stage=='stream_grpo' and 'frac_reward_zero_std' in metrics:
                    self.zero_variance_steps=self.zero_variance_steps+1 if metrics['frac_reward_zero_std']>=1-1e-8 else 0
                    if self.zero_variance_steps>=4:
                        self.early_stop_reason='Four consecutive zero-variance task reward groups; do not extend KL-only updates'
                        control.should_training_stop=True
                update(record,completed=state.global_step,metrics=metrics)
        monitor=Monitor()
        micro=int(os.environ.get('AEROROZUM_MICROBATCH','1'));accum=int(os.environ.get('AEROROZUM_GRAD_ACCUMULATION','4'))
        if micro*accum!=4 or micro not in (1,2):raise ValueError('Preserve effective batch4 in measured topology')
        protocol.update(microbatch=micro,gradient_accumulation=accum,effective_batch=4,visible_gpu_ids=os.environ.get('CUDA_VISIBLE_DEVICES'),grpo_zero_variance_patience=4 if stage=='stream_grpo' else None)
        save(prior,{**protocol,'model':model_name,'seed':seed,'steps':steps,'created_at':utcnow()})
        common=dict(output_dir=str(output),max_steps=steps,per_device_train_batch_size=micro,gradient_accumulation_steps=accum,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,save_steps=max(1,min(25,steps//2)),save_total_limit=2,report_to='none',seed=seed,data_seed=seed,remove_unused_columns=False,dataloader_num_workers=0)
        if stage=='policy_grpo':
            args=GRPOConfig(**common,learning_rate=1e-6,num_generations=settings['training']['grpo_group_size'],max_completion_length=4096,max_tool_calling_iterations=settings['training']['decision_window_actions'],beta=.04,use_vllm=False)
            trainer=GRPOTrainer(model=trained,args=args,processing_class=processor,train_dataset=dataset,environment_factory=lambda:SurveyPolicyEnv(data_root()),callbacks=[monitor])
        elif stage in ('grpo','real_grpo','stream_grpo'):
            args=GRPOConfig(**common,learning_rate=1e-6,num_generations=settings['training']['grpo_group_size'],max_completion_length=384,beta=.04,use_vllm=False)
            reward=dense_detection_reward if stage=='stream_grpo' and protocol.get('reward_protocol')=='soft-iou-f2-v2' else detection_reward
            trainer=GRPOTrainer(model=trained,args=args,processing_class=processor,train_dataset=dataset,reward_funcs=reward,callbacks=[monitor])
        else:
            args=SFTConfig(**common,learning_rate=float(os.environ.get('AEROROZUM_SFT_LEARNING_RATE','2e-5' if stream_stage else '1e-4')),max_length=None,completion_only_loss=True,packing=False)
            trainer=SFTTrainer(model=trained,args=args,processing_class=processor,train_dataset=dataset,callbacks=[monitor])
        checkpoint=get_last_checkpoint(str(output))
        if checkpoint and (output/'gradient_checks.json').is_file():
            previous=json.loads((output/'gradient_checks.json').read_text());monitor.checks=previous.get('checks',0);monitor.nonzero_checks=previous.get('nonzero_checks',0)
        result=trainer.train(resume_from_checkpoint=checkpoint)
        if trainer.state.global_step<steps and not monitor.early_stop_reason: raise RuntimeError('Trainer stopped before requested optimizer updates')
        if monitor.checks<1 and not checkpoint: raise RuntimeError('No verified optimizer gradient checks recorded')
        if monitor.nonzero_checks<1 and not checkpoint: raise RuntimeError('No optimization signal: all observed gradients were zero')
        trainer.save_model(str(output/'adapter'));processor.save_pretrained(output/'adapter')
        if not (output/'adapter'/'adapter_config.json').exists(): raise RuntimeError('Adapter checkpoint was not saved')
        metrics={k:float(v) for k,v in result.metrics.items() if isinstance(v,(int,float)) and math.isfinite(float(v))}
        peaks={str(i):{'allocated_bytes':torch.cuda.max_memory_allocated(i),'reserved_bytes':torch.cuda.max_memory_reserved(i)} for i in range(torch.cuda.device_count())}
        summary={**protocol,'status':'completed','model':model_name,'seed':seed,'optimizer_updates':trainer.state.global_step,'requested_optimizer_updates':steps,'early_stopped':bool(monitor.early_stop_reason),'early_stop_reason':monitor.early_stop_reason,'gradient_checks':monitor.checks,'nonzero_gradient_checks':monitor.nonzero_checks,'recovered_from_checkpoint':bool(checkpoint),'peak_gpu_memory_bytes':max(x['allocated_bytes'] for x in peaks.values()),'peak_gpu_reserved_bytes':max(x['reserved_bytes'] for x in peaks.values()),'per_device_peak_memory':peaks,'image_influence':probe,'metrics':metrics,'adapter':str((output/'adapter').relative_to(data_root()))}
        save(output/'summary.json',summary);update(record,'completed',completed=trainer.state.global_step,summary=summary)
        return summary
    except Exception as ex:
        update(record,'failed',error=redacted_error(ex));save(output/'failure.json',{'status':'failed','error':redacted_error(ex),'at':utcnow(),'research_result':False});raise


def policy_conversations(records):
    """Supervised tool trajectories must not turn hidden evaluator data into prompts."""
    result=[]
    for row in records:
        if row.get('split')!='train':continue
        if not row.get('research_accepted'):raise ValueError('Policy trajectory lacks accepted physics')
        if any(k in row for k in ('truth','targets','evaluator_snapshot','hidden_state')):raise ValueError('Hidden state present in policy training record')
        if not isinstance(row.get('messages'),list) or not row.get('tools'):raise ValueError('Policy trajectories need messages and tool schemas')
        result.append({'messages':row['messages'],'tools':row['tools'],**({'images':row['images']} if 'images' in row else {})})
    if not result:raise ValueError('No accepted policy training trajectories')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--stage',choices=sorted(STAGES),default='compatibility');p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int,default=10);p.add_argument('--model');a=p.parse_args()
    print(json.dumps(train(a.manifest,a.stage,a.seed,a.steps,a.model),indent=2))


POLICY_TOOL = {'type':'function','function':{'name':'act','description':'Take one permitted survey action using public sensor observations. Eight actions maximum in a decision window.','parameters':{'type':'object','properties':{'action':{'type':'string','description':'One of launch, move_to, capture, inspect_observation, inspect_crop, wait_until, land, record_candidate, submit_report.'},'arguments_json':{'type':'string','description':'JSON object containing the documented arguments for this action.'}},'required':['action','arguments_json']}}}
POLICY_OVERRIDE = '\nFor this training interface, call the act tool with action and arguments_json. Do not print the action as ordinary JSON. Your decision window permits at most eight tool calls. Use only public evidence.'


def load_policy_manifest(manifest,stage,root=None):
    root=Path(root or data_root()).resolve()
    rows=[json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    selected=[];groups={}
    for row in rows:
        key=row.get('run_id')
        if not key:raise ValueError('Policy sample lacks run identity')
        split=row.get('split')
        if key in groups and groups[key]!=split:raise ValueError('Campaign identity crosses policy splits')
        groups[key]=split
        if split!='train':continue
        demo=stage=='policy_compatibility'
        if row.get('provenance') not in ({'gemini_policy_trajectory','demo_gemini_policy_trajectory'} if demo else {'gemini_policy_trajectory'}):
            raise ValueError('Unapproved policy trajectory provenance')
        if not demo and not row.get('research_accepted'):raise ValueError('Policy training requires accepted physical campaigns')
        for required in ('campaign_config','campaign_id','prefix_actions','seed','prompt','completion'):
            if required not in row:raise ValueError(f'Policy sample lacks {required}')
        if not isinstance(row['campaign_config'],dict) or not re.fullmatch(r'[a-f0-9]{24}',row['campaign_id']):raise ValueError('Invalid campaign restoration metadata')
        if any(term in row['prompt'] for term in ('"truth":','"evaluator_snapshot":','"qualified_cells":')):raise ValueError('Hidden evaluator state leaked into policy prompt')
        completion=json.loads(row['completion'])
        from .campaigns import ACTIONS
        if completion.get('action') not in ACTIONS or not isinstance(completion.get('arguments',{}),dict):raise ValueError('Invalid policy target action')
        image_paths=[]
        for raw in row.get('images',[]):
            path=Path(raw);path=(path if path.is_absolute() else root/path).resolve()
            if not path.is_relative_to(root) or not path.is_file():raise ValueError('Policy image is missing or outside artifact root')
            image_paths.append(str(path))
        selected.append({**row,'target_action':completion,'image_paths':image_paths})
    if stage=='policy_compatibility':selected=selected[:16]
    if not selected:raise ValueError('No approved policy training examples')
    return selected,{'manifest_hash':digest(rows),'training_examples':len(selected),'provenance':dict((p,sum(r['provenance']==p for r in selected)) for p in sorted({r['provenance'] for r in selected})),'stage':stage,'research_result':stage!='policy_compatibility','decision_window_actions':8}


def build_policy_dataset(rows,stage):
    from datasets import Dataset, Image as DatasetImage, Sequence
    examples=[]
    for row in rows:
        prompt=[{'role':'user','content':[{'type':'image'} for _ in row['image_paths']]+[{'type':'text','text':row['prompt']+POLICY_OVERRIDE}]}]
        item={'prompt':prompt,'images':row['image_paths']}
        if stage=='policy_grpo':
            item['rollout_state_json']=json.dumps({k:row[k] for k in ('campaign_config','campaign_id','seed','prefix_actions')},allow_nan=False)
        else:
            target=row['target_action']
            item['completion']=[{'role':'assistant','content':[],'tool_calls':[{'type':'function','function':{'name':'act','arguments':{'action':target['action'],'arguments_json':json.dumps(target.get('arguments',{}),separators=(',',':'))}}}]}]
            item['tools']=[POLICY_TOOL]
        examples.append(item)
    return Dataset.from_list(examples).cast_column('images',Sequence(DatasetImage()))


class SurveyPolicyEnv:
    """TRL environment with private terminal scoring and eight public actions."""
    def __init__(self,root=None,allow_demo=False):
        self._root=Path(root or data_root()).resolve();self._allow_demo=allow_demo
        self._campaign=None;self._calls=0;self._invalid=0;self._start_score=0.

    def reset(self,rollout_state_json: str,**kwargs) -> None:
        import secrets
        from .environment import Campaign
        state=json.loads(rollout_state_json)
        if not re.fullmatch(r'[a-f0-9]{24}',state['campaign_id']):raise ValueError('Invalid campaign ID')
        out=self._root/'training'/'policy_rollouts'/secrets.token_hex(10)
        self._campaign=Campaign(state['campaign_config'],seed=int(state['seed']),output_dir=out)
        if not self._campaign.status()['research_accepted'] and not self._allow_demo:raise ValueError('Policy GRPO requires accepted physical fields and certified observability')
        self._campaign.id=state['campaign_id'];self._campaign._dir=out/state['campaign_id'];self._campaign._dir.mkdir(parents=True,exist_ok=True)
        for entry in state['prefix_actions']:
            self._campaign.act(entry['action'],entry.get('arguments',{}))
        self._calls=0;self._invalid=0;self._start_score=self._potential()

    def _potential(self):
        from .evaluation import score_submission
        snapshot=self._campaign.evaluator_snapshot()
        result=score_submission(snapshot['status']['candidates'],{o['id']:o for o in snapshot['observations']},snapshot['observability']['evaluable_target_ids'],snapshot['coverage'],completed=True,observability_verified=snapshot['observability']['verified'])
        return float(result['reward'])

    def get_reward(self) -> float:
        # No truth-derived feedback is returned by act; training receives this once per rollout.
        return self._potential()-self._start_score-.05*self._invalid/max(1,self._calls)

    def act(self,action: str,arguments_json: str='{}') -> list:
        """Take a public survey action within the current eight-action decision window.

        Args:
            action: launch, move_to, capture, inspect_observation, inspect_crop, wait_until, land, record_candidate, or submit_report.
            arguments_json: A JSON object containing the action arguments described in the task.

        Returns:
            Public status and optional original camera image as content blocks. Hidden truth and scores are never returned.
        """
        from PIL import Image
        if self._campaign is None:return [{'type':'text','text':'Environment has not been reset'}]
        if self._calls>=8:return [{'type':'text','text':'Decision window exhausted; return your final response.'}]
        self._calls+=1
        try:
            arguments=json.loads(arguments_json)
            if not isinstance(arguments,dict):raise ValueError('Arguments must be a JSON object')
            result=self._campaign.act(action,arguments)
        except (ValueError,TypeError) as error:
            result={'ok':False,'error':str(error),'status':self._campaign.status()}
        self._invalid+=int(not result.get('ok',False))
        blocks=[{'type':'text','text':json.dumps(result,allow_nan=False)}]
        media=result.get('observation') or result.get('crop')
        if isinstance(media,dict) and media.get('image'):
            path=(self._campaign.output_dir/media['image']).resolve()
            if not path.is_relative_to(self._campaign.output_dir):raise ValueError('Observation path escaped rollout directory')
            with Image.open(path) as image:blocks.append({'type':'image','image':image.convert('RGB').copy()})
        return blocks




def load_real_transfer_manifest(manifest,stage,*,root=None,seed=17):
    """Separate real-transfer experiment under user-authorized machine labels."""
    from zipfile import ZipFile
    if stage not in ('real_sft','real_grpo'):raise ValueError('Unknown real-transfer stage')
    root=Path(root or data_root()).resolve();manifest=Path(manifest).resolve()
    document=json.loads(manifest.read_text());metadata=document.get('metadata',{})
    if metadata.get('status') in ('failed','rejected'):raise ValueError('Source label manifest failed')
    rows=document.get('images',[])
    if not isinstance(rows,list):raise ValueError('Source images must be a list')
    has_native=any(isinstance(r.get('source'),dict) and r['source'].get('session_id') for r in rows)
    frozen=document.get('session_splits') or metadata.get('session_splits') or {}
    if not frozen and has_native:
        explicit=metadata.get('session_splits_path');candidates=[Path(explicit)] if explicit else []
        candidates += [parent/'session-splits.json' for parent in manifest.parents if parent.is_relative_to(root)]
        candidates += [root/'labelling/robotics-er2-5fps/session-splits.json']
        found=next((p for p in candidates if p.is_file()),None)
        if not found:raise ValueError('Frozen whole-session split registry required')
        frozen=json.loads(found.read_text())
    frozen=frozen.get('sessions',frozen)
    if not isinstance(frozen,dict):raise ValueError('Invalid frozen session splits')
    zone_splits={**{f'zone{i}':'train' for i in (1,2,3,4,5)},'zone7':'validation','zone8':'test','zone9':'test'}
    excluded={'held_out':0,'uncertain':0,'duplicate':0,'cross_split_duplicate':0}
    held_hashes={r.get('sha256') for r in rows if r.get('split') in ('validation','test')}
    positives=[];negatives=[];seen=set();observed_groups={};archive_cache={};member_cache={}
    try:
        for row in rows:
            source=row.get('source') if isinstance(row.get('source'),dict) else {}
            split=row.get('split');session=source.get('session_id')
            if session:
                if session not in frozen or frozen[session]!=split:raise ValueError('Frame split differs from frozen recording-session split')
                group='video:'+session
                video_hash=source.get('video_sha256')
                if not isinstance(video_hash,str) or not re.fullmatch(r'[a-f0-9]{64}',video_hash):raise ValueError('Native video source hash required')
            else:
                archive_hash=source.get('source_archive_sha256') or row.get('source_archive_sha256') or metadata.get('source_archive_sha256')
                parent_hash=source.get('parent_image_sha256') or row.get('parent_image_sha256') or row.get('parent_sha256')
                member=source.get('archive_member') or row.get('archive_member') or row.get('parent_archive_member')
                zone=source.get('zone') or row.get('zone') or row.get('parent_zone')
                if not all(isinstance(x,str) and re.fullmatch(r'[a-f0-9]{64}',x) for x in (archive_hash,parent_hash)) or not member or '..' in Path(member).parts or Path(member).is_absolute():raise ValueError('Archive source SHA, parent-image SHA and safe archive_member required')
                member_zone=re.search(r'__(zone[1-9])__',Path(member).name)
                if member_zone and member_zone.group(1)!=zone:raise ValueError('Archive member zone differs from declared source zone')
                if zone not in zone_splits or zone_splits[zone]!=split:raise ValueError('Frame split differs from frozen archive-zone split')
                group=f'archive:{archive_hash}:{zone}'
                if split=='train':
                    source_path=Path(os.environ.get('AEROROZUM_SOURCE_ARCHIVE') or source.get('archive_path') or metadata.get('archive_path') or config()['dataset_zip'])
                    if archive_hash not in archive_cache:
                        if not source_path.is_file():raise ValueError('Original reviewed ZIP required; set AEROROZUM_SOURCE_ARCHIVE')
                        if hashlib.sha256(source_path.read_bytes()).hexdigest()!=archive_hash:raise ValueError('Source archive checksum mismatch')
                        archive_cache[archive_hash]=ZipFile(source_path)
                    key=(archive_hash,member)
                    if key not in member_cache:member_cache[key]=hashlib.sha256(archive_cache[archive_hash].read(member)).hexdigest()
                    if member_cache[key]!=parent_hash:raise ValueError('Parent image checksum differs from original archive member')
            if group in observed_groups and observed_groups[group]!=split:raise ValueError('Source background group crosses splits')
            observed_groups[group]=split
            if split!='train':excluded['held_out']+=1;continue
            if row.get('provenance')!='user_accepted_model' or row.get('model')!='gemini-robotics-er-2-preview' or row.get('user_threshold')!=.6 or row.get('human_reviewed') is not False:raise ValueError('Real transfer requires exact Robotics ER 2 labels accepted by user threshold 0.6; not human review')
            image_hash=row.get('sha256')
            if not isinstance(image_hash,str) or not re.fullmatch(r'[a-f0-9]{64}',image_hash):raise ValueError('Frame checksum required')
            if image_hash in held_hashes:excluded['cross_split_duplicate']+=1;continue
            if image_hash in seen:excluded['duplicate']+=1;continue
            image=Path(row.get('image',''));image=(image if image.is_absolute() else root/image).resolve()
            if not image.is_relative_to(root) or not image.is_file():raise ValueError('Real training frame missing or outside artifact root')
            if hashlib.sha256(image.read_bytes()).hexdigest()!=image_hash:raise ValueError('Real training frame checksum mismatch')
            detections=row.get('detections')
            if not isinstance(detections,list):raise ValueError('Original detection list required for class decisions')
            if row.get('model_output') is not None and row['model_output'].get('detections')!=detections:raise ValueError('Model output and cached detections disagree')
            for detection in detections:
                confidence=detection.get('confidence')
                if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid machine-label confidence')
                checked_box(detection.get('bbox'))
            if row.get('uncertain_boxes') or row.get('training_excluded_reason') or any(d['confidence']<.6 for d in detections):excluded['uncertain']+=1;continue
            boxes=[checked_box(b.get('bbox',b)) for b in row.get('boxes',[])]
            if detections:
                if len(boxes)!=len(detections) or any(checked_box(d['bbox'])!=b for d,b in zip(detections,boxes)):raise ValueError('Accepted boxes do not match all high-confidence detections')
                negative=False
            else:
                if row.get('zero_candidates') is not True or boxes:raise ValueError('Empty boxes alone are not an explicit zero-candidate model response')
                negative=True
            seen.add(image_hash);item={**row,'image_path':str(image),'gold_boxes':boxes,'source_group':group}
            (negatives if negative else positives).append(item)
    finally:
        for archive in archive_cache.values():archive.close()
    if len(positives)<32 or len(negatives)<32:raise ValueError(f'Real-transfer class gate: need at least 32 unique positive and 32 explicit zero-candidate training frames; found {len(positives)} positive / {len(negatives)} zero')
    rng=random.Random(int(seed));groups={}
    for row in negatives:groups.setdefault(row['source_group'],[]).append(row)
    for group in groups.values():rng.shuffle(group)
    selected_negative=[];maximum=min(len(negatives),2*len(positives))
    while len(selected_negative)<maximum:
        for group in sorted(groups):
            if groups[group] and len(selected_negative)<maximum:selected_negative.append(groups[group].pop())
    selected=positives+selected_negative;rng.shuffle(selected)
    protocol={'manifest_hash':digest(document),'stage':stage,'experiment_namespace':'real-transfer-machine-labels-v1',
        'training_images':len(selected),'positive_frames':len(positives),'zero_candidate_frames':len(selected_negative),
        'available_zero_candidate_frames':len(negatives),'negative_to_positive_cap':2,
        'balancing':'all unique positives; seeded source-group round robin negatives; no first-N boot-frame sampling',
        'source_split_hash':digest({'recording_sessions':frozen,'archive_zones':zone_splits}),'training_source_groups':sorted({r['source_group'] for r in selected}),
        'provenance':{'user_accepted_model':len(selected)},'model_labeler':'gemini-robotics-er-2-preview','user_threshold':.6,
        'human_reviewed':False,'independently_verified_identity':False,'physics_acceptance_claimed':False,'research_result':True,
        'interpretation':'Real-transfer study under user-authorized machine labels. Manual ZIP held-out-zone agreement is separate evidence; no physical mine accuracy is established.','excluded':excluded}
    return selected,protocol


def validate_parent_adapter(parent,stage,model,seed,manifest_hash):
    parent=Path(parent)
    if not (parent/'adapter_config.json').is_file():raise ValueError('GRPO requires the completed corresponding SFT adapter')
    summary_path=parent.parent/'summary.json'
    if not summary_path.is_file():raise ValueError('SFT completion summary required before GRPO')
    summary=json.loads(summary_path.read_text())
    expected={'status':'completed','stage':stage,'model':model,'seed':seed,'manifest_hash':manifest_hash}
    if any(summary.get(key)!=value for key,value in expected.items()):raise ValueError('GRPO parent does not match completed SFT stage/model/seed/frozen dataset')
    return summary




def freeze_real_transfer_manifest(sources,output,session_splits,*,seed=17,root=None):
    """Freeze a balanced early transfer batch from paid cached results, never infer.

    Sources can be label manifest JSON files or labels/results directories.
    The main synthetic/three-seed experiment manifest is never modified.
    """
    root=Path(root or data_root()).resolve();output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        document=json.loads(output.read_text())
        if not document.get('metadata',{}).get('frozen_real_transfer'):raise ValueError('Refusing to replace a different existing manifest')
        selected,protocol=load_real_transfer_manifest(output,'real_sft',root=root,seed=seed)
        return {'status':'already_frozen','manifest':str(output),'manifest_hash':protocol['manifest_hash'],'training_images':len(selected),'positive_frames':protocol['positive_frames'],'zero_candidate_frames':protocol['zero_candidate_frames']}
    if isinstance(session_splits,(str,Path)):session_splits=json.loads(Path(session_splits).read_text())
    session_splits=session_splits.get('sessions',session_splits)
    rows=[];origins=[]
    for source in sources:
        path=Path(source)
        if path.is_dir():
            directory=path/'results' if (path/'results').is_dir() else path
            files=sorted(directory.glob('*.json'))
        else:files=[path]
        origins.append({'source':str(path),'files':len(files)})
        for file in files:
            document=json.loads(file.read_text())
            if isinstance(document.get('images'),list):rows.extend(document['images'])
            elif document.get('image') and document.get('model') and document.get('id'):rows.append(document)
    identities={};unique=[]
    for row in rows:
        key=(row.get('id'),row.get('sha256'))
        if key in identities:
            previous=identities[key]
            if previous.get('split')!=row.get('split') or previous.get('detections')!=row.get('detections'):raise ValueError('Conflicting labels or splits for the same cached frame')
            continue
        identities[key]=row;unique.append(row)
    candidate=output.with_name('.'+output.name+'.candidate.json')
    interim={'metadata':{'status':'passed','scope':'early_real_transfer_candidate'},'session_splits':session_splits,'images':unique}
    save(candidate,interim)
    try:selected,protocol=load_real_transfer_manifest(candidate,'real_sft',root=root,seed=seed)
    finally:candidate.unlink(missing_ok=True)
    clean=[{k:v for k,v in row.items() if k not in ('image_path','gold_boxes','source_group')} for row in selected]
    document={'metadata':{'status':'passed','frozen_real_transfer':True,'created_at':utcnow(),'experiment_namespace':'real-transfer-machine-labels-v1','scope':'bounded early real-transfer experiment; main full three-seed physical protocol remains unchanged','seed':seed,'input_records':len(unique),'source_cache_locations':origins,'selection':protocol,'human_reviewed':False,'identity_verified':False},'session_splits':session_splits,'images':clean}
    save(output,document)
    _,frozen=load_real_transfer_manifest(output,'real_sft',root=root,seed=seed)
    return {'status':'frozen','manifest':str(output),'manifest_hash':frozen['manifest_hash'],'training_images':len(clean),'positive_frames':frozen['positive_frames'],'zero_candidate_frames':frozen['zero_candidate_frames']}


if __name__=='__main__':main()
