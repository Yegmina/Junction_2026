"""One frozen incremental pass: SFT, GRPO and label-blind development scoring."""
from __future__ import annotations
import argparse,gc,json,os,time,fcntl
from pathlib import Path
from .common import data_root,digest,save,utcnow

def reward_signal_diagnostics(path):
    """Separate relative task advantages from nonzero KL-only gradients."""
    rows=[]
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:continue
    measured=[r for r in rows if isinstance(r.get('frac_reward_zero_std'),(int,float))]
    informative=[r for r in measured if r['frac_reward_zero_std']<1-1e-8]
    return {'measured_steps':len(measured),'informative_steps':len(informative),'task_signal_verified':bool(informative),
        'mean_zero_reward_variance_fraction':sum(r['frac_reward_zero_std'] for r in measured)/len(measured) if measured else None,
        'interpretation':'Nonzero gradients alone do not establish a task-learning signal or an RL improvement'}

def checkpoint_utility(baseline,sft,grpo,signal):
    """Choose a useful checkpoint without equating its utility to causal RL gain."""
    def qualifies(score):
        alarm=baseline.get('negative_false_alarm_rate')
        return (score.get('micro_f2',0)>0 and score['micro_f2']>=baseline['micro_f2']
            and score['valid_json_fraction']>=baseline['valid_json_fraction']
            and (alarm is None or (score.get('negative_false_alarm_rate') is not None and score['negative_false_alarm_rate']<=alarm)))
    selected=None
    if qualifies(sft):selected='sft'
    if qualifies(grpo) and (selected is None or (signal.get('task_signal_verified') and grpo['micro_f2']>sft['micro_f2'])):selected='grpo'
    return {'selected_stage':selected,'sft_qualifies':qualifies(sft),'grpo_qualifies':qualifies(grpo),
        'task_signal_verified':signal.get('task_signal_verified',False),'rl_improvement_claimed':False,
        'interpretation':'Checkpoint selected by fixed development utility; causal RL benefit requires the planned paired ablations'}

def evaluate_adapter(adapter,manifest,output,model_path,*,seed=17,max_tokens=192):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText,set_seed
    from peft import PeftModel
    from .training import PROMPT,parse_prediction,detection_score
    from .fast_fields import file_hash
    root=data_root().resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    doc=json.loads(Path(manifest).read_text());rows=doc['images']
    if not rows or any(r.get('split')!='validation' or r.get('zone') in ('zone8','zone9') for r in rows):raise ValueError('Use development validation only; test zones stay sealed')
    protocol_hash=digest({'adapter':None if adapter is None else file_hash(Path(adapter)/'adapter_model.safetensors'),
        'manifest':digest(doc),'seed':seed,'max_tokens':max_tokens,'prompt':PROMPT,'use_cache':True,'decoding_protocol':'explicit-kv-cache-v2'})
    if (output/'summary.json').exists():
        old=json.loads((output/'summary.json').read_text())
        if old['protocol_hash']!=protocol_hash:raise ValueError('Evaluation identity changed')
        return old
    set_seed(seed);torch.set_num_threads(4)
    if os.environ.get('AEROROZUM_GPU_MEMORY_FRACTION'):
        fraction=float(os.environ['AEROROZUM_GPU_MEMORY_FRACTION'])
        if not 0<fraction<=.9:raise ValueError('Evaluation allocator must preserve10% headroom')
        torch.cuda.set_per_process_memory_fraction(fraction)
    processor=AutoProcessor.from_pretrained(model_path)
    model=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,device_map='balanced',attn_implementation='sdpa',low_cpu_mem_usage=True)
    if adapter is not None:model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
    model.config.use_cache=True
    if getattr(model.config,'text_config',None) is not None:model.config.text_config.use_cache=True
    model.eval();scores=[];started=time.monotonic()
    for row in rows:
        key=digest([row['sha256'],protocol_hash]);raw_path=output/'raw'/(key+'.json')
        if raw_path.exists():raw=json.loads(raw_path.read_text())
        else:
            path=(root/row['image']).resolve()
            if not path.is_relative_to(root) or file_hash(path)!=row['sha256']:raise ValueError('Evaluation image path/hash changed')
            with Image.open(path) as im:image=im.convert('RGB')
            inputs=processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
            device=model.get_input_embeddings().weight.device
            inputs={k:(v.to(device=device,dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(device)) if torch.is_tensor(v) else v for k,v in inputs.items()}
            prefix=inputs['input_ids'].shape[-1]
            with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,max_new_tokens=max_tokens,use_cache=True)
            tokens=generated[0,prefix:].detach().cpu().tolist();raw={'id':row['id'],'image_sha256':row['sha256'],'protocol_hash':protocol_hash,'text':processor.decode(tokens,skip_special_tokens=True),'token_ids':tokens,'at':utcnow()};save(raw_path,raw)
        try:predictions=parse_prediction(raw['text']);valid=True
        except (ValueError,TypeError,KeyError):predictions=[];valid=False
        scores.append({**detection_score(predictions,row.get('boxes',[])),'valid_json':valid,'id':row['id'],'has_gold':bool(row.get('boxes'))})
        save(output/'status.json',{'status':'running','completed':len(scores),'total':len(rows),'at':utcnow()})
    tp=sum(r['tp'] for r in scores);fp=sum(r['fp'] for r in scores);fn=sum(r['fn'] for r in scores);neg=[r for r in scores if not r['has_gold']]
    result={'status':'completed','protocol_hash':protocol_hash,'examples':len(scores),'tp':tp,'fp':fp,'fn':fn,'micro_f2':5*tp/max(1,5*tp+4*fn+fp),
        'valid_json_fraction':sum(r['valid_json'] for r in scores)/len(scores),'negative_false_alarm_rate':sum(r['fp']>0 for r in neg)/len(neg) if neg else None,
        'elapsed_s':time.monotonic()-started,'scores':scores,'use_cache':True,'decoding_protocol':'explicit-kv-cache-v2','interpretation':'Development label agreement, not verified physical mine accuracy'}
    save(output/'summary.json',result);del model,processor;gc.collect();torch.cuda.empty_cache();return result

def run_pass(batch,validation,output,seed,sft_steps=32,grpo_steps=16,warm_start=None):
    from .training import train
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    document=json.loads(Path(batch).read_text());batch_id=document['metadata']['batch_id']
    os.environ.update(AEROROZUM_TRAINING_NAMESPACE=f'incremental-stream-v1/{batch_id}',AEROROZUM_DEVICE_MAP='balanced',AEROROZUM_SFT_LEARNING_RATE='2e-5')
    if warm_start:os.environ['AEROROZUM_WARMSTART_ADAPTER']=str(warm_start)
    else:os.environ.pop('AEROROZUM_WARMSTART_ADAPTER',None)
    model_path=os.environ.get('AEROROZUM_LOCAL_MODEL_PATH','/opt/aerorozum/models/gemma-4-12B-it')
    def progress(stage):save(root/'status.json',{'status':'running','stage':stage,'batch_id':batch_id,'seed':seed,'at':utcnow()})
    progress('sft');sft=train(batch,'stream_sft',seed,sft_steps);gc.collect()
    import torch
    torch.cuda.empty_cache();os.environ['AEROROZUM_SFT_CHECKPOINT']=str(data_root()/sft['adapter'])
    progress('grpo');grpo=train(batch,'stream_grpo',seed,grpo_steps);gc.collect();torch.cuda.empty_cache()
    task_signal=reward_signal_diagnostics((data_root()/grpo['adapter']).parent/'metrics.jsonl')
    progress('development_evaluation');candidate=evaluate_adapter(data_root()/grpo['adapter'],validation,root/'evaluation'/'candidate',model_path,seed=seed)
    progress('baseline_evaluation');baseline=cached_baseline(warm_start,validation,root/'evaluation'/'parent',model_path,seed=seed)
    promoted=(task_signal['task_signal_verified'] and candidate['micro_f2']>=baseline['micro_f2'] and candidate['valid_json_fraction']>=baseline['valid_json_fraction'] and candidate['micro_f2']>0
        and (baseline['negative_false_alarm_rate'] is None or candidate['negative_false_alarm_rate']<=baseline['negative_false_alarm_rate']))
    result={'status':'completed','batch_id':batch_id,'seed':seed,'sft':sft,'grpo':grpo,'baseline':baseline,'evaluation':candidate,
        'promoted':promoted,'reward_signal':task_signal,'adapter':grpo['adapter'],'at':utcnow(),'scope':'Incremental pass; full184/30k/three-seed protocol unchanged'}
    save(root/'result.json',result);save(root/'status.json',{'status':'completed','batch_id':batch_id,'seed':seed,'promoted':promoted,'at':utcnow()});return result

def cached_baseline(adapter,manifest,reference,model_path,seed=17):
    """Share deterministic base predictions; seed-specific adapters stay distinct."""
    from .fast_fields import file_hash
    from .training import PROMPT
    revision={name:file_hash(Path(model_path)/name) for name in ('config.json','model.safetensors.index.json') if (Path(model_path)/name).is_file()}
    key=digest({'revision':revision,'manifest_sha256':file_hash(manifest),'prompt':PROMPT,'max_tokens':192,'do_sample':False,'use_cache':True,'decoding_protocol':'explicit-kv-cache-v2',
                'adapter':file_hash(Path(adapter)/'adapter_model.safetensors') if adapter else None,'seed':seed if adapter else 0})
    cache=data_root()/'pipeline'/'baseline-cache'/key;cache.mkdir(parents=True,exist_ok=True)
    reference=Path(reference);reference.mkdir(parents=True,exist_ok=True)
    save(reference/'reference.json',{'cache':str(cache),'key':key,'deterministic_base_shared':adapter is None})
    with (cache/'.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        result=evaluate_adapter(adapter,manifest,cache,model_path,seed=seed if adapter else 0)
    save(reference/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch');p.add_argument('--validation',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--sft-steps',type=int,default=32);p.add_argument('--grpo-steps',type=int,default=16);p.add_argument('--warm-start');p.add_argument('--evaluate-only',action='store_true');p.add_argument('--adapter');a=p.parse_args()
    try:
        if a.evaluate_only:
            if not a.adapter:raise ValueError('Evaluation needs an explicit saved adapter')
            result=evaluate_adapter(a.adapter,a.validation,a.output,os.environ.get('AEROROZUM_LOCAL_MODEL_PATH','/opt/aerorozum/models/gemma-4-12B-it'),seed=a.seed)
        else:
            if not a.batch:raise ValueError('Training needs a frozen batch')
            result=run_pass(a.batch,a.validation,a.output,a.seed,a.sft_steps,a.grpo_steps,a.warm_start)
        print(json.dumps(result),flush=True)
    except Exception as exc:save(Path(a.output)/'status.json',{'status':'failed','error':str(exc)[:2000],'at':utcnow()});raise

if __name__=='__main__':main()
