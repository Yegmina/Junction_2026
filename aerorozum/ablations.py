"""Frozen label-source controls, separate from the full synthetic experiment."""
from __future__ import annotations
import argparse,copy,gc,json,math,os,random
from pathlib import Path
from .common import data_root,digest,save,utcnow

KIND='sft_label_source_ablation_v1'
ARMS=('reviewed_anchors','matched_native_mixture')

def _diverse(rows,count,seed):
    rng=random.Random(seed);groups={}
    for row in sorted(rows,key=lambda r:r['sha256']):
        groups.setdefault(row.get('source_group') or row.get('zone') or 'unknown',[]).append(row)
    for group in groups.values():rng.shuffle(group)
    result=[]
    while groups and len(result)<count:
        for key in sorted(list(groups)):
            if groups[key] and len(result)<count:result.append(groups[key].pop())
            if not groups[key]:groups.pop(key)
    return result

def freeze_label_source_ablation(root=None):
    """Use every reviewed anchor and a matched-size native mixture control."""
    from .streaming import StreamCoordinator
    from .training import load_stream_manifest
    base=Path(root or data_root()).resolve();worker=StreamCoordinator(base)
    rows,sessions,_,_=worker._catalog();train=[r for r in rows if r['split']=='train']
    anchors=[r for r in train if r['provenance'] in ('original_manual','reviewed_ai') and r.get('boxes')]
    native=[r for r in train if r['provenance']=='user_accepted_model' and r.get('boxes')]
    zeros=[r for r in train if r['provenance']=='user_accepted_model' and not r.get('boxes')]
    n=len(anchors);native_count=n//2
    if n<16 or len(native)<native_count or len(zeros)<n:raise ValueError('Insufficient clean train anchors/native positives/explicit-zero controls for matched ablation')
    if not (base/'pipeline/validation.json').is_file():raise ValueError('The original frozen development set is required')
    screened=[r for r in zeros if r.get('row_kind')=='zip_screened_background']
    negatives=_diverse(screened,min(n,len(screened)),101)
    excluded={r['sha256'] for r in negatives};negatives+=_diverse([r for r in zeros if r['sha256'] not in excluded],n-len(negatives),103)
    mixed=_diverse(anchors,n-native_count,107)+_diverse(native,native_count,109)
    arms={ARMS[0]:anchors+negatives,ARMS[1]:mixed+negatives}
    family=digest({'kind':KIND,'arms':{name:sorted(r['sha256'] for r in items) for name,items in arms.items()},'epochs':1,'effective_batch':4,'seeds':[17,43,91]})[:24]
    folder=base/'pipeline/ablations'/family;request=base/'pipeline/job-requests'/(family+'.json')
    if request.is_file():return json.loads(request.read_text())
    jobs=[];summary={}
    for arm,items in arms.items():
        items=copy.deepcopy(items);random.Random(113).shuffle(items);batch=digest([family,arm,sorted(r['sha256'] for r in items)])[:24]
        manifest=folder/arm/'manifest.json';steps=math.ceil(len(items)/4)
        document={'metadata':{'namespace':'incremental-stream-v1','frozen_stream_batch':True,'status':'passed','batch_id':batch,
            'experiment_branch':KIND,'ablation_family_id':family,'ablation_arm':arm,'epochs':1,'effective_batch':4,
            'reward_protocol':'soft-iou-f2-v2','accepted_pair_certificates':{},'full_research_protocol_unchanged':True,
            'interpretation':'Matched label-source SFT control; no synthetic data and no RL improvement claim','created_at':utcnow()},
            'session_splits':sessions,'images':items}
        save(manifest,document);load_stream_manifest(manifest,'stream_sft',root=base)
        summary[arm]={'images':len(items),'positive_images':sum(bool(r['boxes']) for r in items),'zero_images':sum(not r['boxes'] for r in items),
            'reviewed_anchor_images':sum(r['provenance'] in ('original_manual','reviewed_ai') for r in items),'native_positive_images':sum(bool(r['boxes']) and r['provenance']=='user_accepted_model' for r in items),'optimizer_updates':steps}
        for seed in (17,43,91):jobs.append({'batch_id':batch,'manifest':str(manifest.relative_to(base)),'arm':arm,'seed':seed,'sft_steps':steps})
    result={'schema_version':1,'kind':KIND,'family_id':family,'created_at':utcnow(),'jobs':jobs,'arms':summary,
        'controls':{'same_negative_image_hashes':sorted(r['sha256'] for r in negatives),'initialization':'same frozen base model, no warm start',
            'validation':'original immutable development set; sealed tests unavailable','learning_rate':2e-5,'epochs':1,'effective_batch':4,
            'grpo':'not repeated; this isolates label-source and sufficient SFT coverage'},'full_research_protocol_unchanged':True}
    save(folder/'protocol.json',result);save(request,result);return result

def run_ablation(batch,validation,output,seed,steps,namespace):
    from .training import train
    from .stream_training import cached_baseline,evaluate_adapter,checkpoint_utility
    document=json.loads(Path(batch).read_text());metadata=document['metadata'];root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if metadata.get('experiment_branch')!=KIND or steps!=math.ceil(len(document['images'])/4):raise ValueError('Frozen one-epoch ablation protocol differs')
    os.environ.update(AEROROZUM_TRAINING_NAMESPACE=namespace,AEROROZUM_DEVICE_MAP='balanced',AEROROZUM_SFT_LEARNING_RATE='2e-5')
    os.environ.pop('AEROROZUM_WARMSTART_ADAPTER',None)
    model=os.environ.get('AEROROZUM_LOCAL_MODEL_PATH','/opt/aerorozum/models/gemma-4-12B-it')
    def progress(stage):save(root/'status.json',{'status':'running','stage':stage,'seed':seed,'batch_id':metadata['batch_id'],'arm':metadata['ablation_arm'],'at':utcnow()})
    progress('baseline_evaluation');baseline=cached_baseline(None,validation,root/'evaluation/parent',model,seed=seed)
    progress('sft');sft=train(batch,'stream_sft',seed,steps);gc.collect()
    import torch
    torch.cuda.empty_cache();progress('development_evaluation')
    evaluation=evaluate_adapter(data_root()/sft['adapter'],validation,root/'evaluation/sft',model,seed=seed)
    utility=checkpoint_utility(baseline,evaluation,evaluation,{'task_signal_verified':False})
    result={'status':'completed','kind':KIND,'batch_id':metadata['batch_id'],'family_id':metadata['ablation_family_id'],'arm':metadata['ablation_arm'],
        'seed':seed,'sft':sft,'baseline':baseline,'evaluation':evaluation,'checkpoint_utility':utility,'adapter':sft['adapter'],
        'rl_improvement_claimed':False,'full_research_protocol_unchanged':True,'at':utcnow()}
    save(root/'result.json',result);save(root/'status.json',{'status':'completed','seed':seed,'arm':metadata['ablation_arm'],'at':utcnow()});return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--root');p.add_argument('--batch');p.add_argument('--validation');p.add_argument('--output');p.add_argument('--seed',type=int);p.add_argument('--steps',type=int);p.add_argument('--namespace');args=p.parse_args()
    if args.freeze:print(json.dumps(freeze_label_source_ablation(args.root)),flush=True);return
    if not all([args.batch,args.validation,args.output,args.seed,args.steps,args.namespace]):p.error('A frozen batch, validation, output, seed, steps and namespace are required')
    try:print(json.dumps(run_ablation(args.batch,args.validation,args.output,args.seed,args.steps,args.namespace)),flush=True)
    except Exception as exc:save(Path(args.output)/'status.json',{'status':'failed','error':str(exc)[:2000],'at':utcnow()});raise

if __name__=='__main__':main()
