#!/usr/bin/env python3
"""Frozen zone-7 inference; immutable raw generations precede all label scoring.

This standalone worker intentionally has no project imports, so copying it cannot
replace source used by an in-flight training process. Heavy imports are lazy.
"""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

PROMPT = ('Inspect this thermal image for provisional mine-like ground signatures. '
          'Return ONLY JSON {"detections":[{"bbox":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000}}]}. '
          'Boxes must tightly bound visible evidence, normalized to 0..1000. '
          'Return an empty detections list when no supported candidate is visible. '
          'Thermal appearance does not confirm a physical mine.')
MODEL = 'google/gemma-4-12B-it'
BOX_KEYS = ('x_min', 'y_min', 'x_max', 'y_max')
LIMITATION = ('Zone 7 contains annotated positive images only: negative-image false-alarm rate cannot be measured. '
              'Agreement is with the supplied manual/reviewed-AI boxes, not independently verified physical mine identities. '
              'Validation may inform subsequent development; zones 8 and 9 remain untouched.')


def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name,suffix='.tmp')
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        os.replace(temp,path)
    finally:Path(temp).unlink(missing_ok=True)

def box(value):
    if not isinstance(value,dict) or any(k not in value for k in BOX_KEYS):raise ValueError('Missing named box coordinate')
    b={k:float(value[k]) for k in BOX_KEYS}
    if not all(math.isfinite(v) for v in b.values()):raise ValueError('Non-finite box')
    if not(0<=b['x_min']<b['x_max']<=1000 and 0<=b['y_min']<b['y_max']<=1000):raise ValueError('Invalid box')
    return b

def iou(a,b):
    a,b=box(a),box(b)
    intersection=max(0,min(a['x_max'],b['x_max'])-max(a['x_min'],b['x_min']))*max(0,min(a['y_max'],b['y_max'])-max(a['y_min'],b['y_min']))
    area=lambda x:(x['x_max']-x['x_min'])*(x['y_max']-x['y_min'])
    return intersection/max(1e-12,area(a)+area(b)-intersection)

def parse_prediction(text):
    text=text.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    d=json.loads(text)
    if not isinstance(d,dict) or not isinstance(d.get('detections'),list) or len(d['detections'])>100:raise ValueError('Missing/invalid detections array')
    return [box(x['bbox']) for x in d['detections']]

def score_boxes(predicted,gold,threshold=.5):
    predicted=[box(b.get('bbox',b)) for b in predicted];gold=[box(b.get('bbox',b)) for b in gold]
    edges={i:sorted([j for j,g in enumerate(gold) if iou(p,g)>=threshold],key=lambda j:iou(p,gold[j]),reverse=True) for i,p in enumerate(predicted)}
    matched={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in matched or augment(matched[j],seen):matched[j]=i;return True
        return False
    for i in range(len(predicted)):augment(i,set())
    tp=len(matched);fp=len(predicted)-tp;fn=len(gold)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'f2':5*tp/(5*tp+4*fn+fp) if gold else float(not predicted),
            'matched_iou':sum(iou(predicted[i],gold[j]) for j,i in matched.items())/max(1,tp),
            'mean_best_gold_iou':sum(max((iou(g,p) for p in predicted),default=0.) for g in gold)/max(1,len(gold))}

def contained(root,relative):
    p=Path(relative)
    if p.is_absolute():raise ValueError('Image must be artifact-relative')
    target=(Path(root)/p).resolve();target.relative_to(Path(root).resolve())
    return target

def freeze_validation(source_manifest,training_manifest,artifact_root,output,expected_count=88):
    source=json.loads(Path(source_manifest).read_text());training=json.loads(Path(training_manifest).read_text())
    train_hashes={r['sha256'] for r in training['images']};rows=[];seen=set()
    for r in source['images']:
        if r.get('split')!='validation':continue
        if r.get('zone')!='zone7' or '__zone7__' not in r['id']:raise ValueError('Validation must contain zone 7 only')
        if r.get('provenance') not in ('original_manual','reviewed_ai'):raise ValueError('Unapproved validation annotation')
        if not r.get('boxes'):raise ValueError('This protocol is explicitly positive-only')
        if r['sha256'] in train_hashes or r['sha256'] in seen:raise ValueError('Duplicate/leaked validation image')
        if file_sha(contained(artifact_root,r['image']))!=r['sha256']:raise ValueError('Validation image checksum mismatch')
        [box(b.get('bbox',b)) for b in r['boxes']];rows.append(r);seen.add(r['sha256'])
    rows.sort(key=lambda r:r['id'])
    if len(rows)!=expected_count:raise ValueError(f'Expected {expected_count} validation images, found {len(rows)}')
    document={'metadata':{'schema_version':1,'split':'validation','zones':['zone7'],'images':len(rows),'positive_only':True,
                         'source_manifest_hash':digest(source),'training_manifest_hash':digest(training),'created_at':now(),
                         'annotation_provenance':{p:sum(r['provenance']==p for r in rows) for p in sorted({r['provenance'] for r in rows})},
                         'limitation':LIMITATION},'images':rows}
    output=Path(output)
    if output.exists():
        old=json.loads(output.read_text())
        if old['images']!=rows or old['metadata']['training_manifest_hash']!=digest(training):raise ValueError('Frozen validation manifest differs')
        return old
    save(output,document);return document

def aggregate(scores):
    tp=sum(x['tp'] for x in scores);fp=sum(x['fp'] for x in scores);fn=sum(x['fn'] for x in scores)
    return {'images':len(scores),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,'micro_f2':5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,
            'macro_f2':sum(x['f2'] for x in scores)/max(1,len(scores)),
            'mean_matched_iou':sum(x['matched_iou']*x['tp'] for x in scores)/max(1,tp),
            'mean_best_gold_iou':sum(x['mean_best_gold_iou'] for x in scores)/max(1,len(scores)),
            'invalid_json':sum(not x['valid_json'] for x in scores),'empty_predictions':sum(x['prediction_count']==0 for x in scores),
            'negative_image_false_alarm_rate':None}

def evaluate_row(row,version,protocol_hash,output,infer):
    """Raw bytes are committed before parsing/scoring, including malformed output."""
    output=Path(output);key=digest({'image_sha256':row['sha256'],'version':version,'protocol_hash':protocol_hash})
    raw_path=output/'raw'/version/(key+'.json');score_path=output/'scores'/version/(key+'.json')
    if raw_path.exists():
        raw=json.loads(raw_path.read_text())
        if raw.get('image_sha256')!=row['sha256'] or raw.get('protocol_hash')!=protocol_hash:raise ValueError('Cache identity mismatch')
    else:
        generated=infer(row)
        raw={'id':row['id'],'image':row['image'],'image_sha256':row['sha256'],'model_version':version,'protocol_hash':protocol_hash,
             'generated_at':now(),**generated}
        save(raw_path,raw)
    try:predicted=parse_prediction(raw['text']);valid=True;error=None
    except (ValueError,KeyError,TypeError,OverflowError) as ex:predicted=[];valid=False;error=str(ex)[:300]
    score={**score_boxes(predicted,row['boxes']),'id':row['id'],'image_sha256':row['sha256'],'model_version':version,
           'valid_json':valid,'parse_error':error,'prediction_count':len(predicted),'predicted_boxes':predicted,
           'provenance':row['provenance'],'raw_path':str(raw_path.relative_to(output)),'raw_sha256':file_sha(raw_path),
           'scored_at':now(),'threshold':.5}
    save(score_path,score);return score

def model_fingerprint(model_path):
    model_path=Path(model_path)
    files=sorted(p for p in model_path.iterdir() if p.name in ('config.json','generation_config.json','model.safetensors.index.json','adapter_config.json') or p.name.startswith('adapter_model.'))
    if not files:raise ValueError('Missing model/adapter identity')
    return {'files':{p.name:file_sha(p) for p in files},'directory':model_path.name}

def run(manifest,artifact_root,output,model_path,training_root,seed=17,max_tokens=384):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText,set_seed
    from peft import PeftModel
    root=Path(artifact_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    document=json.loads(Path(manifest).read_text());rows=document['images']
    if len(rows)!=88 or document['metadata'].get('zones')!=['zone7']:raise ValueError('Frozen 88-image zone-7 validation required')
    for row in rows:
        if row['split']!='validation' or row['zone']!='zone7':raise ValueError('Held-out test/train image in validation')
        if file_sha(contained(root,row['image']))!=row['sha256']:raise ValueError('Image checksum mismatch')
    training_root=Path(training_root);adapters={v:training_root/v/f'seed-{seed}'/'adapter' for v in ('real_sft','real_grpo')}
    for v,adapter in adapters.items():
        summary=json.loads((adapter.parent/'summary.json').read_text())
        if summary['status']!='completed' or summary['seed']!=seed or summary['model']!=MODEL:raise ValueError('Training stage incomplete/mismatched')
        if summary['manifest_hash']!=document['metadata']['training_manifest_hash']:raise ValueError('Training dataset differs from frozen validation pairing')
    cfg=json.loads((Path(model_path)/'config.json').read_text())
    if cfg.get('model_type')!='gemma4_unified':raise ValueError('Expected unquantized Gemma4 Unified')
    protocol={'model':MODEL,'seed':seed,'manifest_hash':digest(document),'prompt':PROMPT,'decoding':{'do_sample':False,'num_beams':1,'max_new_tokens':max_tokens},
              'iou_threshold':.5,'base':model_fingerprint(model_path),'adapters':{k:model_fingerprint(v) for k,v in adapters.items()},
              'limitation':LIMITATION,'gold_visible_to_model':False,'raw_before_scoring':True}
    protocol_hash=digest(protocol);protocol['protocol_hash']=protocol_hash
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('Evaluation protocol changed; use a separate output directory')
    save(output/'protocol.json',protocol);save(output/'status.json',{'status':'loading_model','completed':0,'total':len(rows)*3,'at':now()})
    set_seed(seed);torch.set_num_threads(8)
    processor=AutoProcessor.from_pretrained(model_path)
    model=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,device_map='balanced',attn_implementation='sdpa',low_cpu_mem_usage=True)
    model.eval();model.config.use_cache=True
    all_scores={};done=0;started=time.monotonic()
    for version in ('base','real_sft','real_grpo'):
        if version=='real_sft':model=PeftModel.from_pretrained(model,str(adapters[version]),adapter_name=version,is_trainable=False);model.eval()
        elif version=='real_grpo':model.load_adapter(str(adapters[version]),adapter_name=version,is_trainable=False);model.set_adapter(version);model.eval()
        def infer(row):
            image=Image.open(contained(root,row['image'])).convert('RGB')
            messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
            inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
            device=model.get_input_embeddings().weight.device
            inputs={k:(v.to(device=device,dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(device)) if torch.is_tensor(v) else v for k,v in inputs.items()}
            token_count=inputs['input_ids'].shape[-1];start=time.monotonic()
            with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=max_tokens,use_cache=True)
            ids=generated[0,token_count:].detach().cpu().tolist()
            text=processor.decode(ids,skip_special_tokens=True)
            return {'text':text,'generated_token_ids':ids,'generated_tokens':len(ids),'input_tokens':token_count,
                    'input_token_hash':digest(inputs['input_ids'].detach().cpu().tolist()),'elapsed_s':time.monotonic()-start,
                    'hit_max_new_tokens':len(ids)>=max_tokens}
        scores=[]
        for row in rows:
            scores.append(evaluate_row(row,version,protocol_hash,output,infer));done+=1
            status={'status':'running','version':version,'completed':done,'version_completed':len(scores),'version_total':len(rows),'total':len(rows)*3,
                    'elapsed_s':time.monotonic()-started,'at':now(),'partial_metrics':aggregate(scores),'limitation':LIMITATION}
            save(output/'status.json',status);print(json.dumps(status),flush=True)
        all_scores[version]=scores
        summary={'status':'completed','model_version':version,'protocol_hash':protocol_hash,**aggregate(scores),
                 'by_provenance':{p:aggregate([s for s in scores if s['provenance']==p]) for p in sorted({s['provenance'] for s in scores})},'limitation':LIMITATION,'at':now()}
        save(output/version/'summary.json',summary)
    summaries={v:json.loads((output/v/'summary.json').read_text()) for v in all_scores}
    comparison={'status':'completed','protocol_hash':protocol_hash,'seed':seed,'validation_images':len(rows),'versions':summaries,'limitation':LIMITATION,'at':now(),
                'differences':{'sft_minus_base_f2':summaries['real_sft']['micro_f2']-summaries['base']['micro_f2'],
                               'grpo_minus_sft_f2':summaries['real_grpo']['micro_f2']-summaries['real_sft']['micro_f2']}}
    save(output/'comparison.json',comparison);save(output/'status.json',{'status':'completed','completed':done,'total':len(rows)*3,'at':now(),'comparison':comparison})
    return comparison

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--artifact-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-path',default='/opt/aerorozum/models/gemma-4-12B-it');p.add_argument('--training-root',default='/opt/aerorozum/artifacts/training/real-transfer-machine-labels-v1/google_gemma-4-12B-it')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--max-tokens',type=int,default=384);a=p.parse_args()
    try:print(json.dumps(run(a.manifest,a.artifact_root,a.output,a.model_path,a.training_root,a.seed,a.max_tokens),indent=2))
    except Exception as ex:
        save(Path(a.output)/'failure.json',{'status':'failed','error':str(ex)[:2000],'at':now()});raise
if __name__=='__main__':main()
