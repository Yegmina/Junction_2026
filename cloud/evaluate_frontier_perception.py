"""Bounded frontier controls on the same development images and visible prompt.

No gold boxes enter the provider request. Raw paid responses are retained by
GeminiProvider before parsing, and valid cached responses are reused. Provider
errors stop the comparison instead of being counted as model-skill failures.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path


async def evaluate(model,output):
    from aerorozum.common import data_root,save,utcnow,digest
    from aerorozum.providers import GeminiProvider
    from aerorozum.training import PROMPT,parse_prediction,detection_score
    root=data_root().resolve();output=Path(output)/model;output.mkdir(parents=True,exist_ok=True)
    document=json.loads((root/'pipeline/validation.json').read_text());rows=document['images']
    if not rows or any(r.get('split')!='validation' or r.get('zone') in ('zone8','zone9') for r in rows):raise ValueError('Development-only experiment; sealed test access refused')
    cfg=dict(bulk_model=model,temperature=0.,concurrency=4,requests_per_minute=30,input_tokens_per_minute=200000,max_requests=len(rows)+4,max_retries=1,reserve_output_tokens=8192)
    protocol=dict(model=model,prompt=PROMPT,validation_manifest_hash=digest(document),generation=cfg,
        scope='Development perception comparison; not native mission-policy evaluation',
        note='Same visible prompt and images as Gemma; frontier receives native JSON mode and 8192 total output tokens including its default reasoning, versus Gemma greedy 192 output tokens. No ground truth sent.')
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Refusing to reuse a changed frontier comparison identity')
    save(path,protocol);provider=GeminiProvider(cfg,output/'provider');scores=[]
    job_id='frontier-development-'+model;job=root/'jobs'/(job_id+'.json')
    def status(state,**more):save(job,dict(id=job_id,kind='frontier_perception_evaluation',status=state,model=model,completed=len(scores),total=len(rows),updated_at=utcnow(),detail='Frozen development images and prompt; paid outputs retained. Native Harbor scores are separate.',**more))
    status('running')
    async def one(row):
        image=(root/row['image']).resolve()
        if not image.is_relative_to(root) or hashlib.sha256(image.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Evaluation image identity changed')
        prediction=await provider.generate('frontier_perception_development_v1',PROMPT,[image],model=model,max_output_tokens=8192)
        try:boxes=parse_prediction(json.dumps(prediction));valid=True
        except (ValueError,TypeError,KeyError):boxes=[];valid=False
        score=dict(id=row['id'],image_sha256=row['sha256'],has_gold=bool(row.get('boxes')),valid_json=valid,**detection_score(boxes,row.get('boxes',[])))
        save(output/'predictions'/(row['sha256']+'.json'),dict(score=score,prediction=prediction))
        scores.append(score);status('running')
    tasks=[asyncio.create_task(one(row)) for row in rows]
    try:
        await asyncio.gather(*tasks)
        tp=sum(r['tp'] for r in scores);fp=sum(r['fp'] for r in scores);fn=sum(r['fn'] for r in scores);neg=[r for r in scores if not r['has_gold']]
        summary=dict(status='completed',model=model,examples=len(scores),tp=tp,fp=fp,fn=fn,micro_f2=5*tp/max(1,5*tp+4*fn+fp),valid_json_fraction=sum(r['valid_json'] for r in scores)/len(scores),negative_false_alarm_rate=sum(r['fp']>0 for r in neg)/len(neg) if neg else None,protocol=protocol,sealed_test_accessed=False,frontier_superiority_verified=False,scores=scores)
        save(output/'summary.json',summary);status('completed',metrics={k:v for k,v in summary.items() if k in ('micro_f2','tp','fp','fn','valid_json_fraction','negative_false_alarm_rate')})
    except Exception:
        provider.request_drain();await asyncio.gather(*tasks,return_exceptions=True)
        status('failed',detail_override='Provider or data error; do not count as frontier model failure. Inspect retained raw responses before resuming.')
        raise
    finally:await provider.close()


async def main(args):
    for model in args.model:await evaluate(model,args.output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args();asyncio.run(main(a))
