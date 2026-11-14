"""Explicit held-out image comparison after an immutable development selection."""
import argparse
import asyncio
import json
import os
from pathlib import Path


def student(args):
    from aerorozum.common import data_root
    from aerorozum.frozen_test import validate_test_contract
    from aerorozum.stream_training import _evaluate_images
    receipt,document,contract=validate_test_contract(data_root(),args.selection,args.manifest)
    return _evaluate_images(receipt['selected']['adapter'],document,Path(args.output)/'gemma',
        os.environ['AEROROZUM_LOCAL_MODEL_PATH'],test_contract=contract)


async def frontier(args):
    from aerorozum.common import data_root,save,utcnow
    from aerorozum.frozen_test import validate_test_contract
    from aerorozum.providers import GeminiProvider
    from aerorozum.training import PROMPT,parse_prediction,detection_score
    root=data_root().resolve();_,document,contract=validate_test_contract(root,args.selection,args.manifest)
    output=Path(args.output)/args.model;output.mkdir(parents=True,exist_ok=True)
    cfg=dict(bulk_model=args.model,temperature=0.,concurrency=4,requests_per_minute=30,
        input_tokens_per_minute=200000,max_requests=len(document['images'])+4,max_retries=1,reserve_output_tokens=8192)
    protocol=dict(frozen_test_contract=contract,model=args.model,prompt=PROMPT,generation=cfg,
        inference_difference='Same visible prompt/images. Frontier native JSON, 8192 output tokens including default reasoning; Gemma greedy 192 tokens. Not matched compute.')
    protocol_file=output/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text())!=protocol:raise ValueError('Held-out comparison identity changed')
    save(protocol_file,protocol);provider=GeminiProvider(cfg,output/'provider');scores=[]
    async def one(row):
        prediction=await provider.generate('frozen_zip_test_v1',PROMPT,[root/row['image']],model=args.model,max_output_tokens=8192)
        try:boxes=parse_prediction(json.dumps(prediction));valid=True
        except (ValueError,TypeError,KeyError):boxes=[];valid=False
        score=dict(id=row['id'],image_sha256=row['sha256'],provenance=row['provenance'],
            has_gold=bool(row.get('boxes')),valid_json=valid,**detection_score(boxes,row.get('boxes',[])))
        save(output/'predictions'/(row['sha256']+'.json'),dict(score=score,prediction=prediction));scores.append(score)
        save(output/'status.json',dict(status='running',completed=len(scores),total=len(document['images']),at=utcnow()))
    tasks=[asyncio.create_task(one(row)) for row in document['images']]
    try:
        await asyncio.gather(*tasks)
        tp=sum(r['tp'] for r in scores);fp=sum(r['fp'] for r in scores);fn=sum(r['fn'] for r in scores);neg=[r for r in scores if not r['has_gold']]
        result=dict(status='completed',at=utcnow(),model=args.model,examples=len(scores),tp=tp,fp=fp,fn=fn,
            micro_f2=5*tp/max(1,5*tp+4*fn+fp),valid_json_fraction=sum(r['valid_json'] for r in scores)/len(scores),
            negative_false_alarm_rate=sum(r['fp']>0 for r in neg)/len(neg) if neg else None,
            protocol=protocol,scores=scores,scope='Frozen held-out ZIP label agreement; not native mission reward or field validation',
            frontier_superiority_verified=False)
        save(output/'summary.json',result);return result
    except Exception:
        provider.request_drain();await asyncio.gather(*tasks,return_exceptions=True)
        save(output/'status.json',dict(status='failed',at=utcnow(),note='Provider/data failures are not scored as model failures'))
        raise
    finally:await provider.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--model');args=p.parse_args()
    result=asyncio.run(frontier(args)) if args.model else student(args)
    print(json.dumps({k:v for k,v in result.items() if k not in ['scores','protocol']}))
