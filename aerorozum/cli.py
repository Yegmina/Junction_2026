from __future__ import annotations
import argparse
import asyncio
import json


def main():
    p=argparse.ArgumentParser(prog='aerorozum')
    commands=p.add_subparsers(dest='command',required=True)
    c=commands.add_parser('preflight');c.add_argument('--check-models',action='store_true')
    c=commands.add_parser('audit');c.add_argument('--archive');c.add_argument('--export',action='store_true')
    commands.add_parser('freeze')
    c=commands.add_parser('teacher');c.add_argument('manifest');c.add_argument('--limit',type=int);c.add_argument('--model')
    c=commands.add_parser('serve');c.add_argument('--host',default='127.0.0.1');c.add_argument('--port',type=int,default=18901)
    c=commands.add_parser('train');c.add_argument('--manifest',required=True);c.add_argument('--stage',choices=['compatibility','sft','grpo','policy_compatibility','policy_sft','policy_grpo','real_sft','real_grpo'],default='compatibility');c.add_argument('--seed',type=int,default=17);c.add_argument('--steps',type=int,default=10);c.add_argument('--model')
    c=commands.add_parser('generate');c.add_argument('--manifest',required=True);c.add_argument('--registry',required=True);c.add_argument('--plan-only',action='store_true')
    c=commands.add_parser('generate-demo');c.add_argument('--count',type=int,default=8)
    c=commands.add_parser('campaign-batch');c.add_argument('--manifest',required=True);c.add_argument('--registry',required=True);c.add_argument('--split',default='train');c.add_argument('--model');c.add_argument('--concurrency',type=int)
    c=commands.add_parser('campaign-demo');c.add_argument('--count',type=int,default=2);c.add_argument('--max-actions',type=int,default=16)
    commands.add_parser('cloud-status')
    commands.add_parser('teardown')
    c=commands.add_parser('package-submissions')
    args=p.parse_args()
    if args.command=='preflight':
        from .preflight import snapshot
        result=snapshot(args.check_models)
    elif args.command=='audit':
        from .dataset import audit
        result=audit(args.archive,args.export)
    elif args.command=='freeze':
        from .experiment import freeze
        d=freeze();result=dict(status=d['status'],campaigns=len(d['campaigns']),physical_trajectories=len(d['physics']),config_hash=d['config_hash'])
    elif args.command=='teacher':
        from .providers import analyze_manifest
        result=asyncio.run(analyze_manifest(args.manifest,args.limit,args.model))
    elif args.command=='serve':
        import uvicorn
        uvicorn.run('aerorozum.api:app',host=args.host,port=args.port)
        return
    elif args.command=='train':
        from .training import train
        result=train(args.manifest,args.stage,args.seed,args.steps,args.model)
    elif args.command=='generate':
        from .generation import generate_dataset,plan_generation
        result=(plan_generation if args.plan_only else generate_dataset)(args.manifest,args.registry)
    elif args.command=='generate-demo':
        from .generation import generate_demo_dataset
        result=generate_demo_dataset(count=args.count)
    elif args.command=='campaign-demo':
        from .campaigns import run_demo_campaigns
        result=asyncio.run(run_demo_campaigns(count=args.count,max_actions=args.max_actions))
    elif args.command=='campaign-batch':
        from .campaigns import run_campaign_batch
        result=asyncio.run(run_campaign_batch(args.manifest,args.registry,split=args.split,model=args.model,concurrency=args.concurrency))
    elif args.command=='cloud-status':
        from .cloud import inventory
        result=inventory()
    elif args.command=='teardown':
        from .cloud import teardown
        result=teardown()
    else:
        from .submissions import package
        result=package()
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
