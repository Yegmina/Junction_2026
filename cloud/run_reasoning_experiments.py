"""Run the frozen development prompt/effort study with the private hackathon key."""
import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline',default='2026-11-15T05:45:00+00:00')
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    os.environ['AEROROZUM_GEMINI_KEY_FILE']='/etc/aerorozum/hackathon-key'
    os.environ.pop('GEMINI_API_KEY',None)
    from cloud.airborne_worker import metadata_record_store
    from aerorozum.reasoning_experiments import run
    store=metadata_record_store('aerorozum-jetson-testdata-smart-bird',prefix='robotics-er2-labels/reasoning-study-v1')
    result=asyncio.run(run('/opt/aerorozum/artifacts',args.output,store,datetime.fromisoformat(args.deadline),probe=args.probe))
    print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))
