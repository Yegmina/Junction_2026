"""Hand over the rate window to newly qualified inputs after the main pass."""
from datetime import datetime,timezone
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time


if __name__=='__main__':
    import cloud.run_high_thought_airborne as worker
    from aerorozum.common import config,save,utcnow
    from aerorozum.providers import request_identity
    from aerorozum.label_prompts import PROMPTS
    original=worker.OUTPUT;destination=original.with_name('robotics-er2-high-thinking-requalified-v1')
    inputs=worker.ROOT/'airborne-requalification-v1/frozen-new-high-inputs.json'
    document=json.loads(inputs.read_text())
    if (destination/'settled.json').exists():raise SystemExit('Requalification pass already settled; preserve the first outcomes')
    while datetime.now(timezone.utc)<worker.DEADLINE:
        settled=original/'settled.json'
        if settled.exists():
            primary=json.loads(settled.read_text());time.sleep(max(0,65-(time.time()-datetime.fromisoformat(primary['at']).timestamp())))
            prompt=json.loads((original/'prompt-selection.json').read_text())['prompt_version']
            destination.mkdir(parents=True,exist_ok=True)
            # Reuse an identical paid prompt/profile/pixel completion even if
            # its source-generation binding is new. Raw originals stay intact.
            cfg=dict(config()['gemini'],thinking_level='high');cfg.pop('thinking_budget',None)
            reused=0
            for row in document['images']:
                identity=request_identity(cfg,role='robotics_er2_thermal_labels',prompt=PROMPTS[prompt],
                    images=[row['sha256']],model='gemini-robotics-er-2-preview',max_output_tokens=65536)
                for part,name in [('responses',identity+'.json'),('raw',identity+'-0.json')]:
                    old=original/'provider'/part/name;new=destination/'provider'/part/name
                    if old.exists() and not new.exists():
                        new.parent.mkdir(parents=True,exist_ok=True);os.link(old,new)
                        if part=='responses':reused+=1
            save(destination/'handover.json',dict(at=utcnow(),primary_finished_at=primary['at'],
                input_sha256=document['sha256'],new_unique_frames=len(document['images']),
                identical_paid_completions_reused=reused,original_audits_and_labels_unchanged=True))
            worker.OUTPUT=destination
            asyncio.run(worker.run(prompt,document));break
        time.sleep(30)
