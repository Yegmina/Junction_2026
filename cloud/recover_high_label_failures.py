"""One bounded, separately versioned recovery of failed HIGH-frame requests.

Retains every original error and malformed paid response. Uses original pixels,
unchanged confidence rules, and a format clarification registered as a new
prompt. A failed recovery remains quarantined, never a negative label.
"""
from datetime import datetime,timezone
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/opt/aerorozum');BASE=ROOT/'airborne-labeler/artifacts/labelling'
NIGHT=ROOT/'artifacts/overnight-20261115T000942'
VERSION='thermal-v2-format-recovery-v1'


def main():
    from aerorozum.common import save,digest,utcnow
    from aerorozum.label_prompts import PROMPTS,PROMPT_V2
    import cloud.run_high_thought_airborne as worker
    folders=[BASE/'robotics-er2-high-thinking-v1',BASE/'robotics-er2-high-thinking-requalified-v1']
    out=BASE/'robotics-er2-high-thinking-recovery-v1'
    if (out/'settled.json').exists():raise ValueError('Recovery already settled; never replay its outcomes')
    while datetime.now(timezone.utc)<worker.DEADLINE:
        if all((p/'settled.json').exists() for p in folders):break
        time.sleep(20)
    else:raise ValueError('First-pass labels did not settle before recovery cutoff')
    latest=max(datetime.fromisoformat(json.loads((p/'settled.json').read_text())['at']).timestamp() for p in folders)
    time.sleep(max(0,65-(time.time()-latest)))
    manifest=out/'frozen-inputs.json'
    if manifest.exists():raise ValueError('A partial bounded recovery exists; review it before any restart')
    frozen=[json.loads((folders[0]/'frozen-inputs.json').read_text()),json.loads((ROOT/'airborne-requalification-v1/frozen-new-high-inputs.json').read_text())]
    originals={r['id']:r for document in frozen for r in document['images']}
    valid=set();failures={}
    for folder in folders:
        for path in (folder/'results').glob('*.json'):
            value=json.loads(path.read_text())
            if value.get('status')!='invalid_model_response' and isinstance(value.get('detections'),list):valid.add(value['id'])
        for path in (folder/'checkpoints').glob('*.json'):
            value=json.loads(path.read_text())
            if value.get('status')=='failed':failures.setdefault(value['id'],[]).append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),error=value.get('error'),category=value.get('category')))
    rows=[dict(originals[key],high_thinking_recovery=dict(protocol=VERSION,original_failures=records,
        prior_paid_outputs_unchanged=True,maximum_additional_attempts=1)) for key,records in failures.items() if key not in valid]
    prompt=PROMPT_V2+'\nReturn exactly ONE valid JSON object and no extra prose, markdown or second JSON object. Every candidate bbox must contain finite numeric x_min, y_min, x_max, y_max in 0..1000, with x_min < x_max and y_min < y_max. Preserve every plausible subtle candidate with its calibrated confidence; this is a formatting clarification, not a change in the evidence or acceptance standard.'
    PROMPTS[VERSION]=prompt;os.environ['AEROROZUM_LABEL_PROMPT_VERSION']=VERSION
    # labelling may already be imported through an archival helper. Set its
    # explicit active prompt before constructing this isolated recovery worker.
    import aerorozum.labelling as labelling
    labelling.PROMPTS[VERSION]=prompt;labelling.PROMPT=prompt
    document=dict(created_at=utcnow(),images=rows,missing=[],unique_eligible_available=len(rows),
        parent_label_folders=[str(p) for p in folders],prompt_version=VERSION,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),maximum_additional_attempts_per_frame=1,
        same_original_pixels=True,threshold_unchanged=.6,test_accessed=False,original_failures_preserved=True)
    document['sha256']=digest(document);save(manifest,document)
    worker.OUTPUT=out;asyncio.run(worker.run(VERSION,document))


if __name__=='__main__':main()
