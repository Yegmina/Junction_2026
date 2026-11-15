"""Export measured three-seed development results with source-file identities."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/opt/aerorozum/artifacts')
FIELDS=['micro_f2','tp','fp','fn','valid_json_fraction','negative_false_alarm_rate']


def main():
    from aerorozum.common import save,utcnow
    files={};rows=[]
    def load(relative):
        path=ROOT/relative
        files[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    for seed in [17,43,91]:
        if seed==17:
            initial=load('submission-visual-projection-v2/updates-480/result.json')
            final=load('submission-grpo-continuation-v1/updates-96/result.json')
        else:
            initial=load(f'postsubmission-visual-replication-v1/seed-{seed}/initial/result.json')
            final=load(f'postsubmission-visual-replication-v1/seed-{seed}/continuation/result.json')
        if initial['status']!='completed' or final['status']!='completed':raise ValueError('Incomplete replication')
        measured=[]
        for stage,result,key in [('480 SFT',initial,'sft_evaluation'),('480 SFT + 48 GRPO',initial,'grpo_evaluation'),('480 SFT + 144 GRPO',final,'grpo_evaluation')]:
            evaluation=result['results'][key]
            measured.append(dict(stage=stage,**{k:evaluation.get(k) for k in FIELDS}))
        rows.append(dict(seed=seed,evaluations=measured,reward_signal=final['results'].get('reward_signal',{})))
    control=load('postsubmission-visual-replication-v1/sft-control-17/initial/result.json')['results']['sft_evaluation']
    document=dict(at=utcnow(),status='completed',seeds=rows,source_sha256=files,
        additional_96_sft_seed17={k:control.get(k) for k in FIELDS},
        training_images=602,development_images=120,
        development_reference_provenance={'original_manual':30,'reviewed_ai':58,'machine_screened_zero':32},
        production_model_changed=False,test_accessed=False,automatic_promotion=False,
        interpretation='Repeated development comparison; reused development references do not establish field generalization. Additional SFT/GRPO controls match update counts, not compute. Model-zero references are not verified absence.')
    save(ROOT/'overnight-20261115T000942/visual-replications-summary.json',document)
    print(json.dumps(document,indent=2))


if __name__=='__main__':main()
