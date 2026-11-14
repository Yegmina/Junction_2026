"""Export terminal measurements without copying prompts, credentials or thoughts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

FIELDS=['status','examples','tp','fp','fn','micro_f2','valid_json_fraction',
        'negative_false_alarm_rate','protocol_hash','decoding_protocol']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root,output):
    root=Path(root);trials=[];frontier=[];other=[]
    for file in sorted((root/'harbor/jobs').glob('*/*/result.json')):
        result=json.loads(file.read_text())
        if not result.get('finished_at'):continue
        details=file.parent/'verifier/details.json';ledger=file.parent/'verifier/ledger.json'
        trials.append(dict(job=file.parents[1].name,trial=file.parent.name,
            task_checksum=result.get('task_checksum'),model=result.get('config',{}).get('agent',{}).get('model_name'),
            rewards=(result.get('verifier_result') or {}).get('rewards'),
            exception_type=(result.get('exception_info') or {}).get('exception_type'),finished_at=result['finished_at'],
            details=json.loads(details.read_text()) if details.is_file() else None,
            instance=json.loads(ledger.read_text()).get('instance') if ledger.is_file() else None,
            result_sha256=sha(file)))
    for file in sorted((root/'artifacts/frontier-development-v1').glob('*/summary.json')):
        data=json.loads(file.read_text())
        if data.get('status')!='completed':continue
        frontier.append(dict(model=data['model'],**{k:data[k] for k in FIELDS if k in data},
            protocol=data['protocol'],source_sha256=sha(file)))
    for family in ['submission-reward-curriculum-v1','submission-visual-projection-v2','submission-refinement-v3',
                   'submission-visual-continuation-v1','submission-geometric-augmentation-v1']:
        for file in sorted((root/'artifacts'/family).rglob('result.json')):
            data=json.loads(file.read_text())
            if data.get('status')!='completed':continue
            result=data.get('results',data);evaluations=dict(result.get('evaluation',{}))
            for stage in ('sft','grpo'):
                if stage+'_evaluation' in result:evaluations[stage]=result[stage+'_evaluation']
            if 'micro_f2' in data:evaluations['base_control']=data
            other.append(dict(family=family,arm=file.parent.name,
                evaluations={name:{k:value[k] for k in FIELDS if k in value} for name,value in evaluations.items() if isinstance(value,dict)},
                training={stage:{key:result[stage].get(key) for key in ['optimizer_updates','requested_optimizer_updates',
                    'early_stopped','nonzero_gradient_checks','lora_scope','augmentation','peak_gpu_memory_bytes']}
                    for stage in ('sft','grpo') if isinstance(result.get(stage),dict)},
                source_sha256=sha(file)))
    record=dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Development measurements; not sealed-test results. Harbor mission scores and image-level F2 are distinct. Errored trials are explicitly marked.',
        trials=trials,frontier_perception=frontier,other_training=other)
    Path(output).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(terminal_trials=len(trials),frontier_controls=len(frontier),training_arms=len(other))))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/opt/aerorozum');parser.add_argument('--output',required=True)
    args=parser.parse_args();export(args.root,args.output)
