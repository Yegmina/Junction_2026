"""Create a portable judging bundle, with checksummed assets and real run evidence.

Only the allow-listed task tree is copied. Credentials, weights, flight videos,
agent caches and host environment files never enter the archive. Original full
Harbor jobs remain in private storage; the public event log omits reasoning.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):h.update(block)
    return h.hexdigest()


def public_events(source,destination):
    counts=dict(events=0,tools=0,omitted_reasoning_events=0,externalized_image_payloads=0,image_artifacts=[])
    images={}
    def externalize(value):
        if isinstance(value,str) and value.startswith('data:image/') and ';base64,' in value:
            header,encoded=value.split(';base64,',1)
            suffix={'data:image/png':'.png','data:image/jpeg':'.jpg','data:image/webp':'.webp'}.get(header)
            if not suffix:raise ValueError('Unknown inline image format in public event log')
            payload=base64.b64decode(encoded,validate=True)
            signatures={'.png':payload.startswith(b'\x89PNG\r\n\x1a\n'),'.jpg':payload.startswith(b'\xff\xd8'),
                        '.webp':payload.startswith(b'RIFF') and payload[8:12]==b'WEBP'}
            if not signatures[suffix]:raise ValueError('Inline image payload does not match its declared format')
            identity=hashlib.sha256(payload).hexdigest();name='inline-images/'+identity+suffix
            path=destination.parent/name;path.parent.mkdir(exist_ok=True);path.write_bytes(payload)
            counts['externalized_image_payloads']+=1
            images[name]=dict(path=name,sha256=identity,bytes=len(payload),
                scope='Image attached in an agent tool log; may be a generated crop/mosaic, not a direct camera capture')
            return name
        if isinstance(value,dict):return {k:externalize(v) for k,v in value.items()}
        if isinstance(value,list):return [externalize(v) for v in value]
        return value
    with destination.open('w') as stream:
        for line in source.read_text(errors='replace').splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if not isinstance(event,dict):continue
            if event.get('type')=='reasoning' or event.get('part',{}).get('type')=='reasoning':
                counts['omitted_reasoning_events']+=1;continue
            counts['events']+=1
            counts['tools']+=int(event.get('part',{}).get('type')=='tool')
            stream.write(json.dumps(externalize(event))+'\n')
    counts['image_artifacts']=list(images.values())
    return counts


def package(task,jobs,output,secrets=(),audits=(),task_folder='thermal-investigation'):
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,79}',task_folder):
        raise ValueError('Task folder must be a single safe directory name')
    task=Path(task).resolve();output=Path(output).resolve()
    for required in ('instruction.md','task.toml','environment/Dockerfile','tests/test.sh','solution/solve.sh',
                     'environment/backend/runtime/scene_factory.py','environment/backend/assets/provenance/response-kernels.json'):
        if not (task/required).is_file():raise ValueError('Unstaged task: '+required)
    if output.exists():raise ValueError('Refusing to overwrite a prior submission archive')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='harbor-submission-',dir=output.parent) as temp:
        root=Path(temp)/'aerorozum-submission';root.mkdir();destination=root/task_folder
        # Preserve the immutable task byte-for-byte, including compiled modules
        # that Docker COPY can reach. Reject private/unstaged input rather than
        # silently changing the evaluated environment while packaging it.
        for path in task.rglob('*'):
            if path.is_symlink():raise ValueError('Task contains a symlink; inspect before packaging: '+str(path))
            if path.is_file() and (path.name.startswith('.env') or path.suffix in
                    ('.key','.pem','.safetensors','.mp4','.avi','.mov','.mkv','.sqlite','.db')):
                raise ValueError('Task contains a private or unstaged artifact: '+str(path.relative_to(task)))
        shutil.copytree(task,destination,symlinks=False)
        shutil.copyfile(Path(__file__).with_name('run_harbor_trial.py'),root/'run_harbor_trial.py')
        shutil.copyfile(Path(__file__).with_name('harbor_controls.py'),root/'harbor_controls.py')
        shutil.copyfile(Path(__file__).with_name('audit_harbor_contract.py'),root/'audit_harbor_contract.py')
        summaries=[]
        for job in map(Path,jobs):
            for trial in sorted(job.glob('*/result.json')):
                source=trial.parent;result=json.loads(trial.read_text());verifier=source/'verifier'
                if not (verifier/'reward.txt').is_file():continue
                reward=float((verifier/'reward.txt').read_text())
                if not math.isfinite(reward) or not 0<=reward<=1:raise ValueError('Invalid saved reward')
                recorded=(result.get('verifier_result') or {}).get('rewards',{}).get('reward')
                if recorded!=reward:raise ValueError('Harbor result and reward.txt disagree')
                target=root/'run-evidence'/job.name/source.name;target.mkdir(parents=True)
                shutil.copyfile(trial,target/'result.json')
                shutil.copytree(verifier,target/'verifier')
                agent=target/'agent';agent.mkdir()
                logs=list(source.rglob('opencode.txt'));counts={}
                if logs:counts=public_events(logs[0],agent/'opencode-public.jsonl')
                for filename in ('reward-file-spoof.json','evidence-tamper.json','truth-probe.json'):
                    control=source/'agent'/filename
                    if control.is_file():
                        if control.stat().st_size>1000000:raise ValueError('Unexpectedly large control receipt')
                        json.loads(control.read_text())
                        shutil.copyfile(control,agent/filename)
                (agent/'export.json').write_text(json.dumps(dict(**counts,original_log_sha256=sha(logs[0]) if logs else None,
                    note='Structured OpenCode events; reasoning events omitted. Unmodified original retained privately.'),indent=2))
                details=json.loads((verifier/'details.json').read_text()) if (verifier/'details.json').exists() else {}
                instance=json.loads((verifier/'ledger.json').read_text()).get('instance') if (verifier/'ledger.json').is_file() else None
                summaries.append(dict(job=job.name,trial=source.name,reward=reward,agent=result['config']['agent']['name'],
                    model=result['config']['agent'].get('model_name'),exception_type=(result.get('exception_info') or {}).get('exception_type'),
                    public_tool_events=counts.get('tools',0),completed_mission=details.get('completed'),
                    flight_elapsed_s=details.get('flight_elapsed_s'),budget_s=details.get('budget_s'),
                    observations=details.get('observations'),actions=details.get('actions'),
                    true_positives=details.get('true_positives'),false_positives=details.get('false_positives'),
                    task_checksum=result.get('task_checksum'),result_sha256=sha(trial),instance=instance))
        audit_records=[]
        for source in map(Path,audits):
            audit=json.loads(source.read_text())
            if any(audit.get(k) is not True for k in ('pair_comparable','contract_identical','native_stack_identical','instance_identical')):
                raise ValueError('Comparison audit has not verified a clean comparable pair')
            spec=importlib.util.spec_from_file_location('bundle_contract',Path(__file__).with_name('audit_harbor_contract.py'))
            contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
            if contract.fingerprint(destination)['sha256']!=audit['right_contract_sha256'] or audit['left_contract_sha256']!=audit['right_contract_sha256']:
                raise ValueError('Comparison audit does not describe the bundled task')
            for side in ('left','right'):
                trial=audit['trials'][side]
                if not any(r['job']==trial['job'] and r['result_sha256']==trial['result_sha256']
                           and r['instance']==trial['instance'] and r['exception_type'] is None for r in summaries):
                    raise ValueError('Comparison audit trial is absent or differs from bundled result')
            target=root/'comparison-audits'/source.name;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source,target);audit_records.append(dict(path=str(target.relative_to(root)),sha256=sha(target)))
        clean=any(s['agent']=='opencode' and s['public_tool_events']>0 and s['exception_type'] is None for s in summaries)
        metadata=dict(schema_version=1,runs=summaries,clean_opencode_playthrough=clean,
                      comparison_audits=audit_records,
                      frontier_superiority_verified=False,contains_base_model_weights=False,
                      contains_original_flight_recordings=False)
        (root/'submission.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (root/'README.md').write_text('''# AeroRozum: thermal investigation under a flight budget

The runnable Harbor task is `thermal-investigation/`; its README explains the
mission, difficulty, reward and scope. Scene assets and runtime sources are
included: no access to the team's VM or private bucket is needed to build it.
Use Harbor 0.26.0, OpenCode 1.19.5, Docker and Compose; model credentials are
supplied by the evaluator. See the task README for the tested run command.

From this extracted directory, run the reference with:
`harbor run -p ./thermal-investigation -a oracle`.
For a model, use Harbor's usual `-a opencode -m <provider/model>` arguments.
The included Google runner pins the tested harness configuration, for example:
`python run_harbor_trial.py --task ./thermal-investigation --job-name first-run --key-file /private/key-file --output-tokens 32768`.
Keep credentials outside the task. The task uses a separate CPU-rendered Gazebo
simulator container configured for up to 8 CPU cores and 12 GiB, in addition to
the agent container. A GPU is not required to run the environment.

Controls use the same task without replacing its reference solution:
`PYTHONPATH=. harbor run -p ./thermal-investigation -a harbor_controls:StopEarlyAgent`
and `PYTHONPATH=. harbor run -p ./thermal-investigation -a harbor_controls:RewardFileSpoofAgent`.
Use `-a nop` for the do-nothing control. All should score zero.

`run-evidence/` contains actual Harbor results, scalar rewards, authoritative
ledgers and camera evidence. Public OpenCode event logs retain tool actions,
feedback and final text; reasoning events are omitted explicitly. Original
complete jobs are preserved privately. `submission.json` distinguishes clean
model playthroughs from errored development attempts. Full Harbor task hashes
remain in each result. A comparison across different hashes requires an
explicit audit of identical instructions, environment/assets, verifier,
resources, native package stack and sealed instance. Bundled comparison audits
cover only reference-implementation revisions; they never equate different
grading rules or scenes. No frontier-superiority or real mine-detection claim
is made by this bundle.

`SHA256SUMS` covers every other bundled file. Verify with `sha256sum -c SHA256SUMS`.
'''.replace('thermal-investigation',task_folder))
        with (root/'README.md').open('a') as stream:
            if task_folder=='thermal-investigation-budgeted':
                stream.write('\n## Why this environment matters\n\nA wide scan is cheap; a supported claim needs a close inspection. The agent can prepare code before takeoff and analyze saved evidence after landing, but the aircraft keeps flying during inference. It must decide which uncertain patches deserve more time and reserve the trip home. Good detections do not excuse missing the flight deadline. The reference uses public images and telemetry, with no target locations or counts.\n')
            stream.write('\n## Included actual runs\n\n| Job | Reward | Process | Mission | Flight / budget | Captures |\n|---|---:|---|---|---|---:|\n')
            for run in summaries:
                timing='—' if run['flight_elapsed_s'] is None else f"{run['flight_elapsed_s']:.3f} / {run['budget_s']:g}s"
                stream.write(f"| {run['job']} | {run['reward']:.6f} | {run['exception_type'] or 'clean'} | {'complete' if run['completed_mission'] else 'incomplete'} | {timing} | {run['observations'] if run['observations'] is not None else '—'} |\n")
            stream.write('\nRun names distinguish output allowances and seeds. A clean process exit can still describe an incomplete mission. Short-cap preflight truncations and process errors are not treated as evidence of flight-planning failure. Full configurations, scores, ledgers and public tool logs are included; the table does not replace them.\n')
        # Exact private values plus common credential signatures are checked
        # before writing an archive. Never include the matching secret in errors.
        patterns=[re.compile(rb'AIza[0-9A-Za-z_-]{25,}'),re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
        for path in root.rglob('*'):
            if not path.is_file():continue
            content=path.read_bytes()
            exact_secret=any(value and value.encode() in content for value in secrets)
            generic_signature=path.suffix not in ('.png','.jpg','.jpeg','.webp','.npy','.npz') and any(p.search(content) for p in patterns)
            if exact_secret or generic_signature:
                raise ValueError('Credential scan rejected bundled file: '+str(path.relative_to(root)))
        paths=sorted(p for p in root.rglob('*') if p.is_file())
        (root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in paths))
        if output.suffix=='.zip':
            with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                for path in sorted(root.rglob('*')):
                    if path.is_file():archive.write(path,arcname=str(path.relative_to(root.parent)))
        else:
            with tarfile.open(output,'w:gz') as archive:archive.add(root,arcname=root.name)
    receipt=dict(path=str(output),sha256=sha(output),bytes=output.stat().st_size,**metadata)
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--task-folder',default='thermal-investigation');p.add_argument('--job',action='append',default=[]);p.add_argument('--comparison-audit',action='append',default=[]);p.add_argument('--output',required=True);p.add_argument('--key-file');a=p.parse_args()
    secret=Path(a.key_file).read_text().strip() if a.key_file else ''
    package(a.task,a.job,a.output,[secret],a.comparison_audit,a.task_folder)
