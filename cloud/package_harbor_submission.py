"""Create a portable judging bundle, with checksummed assets and real run evidence.

Only the allow-listed task tree is copied. Credentials, weights, flight videos,
agent caches and host environment files never enter the archive. Original full
Harbor jobs remain in private storage; the public event log omits reasoning.
"""
import argparse
import hashlib
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
    counts=dict(events=0,tools=0,omitted_reasoning_events=0)
    with destination.open('w') as stream:
        for line in source.read_text(errors='replace').splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if not isinstance(event,dict):continue
            if event.get('type')=='reasoning' or event.get('part',{}).get('type')=='reasoning':
                counts['omitted_reasoning_events']+=1;continue
            counts['events']+=1
            counts['tools']+=int(event.get('part',{}).get('type')=='tool')
            stream.write(json.dumps(event)+'\n')
    return counts


def package(task,jobs,output,secrets=()):
    task=Path(task).resolve();output=Path(output).resolve()
    for required in ('instruction.md','task.toml','environment/Dockerfile','tests/test.sh','solution/solve.sh',
                     'environment/backend/runtime/scene_factory.py','environment/backend/assets/provenance/response-kernels.json'):
        if not (task/required).is_file():raise ValueError('Unstaged task: '+required)
    if output.exists():raise ValueError('Refusing to overwrite a prior submission archive')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='harbor-submission-',dir=output.parent) as temp:
        root=Path(temp)/'aerorozum-submission';root.mkdir();destination=root/'thermal-investigation'
        # Avoid copying generated runtime captures or a forgotten credential file.
        def ignore(directory,names):
            return [name for name in names if name in ('__pycache__','.pytest_cache','jobs','media','cache','logs','evidence')
                    or name.startswith('.env') or name.endswith(('.pyc','.key','.pem','.safetensors'))]
        for path in task.rglob('*'):
            if path.is_symlink():raise ValueError('Task contains a symlink; inspect before packaging: '+str(path))
        shutil.copytree(task,destination,ignore=ignore,symlinks=False)
        shutil.copyfile(Path(__file__).with_name('run_harbor_trial.py'),root/'run_harbor_trial.py')
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
                (agent/'export.json').write_text(json.dumps(dict(**counts,original_log_sha256=sha(logs[0]) if logs else None,
                    note='Structured OpenCode events; reasoning events omitted. Unmodified original retained privately.'),indent=2))
                details=json.loads((verifier/'details.json').read_text()) if (verifier/'details.json').exists() else {}
                summaries.append(dict(job=job.name,trial=source.name,reward=reward,agent=result['config']['agent']['name'],
                    model=result['config']['agent'].get('model_name'),exception_type=(result.get('exception_info') or {}).get('exception_type'),
                    public_tool_events=counts.get('tools',0),completed_mission=details.get('completed'),
                    task_checksum=result.get('task_checksum')))
        clean=any(s['agent']=='opencode' and s['public_tool_events']>0 and s['exception_type'] is None for s in summaries)
        metadata=dict(schema_version=1,runs=summaries,clean_opencode_playthrough=clean,
                      frontier_superiority_verified=False,contains_base_model_weights=False,
                      contains_original_flight_recordings=False)
        (root/'submission.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (root/'README.md').write_text('''# AeroRozum: thermal investigation under a flight budget

The runnable Harbor task is `thermal-investigation/`; its README explains the
mission, difficulty, reward and scope. Scene assets and runtime sources are
included: no access to the team's VM or private bucket is needed to build it.
Use Harbor 0.26.0, OpenCode 1.19.5, Docker and Compose; model credentials are
supplied by the evaluator. See the task README for the tested run command.

`run-evidence/` contains actual Harbor results, scalar rewards, authoritative
ledgers and camera evidence. Public OpenCode event logs retain tool actions,
feedback and final text; reasoning events are omitted explicitly. Original
complete jobs are preserved privately. `submission.json` distinguishes clean
model playthroughs from errored development attempts. Scores from differing
task checksums are not a paired model comparison. No frontier-superiority or
real mine-detection claim is made by this bundle.

`SHA256SUMS` covers every other bundled file. Verify with `sha256sum -c SHA256SUMS`.
''')
        # Exact private values plus common credential signatures are checked
        # before writing an archive. Never include the matching secret in errors.
        patterns=[re.compile(rb'AIza[0-9A-Za-z_-]{25,}'),re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
        for path in root.rglob('*'):
            if not path.is_file():continue
            if path.suffix in ('.png','.jpg','.jpeg','.npy','.npz'):continue
            content=path.read_bytes()
            if any(value and value.encode() in content for value in secrets) or any(p.search(content) for p in patterns):
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
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--job',action='append',default=[]);p.add_argument('--output',required=True);p.add_argument('--key-file');a=p.parse_args()
    secret=Path(a.key_file).read_text().strip() if a.key_file else ''
    package(a.task,a.job,a.output,[secret])
