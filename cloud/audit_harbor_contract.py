"""Audit reference-only task revisions without conflating different evaluators.

Every instruction, environment asset/runtime, verifier byte and file mode is
covered. Only the private reference implementation, README and task version
label are outside this contract. Full Harbor checksums remain in run results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tomllib


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fingerprint(task):
    task=Path(task);configuration=tomllib.loads((task/'task.toml').read_text())
    configuration.get('task',{}).pop('version',None)
    for name in ['instruction.md','environment/Dockerfile','tests/test.sh']:
        if not (task/name).is_file():raise ValueError('Missing contract input: '+name)
    files={}
    for path in [task/'instruction.md',*(task/'environment').rglob('*'),*(task/'tests').rglob('*')]:
        if path.is_symlink():raise ValueError('Contract contains a symlink')
        if path.is_file():files[str(path.relative_to(task))]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            mode=stat.S_IMODE(path.stat().st_mode),bytes=path.stat().st_size)
    record=dict(configuration=configuration,files=files)
    return dict(sha256=digest(record),**record)


def image_identity(name):
    record=json.loads(subprocess.check_output(['docker','image','inspect',name],text=True))[0]
    config={k:record['Config'].get(k) for k in ['Env','Entrypoint','Cmd','WorkingDir','User','ExposedPorts']}
    packages=subprocess.check_output(['docker','run','--rm','--entrypoint','dpkg-query',name,'-W',
        '-f=${binary:Package} ${Version} ${Architecture}\n'])
    return dict(image_id=record['Id'],architecture=record['Architecture'],os=record['Os'],
        configuration_sha256=digest(config),packages_sha256=hashlib.sha256(packages).hexdigest(),
        filesystem_layers=record['RootFS']['Layers'])


def terminal_trial(job):
    paths=list(Path(job).glob('*/result.json'))
    if len(paths)!=1:raise ValueError('Expected one completed trial in '+str(job))
    path=paths[0];result=json.loads(path.read_text())
    if not result.get('finished_at'):raise ValueError('Trial is not terminal')
    ledger=path.parent/'verifier/ledger.json'
    if not ledger.is_file():raise ValueError('Sealed instance ledger is missing')
    return dict(job=Path(job).name,task_checksum=result.get('task_checksum'),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rewards=(result.get('verifier_result') or {}).get('rewards'),
        exception_type=(result.get('exception_info') or {}).get('exception_type'),
        instance=json.loads(ledger.read_text())['instance'])


def audit(left,right,left_image=None,right_image=None,left_job=None,right_job=None):
    a,b=fingerprint(left),fingerprint(right)
    def tree(root):
        root=Path(root)
        return {str(p.relative_to(root)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mode=stat.S_IMODE(p.stat().st_mode))
                for p in root.rglob('*') if p.is_file()}
    ta,tb=tree(left),tree(right)
    differences=[p for p in sorted(ta.keys()|tb.keys()) if ta.get(p)!=tb.get(p)]
    unexpected=[p for p in differences if p not in ('README.md','task.toml') and not p.startswith('solution/')
                and p not in a['files'] and p not in b['files']]
    report=dict(scope='Reference-only revision audit; does not equate different grading rules or physical instances.',
        left_contract_sha256=a['sha256'],right_contract_sha256=b['sha256'],
        contract_identical=a==b and not unexpected,configuration_identical=a['configuration']==b['configuration'],
        differing_contract_files=[p for p in sorted(a['files'].keys()|b['files'].keys()) if a['files'].get(p)!=b['files'].get(p)],
        differing_task_files=differences,unexpected_root_changes=unexpected,contract_files=len(a['files']),
        contract_manifest=dict(left=a,right=b),
        excluded=['solution/ (private reference)','README.md','task.version label'],pair_comparable=None)
    if bool(left_image)!=bool(right_image) or bool(left_job)!=bool(right_job):raise ValueError('Provide both sides of an image/job pair')
    stack_equal=None
    if left_image:
        ia,ib=image_identity(left_image),image_identity(right_image)
        stack_equal=all(ia[k]==ib[k] for k in ['architecture','os','configuration_sha256','packages_sha256'])
        report.update(native_images=dict(left=ia,right=ib),native_stack_identical=stack_equal)
    if left_job:
        ja,jb=terminal_trial(left_job),terminal_trial(right_job)
        if not ja['instance'] or not jb['instance']:raise ValueError('Actual instance identity is empty')
        same_instance=ja['instance']==jb['instance']
        report.update(trials=dict(left=ja,right=jb),instance_identical=same_instance,
            pair_comparable=report['contract_identical'] and stack_equal is True and same_instance
                and ja['exception_type'] is None and jb['exception_type'] is None)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['left','right','output']:p.add_argument('--'+name,required=True)
    for name in ['left-image','right-image','left-job','right-job']:p.add_argument('--'+name)
    args=p.parse_args();result=audit(args.left,args.right,args.left_image,args.right_image,args.left_job,args.right_job)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('native_images','trials','contract_manifest')}))
