"""Independently hash, extract and smoke-test a prepared Harbor ZIP.

No private data mount is supplied. The extracted task must reproduce the
frozen source's evaluation contract and full Harbor task checksum.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import stat
import subprocess
import zipfile
from audit_harbor_contract import fingerprint


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):h.update(block)
    return h.hexdigest()


def extract_verified(archive,destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('Refusing to overwrite an earlier extraction')
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        prefix='aerorozum-submission/';names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
        manifest=z.read(prefix+'SHA256SUMS').decode().splitlines();expected={}
        for line in manifest:
            checksum,name=line.split('  ',1)
            if name in expected:raise ValueError('Duplicate manifest entry')
            expected[name]=checksum
        if set(names)!={prefix+name for name in expected}|{prefix+'SHA256SUMS'}:
            raise ValueError('Archive contains unhashed or missing files')
        for info in z.infolist():
            name=PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe archive path or symlink')
            if info.flag_bits&1:raise ValueError('Encrypted member is not independently reviewable')
            relative=info.filename.removeprefix(prefix)
            if relative in expected:
                h=hashlib.sha256()
                with z.open(info) as stream:
                    for block in iter(lambda:stream.read(8388608),b''):h.update(block)
                if h.hexdigest()!=expected[relative]:raise ValueError('Bundled file hash mismatch')
            z.extract(info,destination)
            mode=(info.external_attr>>16)&0o777
            if mode:(destination/info.filename).chmod(mode)
    return len(expected)


def verify(args):
    count=extract_verified(args.archive,args.destination)
    task=Path(args.destination)/'aerorozum-submission'/args.task_folder
    if fingerprint(task)!=fingerprint(Path(args.source_task)):
        raise ValueError('Extracted evaluation contract differs from the frozen source')
    reference=json.loads(Path(args.reference_result).read_text())
    env=os.environ.copy()
    for key in ['AEROROZUM_TASK_PHASE','AEROROZUM_TASK_BUDGET']:env.pop(key,None)
    env['AEROROZUM_TASK_SEED']=str(args.seed)
    subprocess.run([args.harbor,'run','-p',str(task),'-a','nop','-o',args.jobs_dir,'--job-name',args.job,'-n','1'],env=env,check=True)
    results=list((Path(args.jobs_dir)/args.job).glob('*/result.json'))
    if len(results)!=1:raise ValueError('Expected one portability trial')
    result=json.loads(results[0].read_text());ledger=json.loads((results[0].parent/'verifier/ledger.json').read_text())
    if not result.get('finished_at') or result.get('exception_info') or result['verifier_result']['rewards']['reward']!=0:
        raise ValueError('Portability smoke trial did not finish cleanly at zero')
    if result['task_checksum']!=reference['task_checksum']:raise ValueError('Full Harbor task checksum differs')
    instance=ledger['instance']
    if instance['seed']!=args.seed or instance['budget_s']!=args.expected_budget:
        raise ValueError('Extracted task defaults or fresh instance differ')
    if instance['thermal_condition']['phase']!=args.expected_phase or len(instance['field_hashes'])!=4:
        raise ValueError('Extracted factory did not regenerate the expected physical fields')
    receipt=dict(archive_sha256=sha(args.archive),verified_files=count,
        extracted_contract_identical=True,full_task_checksum_identical=True,task_checksum=result['task_checksum'],
        smoke_job=args.job,smoke_reward=0,smoke_exception=None,smoke_result_sha256=sha(results[0]),
        fresh_instance=instance,scope='Extracted bundle ran with existing Docker layer caches; no private source/bucket mount. Factory regenerated all four fields.')
    output=Path(args.output)
    with output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['archive','destination','task-folder','source-task','reference-result','harbor','jobs-dir','job','output']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=930117);p.add_argument('--expected-budget',type=float,default=75)
    p.add_argument('--expected-phase',default='dawn');verify(p.parse_args())
