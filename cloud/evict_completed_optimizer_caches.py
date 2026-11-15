"""Evict only completed, remotely byte-verified temporary trainer checkpoints.

Final adapters, active namespaces and original data are preserved. Every file
must already have a matching GCS server digest, with a durable restore manifest
uploaded before removal. The archive uses no remote-delete synchronization.
"""
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

ROOT=Path('/opt/aerorozum');ART=ROOT/'artifacts';STUDY=ART/'overnight-20261115T000942'
BUCKET='aerorozum-thermal-rl-avid-relic-509408-e4';PREFIX='research/overnight-20261115/'


def main():
    from aerorozum.common import save,utcnow,digest
    from cloud.airborne_worker import metadata_record_store
    receipt_path=STUDY/'completed-optimizer-cache-eviction-v1.json'
    if receipt_path.exists():raise ValueError('A cleanup receipt exists; review it before any further deletion')
    ledger=json.loads((STUDY/'terminal-archive-ledger.json').read_text())['completed']
    store=metadata_record_store(BUCKET,prefix=PREFIX+'checkpoint-restore-manifests')
    selected=[]
    for specfile in (ART/'postsubmission-visual-replication-v1').glob('*/specification.json'):
        state=json.loads((specfile.parent/'status.json').read_text())
        if state.get('status')!='completed':continue
        spec=json.loads(specfile.read_text());namespace=spec['namespace'];key='training/'+namespace
        if not namespace.startswith('postsubmission-visual-replication-v1/') or '..' in Path(namespace).parts:
            raise ValueError('Eviction is restricted to the completed original replication cohort')
        if key not in ledger:continue
        expected='gs://'+BUCKET+'/'+PREFIX+key
        if ledger[key]['uri'].rstrip('/')!=expected:raise ValueError('Unexpected immutable training archive location')
        folder=ART/'training'/namespace
        for checkpoint in folder.rglob('checkpoint-*'):
            if not checkpoint.is_dir() or checkpoint.is_symlink() or not re.fullmatch(r'checkpoint-\d+',checkpoint.name):continue
            if not checkpoint.resolve().is_relative_to((ART/'training/postsubmission-visual-replication-v1').resolve()):
                raise ValueError('Checkpoint escaped its completed training namespace')
            if not all((checkpoint/name).is_file() for name in ['optimizer.pt','trainer_state.json']):continue
            entries=list(checkpoint.rglob('*'))
            if any(p.is_symlink() for p in entries):raise ValueError('Do not evict a checkpoint containing links')
            files=[p for p in entries if p.is_file()]
            if not files:continue
            selected.append(dict(path=str(checkpoint),namespace=namespace,files=[dict(path=str(p),
                object=PREFIX+key+'/'+p.relative_to(folder).as_posix(),size=p.stat().st_size,
                inode=p.stat().st_ino,mtime_ns=p.stat().st_mtime_ns) for p in files]))
    def verify(record):
        path=Path(record['path'])
        with path.open('rb') as stream:md5=base64.b64encode(hashlib.file_digest(stream,'md5').digest()).decode()
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        url='https://storage.googleapis.com/storage/v1/b/'+BUCKET+'/o/'+urllib.parse.quote(record['object'],safe='')
        request=urllib.request.Request(url+'?fields=size,md5Hash,generation',headers=store._headers())
        with urllib.request.urlopen(request,timeout=60) as response:remote=json.load(response)
        if remote.get('md5Hash')!=md5 or int(remote.get('size',-1))!=record['size']:raise ValueError('Cloud copy does not match: '+record['path'])
        return dict(record,sha256=sha,md5_base64=md5,gcs_generation=remote['generation'],
                    restore_uri='gs://'+BUCKET+'/'+record['object']+'#'+remote['generation'])
    for checkpoint in selected:
        with ThreadPoolExecutor(max_workers=4) as pool:checkpoint['files']=list(pool.map(verify,checkpoint['files']))
    receipt=dict(at=utcnow(),status='all_files_verified_before_eviction',checkpoints=selected,
        final_adapters_preserved=True,active_training_preserved=True,original_data_preserved=True,
        archive_remote_delete_enabled=False,cloud_server_digest='MD5',local_content_digest='SHA256',
        restore_method='Copy each recorded generation-pinned restore_uri back to its path and verify SHA256',removed_bytes=0)
    save(receipt_path,receipt)
    receipt['pre_eviction_restore_manifest_uri']=store.put(digest(receipt),receipt,'progress')
    save(receipt_path,receipt)
    for checkpoint in selected:
        folder=Path(checkpoint['path']);current={str(p) for p in folder.rglob('*') if p.is_file()}
        if current!={r['path'] for r in checkpoint['files']}:raise ValueError('Checkpoint file set changed after verification')
        for record in checkpoint['files']:
            path=Path(record['path']);stat=path.stat()
            if path.is_symlink() or (stat.st_ino,stat.st_mtime_ns,stat.st_size)!=(record['inode'],record['mtime_ns'],record['size']):raise ValueError('Completed checkpoint changed before eviction')
        shutil.rmtree(folder);checkpoint['evicted']=True
        receipt['removed_bytes']+=sum(r['size'] for r in checkpoint['files']);save(receipt_path,receipt)
    receipt.update(status='completed',finished_at=utcnow(),cloud_disk_free_gib=shutil.disk_usage(ROOT).free/1024**3)
    save(receipt_path,receipt);uri=store.put(digest(receipt),receipt,'progress')
    print(json.dumps(dict(status=receipt['status'],evicted_completed_checkpoint_directories=len(selected),
        removed_gib=receipt['removed_bytes']/1024**3,cloud_disk_free_gib=receipt['cloud_disk_free_gib'],
        final_receipt_uri=uri,final_adapters_and_original_data_preserved=True)))


if __name__=='__main__':main()
