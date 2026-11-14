"""Durably mirror eligible local image arrivals before publishing their VM index."""
from __future__ import annotations
import argparse,json,shlex,subprocess,tempfile,time
from pathlib import Path
from datetime import datetime,timezone
from .common import data_root,save,utcnow,digest
from .preflight import sdk_path

def redacted_bridge_snapshot(document):
    """Test labels/pixels stay local; only their hashes can exclude duplicates."""
    result={key:document[key] for key in ('schema_version','updated_at','counts','session_splits','zone_splits','dataset_signature') if key in document}
    result['images']=[row for row in document.get('images',[]) if row.get('split') in ('train','validation')]
    result['sealed_test_image_hashes']=sorted({row['sha256'] for row in document.get('images',[]) if row.get('split') not in ('train','validation') and row.get('sha256')})
    result['transfer_scope']='Training and development only; sealed test labels, paths and source previews are excluded'
    return result


class InputBridge:
    def __init__(self,root=None,vm='aerorozum-gemma-pilot-20261114a',zone='europe-west4-a',project='avid-relic-509408-e4',remote_root='/opt/aerorozum/artifacts'):
        self.root=Path(root or data_root()).resolve();self.remote_root=remote_root;self.signature=None
        command=[sdk_path(),'compute','ssh',vm,f'--zone={zone}',f'--project={project}','--quiet','--dry-run']
        parts=shlex.split(subprocess.check_output(command,text=True).strip());self.host=parts[-1]
        self.ssh=shlex.join([part for part in parts[:-1] if part!='-t'])
        if not self.host or any(c.isspace() for c in self.host):raise ValueError('Unsafe cloud SSH destination')

    def sync(self):
        source=self.root/'pipeline/inputs/real.json'
        if not source.is_file():return {'status':'waiting','reason':'Input ingester has not published an index'}
        doc=redacted_bridge_snapshot(json.loads(source.read_text()));signature=doc.get('dataset_signature') or digest(doc)
        if signature==self.signature:return {'status':'unchanged','signature':signature}
        names=[]
        for row in doc.get('images',[]):
            if row.get('split') not in ('train','validation'):continue
            path=Path(row['image'])
            if path.is_absolute() or '..' in path.parts or '\n' in str(path):raise ValueError('Unsafe image path in arrival index')
            if not (self.root/path).resolve().is_relative_to(self.root):raise ValueError('Input symlink escapes artifact root')
            if (self.root/path).is_file():names.append(str(path))
        with tempfile.TemporaryDirectory(prefix='aerorozum-input-bridge-') as temp:
            listing=Path(temp)/'files.txt';listing.write_text('\n'.join(sorted(set(names)))+'\n')
            snapshot=Path(temp)/'real.json';snapshot.write_text(json.dumps(doc,allow_nan=False))
            common=['rsync','-a','--rsync-path=sudo rsync','-e',self.ssh]
            subprocess.run(common+[f'--files-from={listing}',str(self.root)+'/',self.host+':'+self.remote_root+'/'],check=True,timeout=600)
            # The index becomes visible only after its pixels have arrived.
            subprocess.run(common+[str(snapshot),self.host+':'+self.remote_root+'/pipeline/inputs/real.json'],check=True,timeout=60)
        self.signature=signature
        report={'status':'synced','images':len(set(names)),'signature':signature,'at':utcnow()}
        save(self.root/'pipeline/input-bridge.json',report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--once',action='store_true');p.add_argument('--poll',type=int,default=30)
    p.add_argument('--deadline',default='2026-11-15T06:30:50+00:00');a=p.parse_args();bridge=InputBridge(a.root)
    while datetime.now(timezone.utc)<datetime.fromisoformat(a.deadline).astimezone(timezone.utc):
        try:result=bridge.sync();print(json.dumps(result),flush=True)
        except Exception as exc:save(bridge.root/'pipeline/input-bridge.json',{'status':'retrying','error':str(exc)[:1000],'at':utcnow()})
        if a.once:return
        time.sleep(max(5,a.poll))

if __name__=='__main__':main()
