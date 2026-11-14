#!/usr/bin/env python3
"""Stage exact frozen pending pixels and audit documents; no inference client."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
from aerorozum.common import data_root,save,utcnow

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);a=p.parse_args();root=data_root().resolve();manifest=a.manifest.resolve();doc=json.loads(manifest.read_text())
    if doc.get('subset')!='pending_only' or doc.get('policy',{}).get('version')!='airborne-only-v2-primary-state':raise ValueError('Corrected frozen pending-only snapshot required')
    files={manifest.relative_to(root).as_posix(),doc['full_manifest'],doc['eligibility_manifest']}
    for row in doc['images']:
        path=(root/row['image']).resolve()
        if not path.is_relative_to(root) or sha(path)!=row['sha256']:raise ValueError('Exact frozen input pixels differ')
        files.add(path.relative_to(root).as_posix())
    for name in ['labelling/airborne-drain-receipt.json','pipeline/leases/prompt-comparison.json','source-videos/labelling-sources.json']:
        if (root/name).is_file():files.add(name)
    folder=root/'cloud/label-worker-migration';folder.mkdir(parents=True,exist_ok=True);bundle=folder/('airborne-pending-'+manifest.parent.name+'.tar.gz')
    with tarfile.open(bundle,'w:gz',compresslevel=1) as archive:
        for name in sorted(files):
            path=(root/name).resolve()
            if not path.is_relative_to(root):raise ValueError('Unsafe input archive path')
            archive.add(path,arcname=name,recursive=False)
    sdk='/home/yehormsi/.local/google-cloud-cli-install/google-cloud-sdk/bin/gcloud';uri='gs://aerorozum-thermal-rl-avid-relic-509408-e4/label-worker-migration/'+manifest.parent.name+'/'+bundle.name
    subprocess.run([sdk,'storage','cp',str(bundle),uri,'--quiet'],check=True)
    metadata=json.loads(subprocess.check_output([sdk,'storage','objects','describe',uri,'--format=json'],text=True))
    receipt={'status':'staged_exact_frozen_inputs','at':utcnow(),'uri':uri,'generation':str(metadata['generation']),'sha256':sha(bundle),'bytes':bundle.stat().st_size,'files':len(files),'rows':len(doc['images']),'manifest':manifest.relative_to(root).as_posix(),'manifest_sha256':sha(manifest),'eligibility_manifest':doc['eligibility_manifest'],'full_manifest':doc['full_manifest'],'full_manifest_sha256':doc['full_manifest_sha256'],'paid_calls':0}
    save(folder/'input-staging.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
