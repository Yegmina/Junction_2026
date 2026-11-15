"""Freeze only newly qualified, nonconflicting airborne inputs; no inference."""
from pathlib import Path
import hashlib
import json


def main():
    from aerorozum.requalified_frames import stage
    from aerorozum.common import save,utcnow,digest
    root=Path('/opt/aerorozum');oldroot=root/'airborne-labeler/artifacts/pipeline/inputs/airborne-snapshots'
    newroot=root/'airborne-requalification-v1/artifacts'
    destination=root/'airborne-requalification-v1/frozen-new-high-inputs.json'
    if destination.exists():raise ValueError('A frozen requalification already exists; preserve its identity')
    paths=list(oldroot.glob('*/airborne-eligibility.json'));old=[json.loads(p.read_text()) for p in paths]
    audit=json.loads((newroot/'airborne-eligibility.json').read_text())
    eligible=json.loads((newroot/'airborne-eligible.json').read_text())['images']
    for row in eligible:row['image']=str(newroot/row['image'])
    previous=json.loads((root/'airborne-labeler/artifacts/labelling/robotics-er2-high-thinking-v1/frozen-inputs.json').read_text())
    registry_path=root/'airborne-labeler/artifacts/labelling/robotics-er2-airborne/session-splits.json'
    registry=json.loads(registry_path.read_text());sessions=registry.get('sessions',registry)
    rows,receipt=stage(old,audit,eligible,previous['images'],sessions)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt.update(at=utcnow(),original_audit_sha256={str(p):sha(p) for p in paths},
        new_audit_sha256=sha(newroot/'airborne-eligibility.json'),prepared_eligible_frames=len(eligible),
        session_registry_sha256=sha(registry_path),
        quarantined_sources=json.loads((newroot/'result.json').read_text())['quarantined_sources'])
    for row in rows:
        row['flight_requalification'].update(prior_audit_sha256=receipt['original_audit_sha256'],
            new_audit_sha256=receipt['new_audit_sha256'],session_registry_sha256=receipt['session_registry_sha256'])
        if sha(Path(row['image']))!=row['sha256']:raise ValueError('Requalified pixels changed')
    document=dict(images=rows,missing=[],unique_eligible_available=len(rows),at=utcnow(),receipt=receipt)
    document['sha256']=digest(document);save(destination,document)
    save(root/'artifacts/overnight-20261115T000942/requalification-admission.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
