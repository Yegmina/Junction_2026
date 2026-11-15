"""Rescore sealed development evidence under the stricter association rules.

This is an offline verifier audit, not another Harbor playthrough. It never
changes original rewards, targets, images or runtime state.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def audit(task,jobs,output):
    task=Path(task);sys.path.insert(0,str(task/'environment'))
    path=task/'environment/backend/server.py'
    spec=importlib.util.spec_from_file_location('association_audit_server',path)
    server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
    server.ROOT=task/'environment/backend/assets';reference=json.loads((server.ROOT/'provenance/scene.truth.json').read_text())
    results=[]
    for job in map(Path,jobs):
        for ledger_path in job.glob('*/verifier/ledger.json'):
            ledger=json.loads(ledger_path.read_text());identity=ledger.get('instance',{})
            if identity.get('seed')!=reference['seed'] or identity.get('field_hashes')!=reference['field_hashes']:raise ValueError('Reference does not describe this sealed instance')
            for row in ledger['observations'].values():
                for image in row['images']:
                    file=ledger_path.parent/'evidence'/Path(image['filename']).name
                    if hashlib.sha256(file.read_bytes()).hexdigest()!=image['sha256']:raise ValueError('Sealed camera evidence identity changed')
            states=[entry['result'] for entry in ledger['trace'] if isinstance(entry.get('result'),dict) and 'pose' in entry['result']]
            if not states:raise ValueError('No authoritative recorded pose')
            landed=[entry['result'] for entry in ledger['trace'] if entry['command'].get('command')=='land']
            old=ledger['result']
            if old['returned_home'] and not landed:raise ValueError('Missing recorded landing proof')
            final=landed[-1] if landed else states[-1];server.native=lambda *a,**k:final
            state=server.Investigation();state.start=0.;state.end=old['flight_elapsed_s'];state.budget=old['budget_s'];state.actions=old['actions']
            state.home_return=old['returned_home'];state.observations=ledger['observations'];state.report=ledger['report']
            for row in state.observations.values():
                polygon=[server.project(row,u,v) for u,v in [(0,0),(1000,0),(1000,1000),(0,1000)]]
                state.covered.update(i for i,cell in enumerate(state.cells) if server.inside(cell,polygon))
            score=state.score()
            if abs(score['qualified_coverage']-old['qualified_coverage'])>1e-9:raise ValueError('Coverage changed in an association-only audit')
            results.append(dict(job=job.name,trial=ledger_path.parents[1].name,original=old,association_audit=score,ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest()))
    document=dict(kind='offline_verifier_association_audit',native_flight_replayed=False,
        server_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=results)
    Path(output).write_text(json.dumps(document,indent=2)+'\n');print(json.dumps(document,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--job',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.task,a.job,a.output)
