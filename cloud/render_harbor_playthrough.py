"""Publish an interactive replay of recorded tool states and original captures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def render(trial,template,output,key_file=None):
    trial=Path(trial);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=(trial/'verifier/ledger.json').read_bytes();ledger=json.loads(raw)
    metadata=json.loads((trial/'result.json').read_text());agent=metadata['config']['agent']
    agent_label=agent.get('model_name') or ('Public-image reference' if agent['name']=='oracle' else agent['name'])
    if not ledger['observations']:raise ValueError('No camera observations in this run')
    images=output/'images';images.mkdir(exist_ok=True);observations={}
    annotations={}
    for index,candidate in enumerate((ledger.get('report') or {}).get('candidates',[]),1):
        for item in candidate.get('evidence',[]):
            annotations.setdefault(item['observation_id'],[]).append(dict(candidate=index,bbox=item['bbox']))
    for identity,row in ledger['observations'].items():
        record=dict(id=identity,pose=row['pose'],sim_time_s=row['sim_time_s'],claims=annotations.get(identity,[]),images={})
        for item in row['images']:
            payload=(trial/'verifier/evidence'/Path(item['filename']).name).read_bytes()
            checksum=hashlib.sha256(payload).hexdigest()
            if checksum!=item['sha256']:raise ValueError('Sealed image hash differs')
            name=checksum+'.png';(images/name).write_bytes(payload)
            record['images'][item['kind']]=dict(url='images/'+name,sha256=checksum)
        observations[identity]=record
    events=[];last_observation=None;last_pose=None
    for row in ledger['trace']:
        command=row['command'];name=command.get('command')
        if name in ('status','result'):continue
        result=row['result'];state=result.get('status',result)
        if result.get('observation_id'):last_observation=result['observation_id']
        if isinstance(state.get('pose'),dict):last_pose=state['pose']
        public_command={k:v for k,v in command.items() if k in ('command','x','y','altitude','seconds')}
        events.append(dict(command=public_command,at=row.get('at'),pose=last_pose,
            flight_elapsed_s=state.get('flight_elapsed_s'),remaining_s=state.get('remaining_s'),observation=last_observation))
    events.append(dict(command=dict(command='verifier sealed'),pose=last_pose,
        flight_elapsed_s=ledger['result']['flight_elapsed_s'],remaining_s=max(0,ledger['result']['budget_s']-ledger['result']['flight_elapsed_s']),
        observation=last_observation))
    document=dict(agent_label=agent_label,source_ledger_sha256=hashlib.sha256(raw).hexdigest(),result=ledger['result'],
        instance=ledger['instance'],observations=observations,events=events,
        scope='Compressed replay of recorded tool states, not continuous video. Blue boxes are final submitted claims, not private truth.')
    encoded=json.dumps(document,indent=2,allow_nan=False).encode()
    if key_file:
        secret=Path(key_file).read_text().strip().encode()
        if secret and (secret in encoded or any(secret in p.read_bytes() for p in images.iterdir())):
            raise ValueError('Private credential scan rejected replay')
    (output/'replay.json').write_bytes(encoded);shutil.copyfile(template,output/'index.html')
    print(json.dumps(dict(events=len(events),captures=len(observations),ledger_sha256=document['source_ledger_sha256'],output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',required=True);p.add_argument('--template',required=True)
    p.add_argument('--output',required=True);p.add_argument('--key-file');a=p.parse_args();render(a.trial,a.template,a.output,a.key_file)
