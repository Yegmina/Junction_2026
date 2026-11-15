"""Reviewable challenge packages. This module never posts or submits anything."""
import json
import zipfile
from .common import data_root, save, utcnow
from .preflight import SPONSORS


def package():
    root=data_root();target=root/'submissions';target.mkdir(parents=True,exist_ok=True)
    current=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    statuses={x['id']:x for x in current.get('sponsors',[])}
    records=[]
    for identifier,name,status,role in SPONSORS:
        actual=statuses.get(identifier,dict(status=status))
        record=dict(challenge=name,integration_role=role,integration_status=actual['status'],eligibility_confirmed=False,submitted=False,evidence=[],pending=['Verify official platform requirements','Add demonstrated usage evidence','Complete required submission forms'])
        if identifier=='gemini':
            responses=list((root/'gemini/responses').glob('*.json'))
            record['evidence']=[dict(kind='completed_request_count',value=len(responses)),dict(kind='backend',value='developer_api; no enterprise fallback')]
        if identifier=='codex':
            record['evidence']=[dict(kind='implementation',value='Source files, tests and development history in repository')]
        save(target/(identifier+'.json'),record);records.append(record)
    lines=['# AeroRozum challenge evidence','',f'Generated {utcnow()}. Nothing has been submitted automatically.','', '| Challenge | Integration | Status |','|---|---|---|']
    lines += [f"| {r['challenge']} | {r['integration_role']} | {r['integration_status']} |" for r in records]
    lines += ['', 'Research claims require accepted physics, frozen test runs and actual training results. Original private recordings and credentials are excluded.']
    (target/'challenge-evidence.md').write_text('\n'.join(lines)+'\n')
    bundle=target/'challenge-evidence.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(target.glob('*.json')):z.write(p,p.name)
        z.write(target/'challenge-evidence.md','challenge-evidence.md')
    return dict(status='prepared_not_submitted',path=str(bundle),challenges=len(records),entries_confirmed=0)
