#!/usr/bin/env python3
"""Upload only physics outputs/checkpoints using VM ADC; never log tokens."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import google.auth
from google.auth.transport.requests import AuthorizedSession


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bucket',default='aerorozum-thermal-rl-avid-relic-509408-e4')
    p.add_argument('--prefix',default='physics/cloud-cases')
    p.add_argument('--watch',type=float,default=0,help='Poll interval; 0 performs one scan')
    p.add_argument('--case-name',action='append',default=[],help='Optional exact case-directory allowlist')
    a=p.parse_args();root=a.root.resolve()
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    session=AuthorizedSession(credentials);seen={}
    while True:
        uploads=[]
        def upload(path,object_name):
            if not path.is_file():return
            stamp=(path.stat().st_size,path.stat().st_mtime_ns)
            if seen.get(str(path))==stamp:return
            try:
                with path.open('rb') as payload:body=payload.read(stamp[0])
            except FileNotFoundError:return
            response=session.post('https://storage.googleapis.com/upload/storage/v1/b/'+a.bucket+'/o',params={'uploadType':'media','name':object_name},data=body,timeout=180)
            if not response.ok:raise RuntimeError(f'Artifact upload HTTP {response.status_code}: {path.name}')
            seen[str(path)]=stamp;uploads.append(object_name)
        for path in root.glob('*.json'):upload(path,a.prefix.strip('/')+'/_metadata/'+path.name)
        registry=root.parent.parent/'registry.json'
        if registry.is_file():upload(registry,a.prefix.strip('/')+'/_metadata/registry.json')
        for case in root.iterdir():
            if not case.is_dir() or not case.resolve().is_relative_to(root):continue
            if a.case_name and case.name not in a.case_name:continue
            status_path=case/'run-status.json'
            try:status=json.loads(status_path.read_text())
            except (FileNotFoundError,json.JSONDecodeError):continue
            names=['run-status.json','solver.log','conservation.csv','surface-partial.npz','surface-partial.json','surface-frames.bin','surface-grid.csv','surface-format.json','input.used.input','retirement.json','backend-verification.json','reference-ledger.snapshot.csv','candidate-ledger.snapshot.csv']
            names+=['gpu-linear-solves.jsonl','native-linear-baseline.json','gpu-jacobian-benchmark.json','gpu-backend-parity.json','jacobian-original.mtx','jacobian-scaled.mtx','rhs-original.mtx','rhs-scaled.mtx','correction-native-istl.mtx']
            if status.get('status')=='completed':
                names+=['field.npz','field.json','field.truth.json','input.truth.json','budget-verification.json','acceptance-report.json']
                if (case/'surface-format.json').exists():names += [str(p.relative_to(case)) for p in case.rglob('*') if p.suffix in ('.vtu','.pvtu','.pvd') and p.resolve().is_relative_to(case.resolve())]
            pvd=case/'campaign.pvd'
            if pvd.exists():
                try:
                    entries=ET.parse(pvd).findall('.//DataSet')
                    if entries:
                        last=case/entries[-1].get('file')
                        if last.resolve().is_relative_to(case.resolve()) and last.exists():
                            tree=ET.parse(last)  # Only complete XML checkpoints.
                            names.append(str(last.relative_to(case)))
                            for piece in tree.findall('.//Piece[@Source]'):
                                part=(last.parent/piece.get('Source')).resolve()
                                if part.is_relative_to(case.resolve()) and part.is_file():
                                    ET.parse(part);names.append(str(part.relative_to(case)))
                        names.append(pvd.name)
                except (ET.ParseError,FileNotFoundError):pass
            for name in names:
                path=case/name
                object_name=a.prefix.strip('/')+'/'+case.name+'/'+name
                upload(path,object_name)
        print(json.dumps(dict(uploaded=len(uploads),objects=uploads,time=time.time())),flush=True)
        if not a.watch:return
        time.sleep(a.watch)


if __name__=='__main__':main()
