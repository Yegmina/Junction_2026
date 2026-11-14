"""Read-only VM evidence: actual services, GPU load and completed Harbor results."""
import json
import subprocess
from pathlib import Path


def command(*args):
    result=subprocess.run(args,capture_output=True,text=True,timeout=20)
    return result.stdout.strip()


def main():
    root=Path('/opt/aerorozum');harbor=root/'harbor'
    print('GPU',command('nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'))
    units=command('systemctl','list-units','--type=service','--state=running','--no-legend','--plain').splitlines()
    print('RUNNING_UNITS',[line.split()[0] for line in units if any(word in line for word in ['harbor','projection','curriculum','refinement','frontier','cloud-gemma','physics-3d','ledger-mirror'])])
    for name in ['reference-v4','opencode-gemini-v3','opencode-gemma-v3','opencode-carbon-v1','nop-v4','stop-early-v4',
                 'reference-v5','reference-v5-cooling90','opencode-gemma-v4','opencode-gemma-base-v1']:
        job=harbor/'jobs'/name
        for file in job.glob('*/result.json'):
            data=json.loads(file.read_text());error=data.get('exception_info') or {}
            print('TRIAL',json.dumps(dict(job=name,trial=file.parent.name,
                rewards=(data.get('verifier_result') or {}).get('rewards'),exception_type=error.get('exception_type'),
                finished_at=data.get('finished_at'))))
        for file in job.glob('*/verifier/details.json'):print('SCORE',name,file.read_text())
        for file in job.glob('instance-check.json'):print('INSTANCE',name,file.read_text())
    names=command('docker','ps','--format','{{.Names}}').splitlines()
    for name in names:
        if '__env-simulator-' in name:
            query="import json,urllib.request;d=json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8080/tool',data=b'{\"command\":\"status\"}',headers={'Content-Type':'application/json'}),timeout=3));print(json.dumps({k:d.get(k) for k in ['flight_elapsed_s','remaining_s','action_count','coverage_fraction','report_submitted','sealed']}))"
            print('LIVE_WORLD',name,command('docker','exec',name,'python3','-c',query))
    for family in ['submission-visual-projection-v2','submission-reward-curriculum-v1']:
        for file in (root/'artifacts/training'/family).rglob('gradient_checks.json'):
            print('GRADIENT',str(file.relative_to(root)),file.read_text())


if __name__=='__main__':main()
