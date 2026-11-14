"""Mirror only the isolated VM's paid label artifacts."""
import argparse,shlex,subprocess,time
from datetime import datetime,timezone
from aerorozum.common import data_root,save,utcnow,redacted_error
from aerorozum.preflight import sdk_path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    root=data_root();target=root/'labelling/robotics-er2-airborne';target.mkdir(parents=True,exist_ok=True)
    command=[sdk_path(),'compute','ssh','aerorozum-gemma-pilot-20261114a','--zone=europe-west4-a','--project=avid-relic-509408-e4','--quiet','--dry-run']
    parts=shlex.split(subprocess.check_output(command,text=True).strip());host=parts[-1];ssh=shlex.join([p for p in parts[:-1] if p!='-t'])
    if not host or any(c.isspace() for c in host):raise ValueError('Invalid SSH destination')
    remote=host+':/opt/aerorozum/airborne-labeler/artifacts/labelling/robotics-er2-airborne/'
    while datetime.now(timezone.utc)<datetime.fromisoformat('2026-11-15T06:30:50+00:00'):
        try:
            subprocess.run(['rsync','-rt','--rsync-path=sudo rsync','--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=','--partial','-e',ssh,remote,str(target)+'/'],check=True,timeout=180,capture_output=True,text=True)
            report={'status':'synced','at':utcnow(),'local':'labelling/robotics-er2-airborne'}
        except Exception as error:report={'status':'retrying','at':utcnow(),'error':redacted_error(error)}
        save(root/'pipeline/airborne-label-mirror.json',report)
        if args.once:return
        time.sleep(30)

if __name__=='__main__':main()
