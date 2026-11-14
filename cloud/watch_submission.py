"""Mirror measured training/job status to the public ledger without a UI restart."""
import json,time,shutil
from pathlib import Path
source=Path('/opt/aerorozum/artifacts');target=Path('/opt/aerorozum/deployment/artifacts')

def mirror_harbor_results():
 for path in Path('/opt/aerorozum/harbor/jobs').glob('*/result.json'):
  try:
   result=json.loads(path.read_text());stats=result['stats']
   if not result.get('finished_at') or stats.get('n_running_trials') or stats.get('n_pending_trials'):continue
   evaluations=stats.get('evals',{});scores=[]
   for value in evaluations.values():
    for metric in value.get('metrics',[]):
     if isinstance(metric.get('mean'),(int,float)):scores.append(metric['mean'])
   errors=stats.get('n_errored_trials',0);identity='harbor-'+path.parent.name
   record=dict(id=identity,kind='harbor_evaluation',status='failed' if errors else 'completed',
    completed=stats.get('n_completed_trials',0),total=result.get('n_total_trials',0),
    updated_at=result['updated_at'],created_at=result['started_at'],
    detail=f"Harbor {path.parent.name}: scores {scores}; {errors} errored trials. See saved playthrough evidence; setup errors are excluded from skill comparisons.",
    metrics=dict(harbor_rewards=scores,errored_trials=errors),
    research_result=False,frontier_superiority_verified=False)
   destination=source/'jobs'/(identity+'.json');payload=json.dumps(record,indent=2)
   if not destination.exists() or destination.read_text()!=payload:
    temp=destination.with_suffix('.harbor-tmp');temp.write_text(payload);temp.replace(destination)
  except (OSError,ValueError,KeyError):pass

while True:
 mirror_harbor_results()
 for path in (source/'jobs').glob('*.json'):
  try:
   destination=target/'jobs'/path.name
   if not destination.exists() or json.loads(path.read_text()).get('updated_at','')>json.loads(destination.read_text()).get('updated_at',''):
    temp=destination.with_suffix('.mirror-tmp');shutil.copyfile(path,temp);temp.replace(destination)
  except (OSError,ValueError):pass
 try:
  snapshot=target/'status.json';document=json.loads(snapshot.read_text())
  jobs=[json.loads(p.read_text()) for p in (target/'jobs').glob('*.json')]
  document['jobs']=sorted(jobs,key=lambda job:job.get('updated_at',job.get('created_at','')),reverse=True)
  temp=snapshot.with_suffix('.ledger-tmp');temp.write_text(json.dumps(document));temp.replace(snapshot)
 except (OSError,ValueError):pass
 time.sleep(10)
