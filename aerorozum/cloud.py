"""Run-owned Google Cloud inventory and bounded worker planning."""
from __future__ import annotations
import json
import subprocess
from .common import config, data_root, save, utcnow
from .preflight import sdk_path


def command(*args, timeout=120):
    r=subprocess.run([sdk_path(),*args],capture_output=True,text=True,timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return r.stdout


def inventory():
    project=config().get('cloud',{}).get('project','avid-relic-509408-e4')
    instances=json.loads(command('compute','instances','list',f'--project={project}','--filter=labels.project=aerorozum-thermal-rl','--format=json'))
    public=[]
    for item in instances:
        public.append(dict(name=item['name'],zone=item['zone'].split('/')[-1],machine_type=item['machineType'].split('/')[-1],status=item['status'],created_at=item.get('creationTimestamp'),scheduling=item.get('scheduling'),accelerators=item.get('guestAccelerators',[])))
    result=dict(project=project,checked_at=utcnow(),instances=public,provider='google_cloud',scope='run_owned_only')
    save(data_root()/'cloud/inventory.json',result)
    return result


def plan_schedule(measurements):
    """Do not invent ETA without measured critical-path inputs."""
    required=('physics_trajectory_seconds','direct_scene_seconds','render_images_per_second','sft_seconds_per_seed','grpo_seconds_per_seed','campaign_seconds','cpu_workers','gpu_seed_groups')
    missing=[k for k in required if k not in measurements or measurements[k]<=0]
    if missing:
        return dict(status='awaiting_measurements',missing=missing,estimated_finish_s=None)
    c=config()['counts'];m=measurements
    physics=(2*(c['physical_pairs']+c['validation_pairs'])*m['physics_trajectory_seconds']+c['direct_scenes']*m['direct_scene_seconds'])/m['cpu_workers']
    render=c['images']/m['render_images_per_second']
    training=len(c['training_seeds'])*(m['sft_seconds_per_seed']+m['grpo_seconds_per_seed'])/m['gpu_seed_groups']
    evaluation=(c['fast_test_campaigns']+c['direct_test_campaigns'])*c['frontier_repetitions']*m['campaign_seconds']/m.get('evaluation_workers',1)
    return dict(status='measured_projection',physics_s=physics,render_s=render,training_s=training,evaluation_s=evaluation,estimated_finish_s=1.2*(physics+render+training+evaluation),assumptions='Conservative critical-path bound; includes 20% margin; excludes unknown startup and additional ablation work')


def teardown():
    current=inventory();deleted=[]
    for item in current['instances']:
        command('compute','instances','delete',item['name'],f"--zone={item['zone']}",f"--project={current['project']}",'--quiet')
        deleted.append(item['name'])
    return dict(deleted=deleted,artifacts_preserved=True,actual_billing='Consult billing export; reservations remain conservative until reconciled')
