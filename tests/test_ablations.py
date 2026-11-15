import copy,json
from pathlib import Path
from PIL import Image
from aerorozum.ablations import freeze_label_source_ablation,KIND
from aerorozum.common import save
from aerorozum.fast_fields import file_hash
from aerorozum.streaming import StreamCoordinator
from test_streaming import real_index,settings,Process

def inputs(root):
    doc=real_index(root)
    for i in range(32,60):
        row=copy.deepcopy(doc['images'][24 if i<36 else 28]);path=root/'images'/f'{i}.png'
        Image.new('RGB',(32,32),(i*7%255,i*13%255,i*19%255)).save(path)
        row.update(id=str(i),image=str(path.relative_to(root)),sha256=file_hash(path));doc['images'].append(row)
    save(root/'pipeline/inputs/real.json',doc)
    save(root/'pipeline/validation.json',{'metadata':{'frozen_development':True},'images':[r for r in doc['images'] if r['split']=='validation']})
    return doc

def test_matched_ablation_uses_all_anchors_same_zeros_and_immutable_development(tmp_path):
    inputs(tmp_path);validation=(tmp_path/'pipeline/validation.json').read_bytes()
    request=freeze_label_source_ablation(tmp_path);assert len(request['jobs'])==6
    documents={j['arm']:json.loads((tmp_path/j['manifest']).read_text()) for j in request['jobs']}
    first,second=documents.values()
    assert len(first['images'])==len(second['images'])==32
    assert sum(r['provenance'] in ('original_manual','reviewed_ai') for r in first['images'])==16
    assert sum(r['provenance']=='user_accepted_model' and bool(r['boxes']) for r in second['images'])==8
    assert {r['sha256'] for r in first['images'] if not r['boxes']}=={r['sha256'] for r in second['images'] if not r['boxes']}
    assert all(r['split']=='train' and r.get('zone')!='zone7' for d in documents.values() for r in d['images'])
    assert freeze_label_source_ablation(tmp_path)==request and (tmp_path/'pipeline/validation.json').read_bytes()==validation

def test_ablation_jobs_share_controller_gpu_leases_and_allow_independent_arms(tmp_path):
    inputs(tmp_path);freeze_label_source_ablation(tmp_path);calls=[]
    worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:(calls.append(args) or Process()))
    worker.state['measured_topology']={'gpu_groups':[[0],[1],[2],[3]],'microbatch':2,'gradient_accumulation':2}
    worker.ingest_job_requests();worker.manage_jobs();published=worker.publish()
    assert len(worker.state['jobs'])==6 and len(calls)==4
    assert {tuple(j['gpu_group']) for j in worker.state['jobs'] if j['status']=='running'}=={(0,),(1,),(2,),(3,)}
    assert all('aerorozum.ablations' in c[0] and '--warm-start' not in c[0] for c in calls)
    assert {r['kind'] for r in published['training']['active']}=={KIND}
    worker.ingest_job_requests();assert len(worker.state['jobs'])==6

def test_arbitrary_worker_request_cannot_execute(tmp_path):
    save(tmp_path/'pipeline/job-requests/unsafe.json',{'kind':'shell','family_id':'a'*24,'jobs':[]})
    calls=[];worker=StreamCoordinator(tmp_path,settings(),executor=lambda *a:calls.append(a));worker.ingest_job_requests()
    assert not calls and not worker.state['jobs']
    assert next(iter(worker.state['job_requests'].values()))['status']=='rejected'

def test_expiring_external_physics_lease_prevents_a_training_collision(tmp_path):
    inputs(tmp_path);freeze_label_source_ablation(tmp_path);calls=[]
    worker=StreamCoordinator(tmp_path,settings(),executor=lambda *args:(calls.append(args) or Process()))
    worker.state['measured_topology']={'gpu_groups':[[0],[1],[2],[3]],'microbatch':2,'gradient_accumulation':2}
    save(tmp_path/'pipeline/controls.json',{'external_gpu_leases':[{'owner':'physics-amgx','gpu_ids':[1],'expires_at':'2098-01-01T00:00:00+00:00'}]})
    worker.ingest_job_requests();worker.manage_jobs()
    assert len(calls)==3 and all(j.get('gpu_group')!=[1] for j in worker.state['jobs'] if j['status']=='running')
    assert worker.publish()['external_gpu_leases'][0]['status']=='reserved'
    save(tmp_path/'pipeline/controls.json',{'external_gpu_leases':[{'owner':'physics-amgx','gpu_ids':[1],'expires_at':'2000-01-01T00:00:00+00:00'}]})
    worker.manage_jobs();assert len(calls)==4 and not worker.state['external_gpu_leases']
