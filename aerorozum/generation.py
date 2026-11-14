"""Exact-count synthetic data generation with explicit physical acceptance gates.

Demo generation is separate and never marks analytic labels research accepted.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects

from .common import config as load_config, data_root, digest, save, utcnow
from .environment import Campaign
from .renderer import ThermalRenderer


def _load(value):
    return copy.deepcopy(value) if isinstance(value,dict) else json.loads(Path(value).read_text())


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def plan_generation(frozen_manifest,field_registry):
    try:frozen=_load(frozen_manifest)
    except (OSError,ValueError) as exc:
        return {'status':'blocked','research_accepted':False,'requested_images':0,'requested_campaigns':0,'accepted_campaigns':0,'problems':[f'Frozen manifest unavailable: {exc}'],'accepted_fields':[],'config_hash':None}
    try:registry=_load(field_registry)
    except (OSError,ValueError) as exc:
        return {'status':'blocked','research_accepted':False,'requested_images':int(frozen.get('counts',{}).get('images',30000)),
                'requested_campaigns':len(frozen.get('campaigns',[])),'accepted_campaigns':0,'problems':[f'Accepted physical-field registry unavailable: {exc}'],'accepted_fields':[],'config_hash':frozen.get('config_hash')}
    problems=[];rows=registry.get('fields',[]);indexed={r.get('campaign_id'):r for r in rows}
    if len(indexed)!=len(rows):problems.append('Duplicate campaign IDs in field registry')
    if registry.get('config_hash')!=frozen.get('config_hash'):problems.append('Registry is not bound to the frozen experiment configuration')
    if not frozen.get('campaigns') or not frozen.get('config_hash'):problems.append('A frozen campaign manifest is required')
    seen_background={};accepted=[]
    for spec in frozen.get('campaigns',[]):
        cid=spec['id'];row=indexed.get(cid)
        if not row:
            problems.append(f'{cid}: missing accepted physical field registration');continue
        try:
            if row.get('split')!=spec['split']:raise ValueError('Registry split differs from frozen campaign')
            if row.get('status')!='accepted':raise ValueError('Field registration is not accepted')
            if spec.get('layout') and row.get('layout')!=spec['layout']:raise ValueError('Registry layout differs from frozen relocated/control scene')
            for key in ('field_path','counterfactual_path','observability_manifest_path','acceptance_report_path'):
                if not row.get(key) or not Path(row[key]).is_file():raise ValueError(f'Missing {key}')
            present=Path(row['field_path']);absent=Path(row['counterfactual_path'])
            pm=_load(present.with_suffix('.json'));am=_load(absent.with_suffix('.json'))
            report=_load(row['acceptance_report_path'])
            if not all(m.get('research_accepted') is True for m in (pm,am,report)):
                raise ValueError('Physical fields and acceptance report must be explicitly research accepted')
            from .fast_fields import validate_pair
            validate_pair(row)
            if not row.get('background_group'):raise ValueError('Registry needs a background group for split isolation')
            group=row['background_group']
            if group in seen_background and seen_background[group]!=spec['split']:
                raise ValueError('Background group crosses experiment splits')
            seen_background[group]=spec['split']
            if not row.get('paired_environment_hash') or pm.get('paired_environment_hash')!=row['paired_environment_hash'] or am.get('paired_environment_hash')!=row['paired_environment_hash']:
                raise ValueError('Counterfactual pair must share a verified environment hash')
            with np.load(present,allow_pickle=False) as p,np.load(absent,allow_pickle=False) as a:
                for axis in ('x_m','y_m'):
                    if not np.array_equal(p[axis],a[axis]):raise ValueError('Counterfactual spatial coordinate axes differ')
                if min(p['times_s'][-1],a['times_s'][-1])<=max(p['times_s'][0],a['times_s'][0]):raise ValueError('Counterfactual fields have no common physical-time coverage')
            absent_truth=_load(absent.with_suffix('.truth.json'))
            if absent_truth.get('targets'):raise ValueError('Target-free counterfactual contains target records')
            cert=_load(row['observability_manifest_path'])
            if cert.get('verified') is not True or cert.get('field_sha256')!=_hash(present):
                raise ValueError('Frozen observability certificate does not match field')
            accepted.append({**row,'spec':spec,'metadata':pm})
        except (OSError,KeyError,ValueError,TypeError) as exc:
            problems.append(f'{cid}: {exc}')
    total=int(frozen.get('counts',{}).get('images',30000))
    return {'status':'blocked' if problems else 'ready','research_accepted':not problems,
            'requested_images':total,'requested_campaigns':len(frozen.get('campaigns',[])),
            'accepted_campaigns':len(accepted),'problems':problems,'accepted_fields':accepted,
            'config_hash':frozen.get('config_hash')}


def _counterfactual_labels(renderer,counterfactual,pose,time_s):
    xs,ys=renderer.ground_grid(pose)
    difference=np.abs(renderer.temperature(xs,ys,time_s)-counterfactual.temperature(xs,ys,time_s))
    inside=(xs>=0)&(xs<=renderer.config['width_m'])&(ys>=0)&(ys<=renderer.config['height_m'])
    components,count=label((difference>=.12)&inside)
    height,width=difference.shape
    candidates=[]
    projected=[(target,renderer.projected_box(target,pose)) for target in renderer.targets]
    for index,slices in enumerate(find_objects(components),1):
        if slices is None:continue
        yy,xx=slices
        if int(np.sum(components[slices]==index))<9 or xx.stop-xx.start<3 or yy.stop-yy.start<3:continue
        cx=(xx.start+xx.stop)/2/width*1000;cy=(yy.start+yy.stop)/2/height*1000
        associations=[]
        for target,box in projected:
            if box is None:continue
            tx=(box['x_min']+box['x_max'])/2;ty=(box['y_min']+box['y_max'])/2
            distance=(cx-tx)**2+(cy-ty)**2
            associations.append((distance,target))
        if not associations:continue
        _,target=min(associations,key=lambda item:item[0])
        candidates.append({'label':'provisional thermal target signature','target_id':str(target['id']),
            'bbox':{'x_min':xx.start/width*1000,'y_min':yy.start/height*1000,'x_max':xx.stop/width*1000,'y_max':yy.stop/height*1000},
            'label_method':'matched_counterfactual_surface_temperature_component',
            'contrast_threshold_k':.12,'physical_mine_identity_confirmed':False})
    return candidates


def generate_dataset(frozen_manifest,field_registry,output_dir=None,count=None,cfg=None,artifact_root=None):
    """Generate the frozen exact image count or stop before producing research data."""
    frozen=_load(frozen_manifest);plan=plan_generation(frozen,field_registry)
    root=Path(output_dir or data_root()/'datasets'/'synthetic').resolve();root.mkdir(parents=True,exist_ok=True)
    if count is not None and int(count)!=plan['requested_images']:
        raise ValueError('Generation cannot silently reduce the frozen image count')
    if plan['status']=='blocked':
        result={k:v for k,v in plan.items() if k!='accepted_fields'}
        save(root/'generation-status.json',result)
        return result
    cfg=copy.deepcopy(cfg or load_config());rows=[];total=plan['requested_images'];fields=plan['accepted_fields']
    base=Path(artifact_root or data_root()).resolve()
    if not root.is_relative_to(base):raise ValueError('Research dataset output must be inside its explicit artifact root')
    journal=root/'images.jsonl';existing={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                row=json.loads(line);existing[row['id']]=row
            except (ValueError,KeyError):pass
    for ordinal,field in enumerate(fields):
        spec=field['spec'];n=total//len(fields)+(ordinal<total%len(fields))
        metadata=field['metadata'];domain=metadata.get('domain',{})
        local_cfg=copy.deepcopy(cfg)
        local_cfg.setdefault('campaign',{}).update(width_m=domain.get('width_m',cfg.get('campaign',{}).get('width_m',64)),height_m=domain.get('height_m',cfg.get('campaign',{}).get('height_m',64)))
        local_cfg.update(thermal_field_path=field['field_path'],observability_manifest_path=field['observability_manifest_path'],mode='research')
        campaign=Campaign(local_cfg,seed=spec['seed'],output_dir=root/'campaign-artifacts')
        renderer=campaign.renderer
        absent_cfg=copy.deepcopy(campaign.config);absent_cfg['thermal_field_path']=field['counterfactual_path']
        counterfactual=ThermalRenderer(absent_cfg,spec['seed'],[])
        source_hash=_hash(field['field_path']);counterfactual_hash=_hash(field['counterfactual_path'])
        common_start=max(21600,float(renderer.field['times_s'][0]),float(counterfactual.field['times_s'][0]))
        common_end=min(79200,float(renderer.field['times_s'][-1]),float(counterfactual.field['times_s'][-1]))
        if common_end<=common_start:raise ValueError('No common daytime physical coverage; extrapolation is forbidden')
        rng=np.random.default_rng(spec['seed'])
        for i in range(n):
            identity=digest([spec['id'],i]);relative=Path(spec['split'])/(identity+'.png')
            height=float(rng.uniform(2,12));fw=2*height*math.tan(math.radians(campaign.config['camera']['hfov_deg'])/2);fh=2*height*math.tan(math.radians(campaign.config['camera']['vfov_deg'])/2)
            width=campaign.config['width_m'];depth=campaign.config['height_m']
            if fw>width or fh>depth:raise ValueError('Accepted domain cannot contain requested camera footprint; no stretching allowed')
            pose={'x_m':float(rng.uniform(fw/2,width-fw/2)),'y_m':float(rng.uniform(fh/2,depth-fh/2)),
                  'altitude_m':height,'yaw_deg':0.}
            time_s=float(rng.uniform(common_start,common_end))
            request_hash=digest({'camera':campaign.config['camera'],'pose':pose,'time_s':time_s,
                                 'source':source_hash,'counterfactual':counterfactual_hash,'seed':spec['seed'],
                                 'label_protocol':'counterfactual_0.12k_components_min3pixels_v1'})
            old=existing.get(identity)
            old_image=(base/old['image']).resolve() if old else None
            if old and old.get('provenance') in ('direct_physics','validated_surrogate') and old.get('render_request_hash')==request_hash and old.get('source_field_sha256')==source_hash and old.get('counterfactual_field_sha256')==counterfactual_hash and old_image.is_relative_to(base) and old_image.is_file() and _hash(old_image)==old.get('sha256'):
                rows.append(old);continue
            info=renderer.render(pose,time_s,root/relative,i)
            boxes=_counterfactual_labels(renderer,counterfactual,pose,time_s)
            row={'id':identity,'campaign_id':spec['id'],'background_group':field['background_group'],'split':spec['split'],
                'image':str((root/relative).relative_to(base)),'sha256':_hash(root/relative),'image_sha256':_hash(root/relative),'width':info['width'],'height':info['height'],
                'pose':pose,'time_s':time_s,'boxes':boxes,'provenance':'validated_surrogate' if 'surrogate' in renderer.fidelity.lower() else 'direct_physics',
                'research_accepted':True,'fidelity':renderer.fidelity,'source_field_sha256':source_hash,
                'counterfactual_field_sha256':counterfactual_hash,'acceptance_certificate_sha256':_hash(field['acceptance_report_path']),
                'label_method':'matched_counterfactual_surface_temperature_component','render_request_hash':request_hash}
            rows.append(row)
            with journal.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
    assert len(rows)==total
    manifest={'schema_version':1,'status':'completed','research_accepted':True,'config_hash':plan['config_hash'],
              'metadata':{'status':'passed','research_accepted':True,'config_hash':plan['config_hash']},
              'created_at':utcnow(),'images':rows,'counts':{'images':len(rows),'campaigns':len(fields)},
              'interpretation':'Synthetic thermal signatures; no claim of real-world mine detection validation'}
    save(root/'manifest.json',manifest)
    return {'status':'completed','research_accepted':True,'images':len(rows),'manifest':str(root/'manifest.json')}


def generate_demo_dataset(output_dir=None,count=8,seed=0):
    """Separate analytic plumbing dataset; never eligible for research training."""
    if not 1<=int(count)<=32:raise ValueError('Demo image count must be in 1..32 for one bounded sortie')
    root=Path(output_dir or data_root()/'datasets'/'demo').resolve();root.mkdir(parents=True,exist_ok=True)
    campaign=Campaign(seed=seed,output_dir=root)
    campaign.act('launch',{'altitude_m':6});rows=[]
    for i in range(int(count)):
        if i:
            campaign.act('move_to',{'x_m':8+(i*13)%48,'y_m':8+(i*17)%48})
        result=campaign.act('capture')
        if not result['ok']:raise RuntimeError(result['error'])
        observation=result['observation']
        private=campaign._observations[observation['id']]
        boxes=[{'label':'demonstration thermal signature','bbox':t['box']} for t in private['truth'] if t['observable']]
        rows.append({'id':observation['id'],'split':'demo','image':str(root/observation['image']),
            'width':observation['width'],'height':observation['height'],'boxes':boxes,
            'image_sha256':_hash(root/observation['image']),'provenance':'analytic_demo_not_research',
            'research_accepted':False,'fidelity':observation['fidelity']})
    doc={'status':'demo','research_accepted':False,'created_at':utcnow(),'images':rows,
         'interpretation':'Analytic demonstration labels only; not physical solver data or verified mine identities'}
    save(root/'manifest.json',doc)
    return {'status':'demo','research_accepted':False,'images':len(rows),'manifest':str(root/'manifest.json')}
