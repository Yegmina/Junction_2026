"""Frozen experiment manifests. Manifest creation never invents finished runs."""
from .common import config, data_root, digest, save, utcnow


def freeze():
    cfg=config();counts=cfg['counts'];root=data_root()/'experiments'/cfg['experiment']
    specs=[]
    for split,key in [('train','train_campaigns'),('validation','validation_campaigns')]:
        for i in range(counts[key]):
            specs.append(dict(id=digest([cfg['seed'],split,i])[:24],split=split,seed=cfg['seed']+i+(100000 if split=='validation' else 0),fidelity='fast_generator',status='awaiting_accepted_physics'))
    for split,key in [('fast_test','fast_test_campaigns'),('direct_test','direct_test_campaigns')]:
        for i in range(counts[key]):
            group,layout=divmod(i,3)
            specs.append(dict(id=digest([cfg['seed'],split,group,layout])[:24],split=split,background_group=f'{split}-{group:03d}',layout=['A','relocated_B','target_free'][layout],seed=cfg['seed']+1000000+group+(1000000 if split=='direct_test' else 0),fidelity='direct' if split=='direct_test' else 'fast_generator',distribution='in_distribution' if group%2==0 else 'held_out_combination',status='awaiting_accepted_physics'))
    physics=[]
    for split,key in [('train','physical_pairs'),('validation','validation_pairs')]:
        for i in range(counts[key]):
            for target_present in [True,False]:
                physics.append(dict(id=digest([split,i,target_present,cfg['seed']])[:24],pair_id=f'{split}-{i:03d}',split=split,targets_present=target_present,seed=cfg['seed']+i+(10000 if split=='validation' else 0),status='planned'))
    for s in [s for s in specs if s['split']=='direct_test']:
        physics.append(dict(id=s['id'],split='direct_test',layout=s['layout'],seed=s['seed'],status='planned'))
    doc=dict(created_at=utcnow(),config_hash=digest(cfg),campaigns=specs,physics=physics,counts=counts,status='frozen_manifest_only',research_accepted=False)
    destination=root/'manifest.json'
    if destination.exists():
        import json
        existing=json.loads(destination.read_text())
        if existing['config_hash']!=doc['config_hash']:
            raise ValueError('Experiment already frozen with different configuration; use a new experiment name')
        return existing
    save(destination,doc)
    return doc
