"""A one-time held-out image evaluation bound to a prior development selection.

Training and the normal development evaluator continue to reject test rows.
This module exposes a separate, explicit test-opening contract; it never
relabels a test row or uses its score to choose a checkpoint.
"""
import hashlib
import json
from pathlib import Path
from .common import digest,utcnow


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_once(path,document):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(document,stream,indent=2);stream.write('\n')


def freeze_selection(root,results,output):
    root=Path(root).resolve();choices=[]
    for path in map(Path,results):
        path=path.resolve()
        if not path.is_relative_to(root):raise ValueError('Selection result outside artifact store')
        if not path.is_file():continue
        record=json.loads(path.read_text())
        if record.get('status')!='completed':continue
        result=record['results'];score=result['grpo_evaluation'];training=result['grpo']
        if score.get('status')!='completed' or score.get('examples')!=120:raise ValueError('Need the complete fixed development set')
        if training.get('nonzero_gradient_checks',0)<=0 or training.get('optimizer_updates',0)<=0:
            raise ValueError('Candidate has no verified training updates')
        adapter=(root/training['adapter']).resolve()
        if not adapter.is_relative_to(root/'training'):raise ValueError('Adapter outside training store')
        choices.append(dict(result=str(path),result_sha256=sha(path),adapter=str(adapter),
            adapter_sha256=sha(adapter/'adapter_model.safetensors'),manifest_hash=training['manifest_hash'],
            evaluation_protocol_hash=score['protocol_hash'],micro_f2=score['micro_f2'],
            negative_false_alarm_rate=score['negative_false_alarm_rate'],valid_json_fraction=score['valid_json_fraction']))
    if not choices:raise ValueError('No completed candidate for development selection')
    if len({r['manifest_hash'] for r in choices})!=1:raise ValueError('Candidates used different training manifests')
    selected=max(choices,key=lambda r:(r['micro_f2'],-r['negative_false_alarm_rate'],r['valid_json_fraction']))
    receipt=dict(version=1,at=utcnow(),rule='Maximum completed GRPO development F2, lower negative false-alarm rate, higher valid JSON',
        candidates=choices,selected=selected,test_opened=False,
        scope='One-time image-label agreement on previously held-out zones; no checkpoint selection after opening')
    write_once(output,receipt);return receipt


def validate_selection(path):
    receipt=json.loads(Path(path).read_text());selected=receipt['selected']
    if receipt.get('test_opened') is not False:raise ValueError('Selection must precede test opening')
    if sha(selected['result'])!=selected['result_sha256'] or sha(Path(selected['adapter'])/'adapter_model.safetensors')!=selected['adapter_sha256']:
        raise ValueError('Frozen selection identity changed')
    return receipt


def prepare_manifest(root,selection,source,train,development,output):
    root=Path(root).resolve();receipt=validate_selection(selection)
    forbidden=set()
    for path in [train,development]:
        forbidden.update(r['sha256'] for r in json.loads(Path(path).read_text())['images'])
    source_doc=json.loads(Path(source).read_text());rows=[];seen={};excluded=[]
    for row in source_doc['images']:
        if row.get('split')!='test':continue
        if row.get('zone') not in ('zone8','zone9'):raise ValueError('Unexpected held-out zone')
        identity=row['sha256']
        if identity in forbidden:excluded.append(dict(id=row['id'],reason='training_or_development_hash_overlap'));continue
        if identity in seen:
            if digest(seen[identity]['boxes'])!=digest(row['boxes']):raise ValueError('Duplicate test image has inconsistent labels')
            excluded.append(dict(id=row['id'],reason='duplicate_test_image'));continue
        image=(root/row['image']).resolve()
        if not image.is_relative_to(root) or sha(image)!=identity:raise ValueError('Test image identity changed')
        seen[identity]=row;rows.append(row)
    if not rows:raise ValueError('No disjoint test images')
    document=dict(metadata=dict(selection_sha256=sha(selection),source_manifest_sha256=sha(source),
        training_manifest=str(Path(train).resolve()),development_manifest=str(Path(development).resolve()),
        training_manifest_sha256=sha(train),development_manifest_sha256=sha(development),
        opened_at=utcnow(),excluded=excluded,scope='Previously held-out ZIP zones 8/9; mixed original-manual and reviewed-AI labels; not field accuracy'),images=rows)
    write_once(output,document);return document


def validate_test_contract(root,selection,manifest):
    root=Path(root).resolve();receipt=validate_selection(selection);document=json.loads(Path(manifest).read_text())
    if document['metadata']['selection_sha256']!=sha(selection):raise ValueError('Manifest is not bound to the frozen selection')
    rows=document['images']
    if not rows or len({r['sha256'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate test rows')
    for row in rows:
        if row.get('split')!='test' or row.get('zone') not in ('zone8','zone9'):raise ValueError('Test contract contains non-test rows')
        image=(root/row['image']).resolve()
        if not image.is_relative_to(root) or sha(image)!=row['sha256']:raise ValueError('Test image identity changed')
    forbidden=set()
    for key in ['training_manifest','development_manifest']:
        path=Path(document['metadata'][key]).resolve()
        if not path.is_relative_to(root) or sha(path)!=document['metadata'][key+'_sha256']:
            raise ValueError('Disjointness lineage identity changed')
        forbidden.update(r['sha256'] for r in json.loads(path.read_text())['images'])
    if forbidden.intersection(r['sha256'] for r in rows):raise ValueError('Test rows overlap training or development')
    return receipt,document,dict(selection_sha256=sha(selection),manifest_sha256=sha(manifest),
        adapter_sha256=receipt['selected']['adapter_sha256'],scope='One-time held-out label agreement; not mission reward or physical mine accuracy')
