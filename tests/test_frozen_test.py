import hashlib,json
from pathlib import Path
import pytest
from aerorozum.frozen_test import freeze_selection,prepare_manifest,validate_test_contract
from aerorozum.stream_training import evaluate_adapter


def fixture(tmp_path):
    adapter=tmp_path/'training/adapter';adapter.mkdir(parents=True);(adapter/'adapter_model.safetensors').write_bytes(b'weights')
    result=tmp_path/'result.json';result.write_text(json.dumps(dict(status='completed',results=dict(
        grpo=dict(adapter='training/adapter',nonzero_gradient_checks=4,optimizer_updates=4,manifest_hash='train-id'),
        grpo_evaluation=dict(status='completed',examples=120,protocol_hash='dev-id',micro_f2=.6,negative_false_alarm_rate=.1,valid_json_fraction=1.)))))
    selection=tmp_path/'selection.json';freeze_selection(tmp_path,[result],selection)
    image=tmp_path/'image.jpg';image.write_bytes(b'image-fixture')
    row=dict(id='test-id',image='image.jpg',sha256=hashlib.sha256(image.read_bytes()).hexdigest(),zone='zone8',split='test',boxes=[],provenance='original_manual')
    source=tmp_path/'source.json';source.write_text(json.dumps(dict(images=[row])))
    train=tmp_path/'train.json';train.write_text('{"images":[]}');dev=tmp_path/'dev.json';dev.write_text('{"images":[]}')
    manifest=tmp_path/'test.json';prepare_manifest(tmp_path,selection,source,train,dev,manifest)
    return selection,manifest,train,result


def test_frozen_selection_binds_actual_weights_and_disjoint_test_images(tmp_path):
    selection,manifest,train,result=fixture(tmp_path)
    receipt,doc,contract=validate_test_contract(tmp_path,selection,manifest)
    assert doc['images'][0]['split']=='test' and len(contract['selection_sha256'])==64
    assert receipt['selected']['micro_f2']==.6
    with pytest.raises(FileExistsError):freeze_selection(tmp_path,[result],selection)
    (tmp_path/'training/adapter/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='selection identity'):validate_test_contract(tmp_path,selection,manifest)


def test_contract_rejects_altered_lineage_and_test_rows_cannot_enter_development(tmp_path):
    selection,manifest,train,result=fixture(tmp_path)
    with pytest.raises(ValueError,match='development validation only'):
        evaluate_adapter(None,manifest,tmp_path/'output','not-loaded')
    train.write_text('{"images":[{"sha256":"changed"}]}')
    with pytest.raises(ValueError,match='lineage identity'):validate_test_contract(tmp_path,selection,manifest)


def test_preparation_excludes_training_overlap_without_relabeling_it(tmp_path):
    selection,manifest,train,result=fixture(tmp_path)
    row=json.loads(manifest.read_text())['images'][0];train.write_text(json.dumps(dict(images=[row])))
    with pytest.raises(ValueError,match='No disjoint test images'):
        prepare_manifest(tmp_path,selection,tmp_path/'source.json',train,tmp_path/'dev.json',tmp_path/'other.json')
