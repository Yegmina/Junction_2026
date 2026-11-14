import json
from pathlib import Path
import pytest
from aerorozum.grpo_continuation import validate_continuation


def fixture(tmp_path):
    result={'reward_signal':{'task_signal_verified':True}}
    for name in ('sft','grpo'):
        path=tmp_path/'training'/name/'adapter';path.mkdir(parents=True)
        (path/'adapter_config.json').write_text('{}');(path/'adapter_model.safetensors').write_bytes(('weights-'+name).encode())
        summary=dict(status='completed',stage='stream_'+name,model='test/model',seed=17,manifest_hash='frozen',
            lora_scope='language_visual_projection',optimizer_updates=10,nonzero_gradient_checks=10,adapter=str(path.relative_to(tmp_path)))
        (path.parent/'summary.json').write_text(json.dumps(summary));result[name]=summary
    file=tmp_path/'result.json';file.write_text(json.dumps(dict(status='completed',results=result)))
    return file,dict(root=tmp_path,model='test/model',seed=17,manifest_hash='frozen',lora_scope='language_visual_projection')


def test_continuation_keeps_completed_sft_lineage_and_records_actual_weight_hashes(tmp_path):
    file,kwargs=fixture(tmp_path);parent,receipt=validate_continuation(file,**kwargs)
    assert parent==tmp_path/'training/grpo/adapter'
    assert len(receipt['sft_adapter_sha256'])==len(receipt['grpo_adapter_sha256'])==64
    assert 'Fresh optimizer' in receipt['optimizer_state']


def test_continuation_rejects_different_data_and_incomplete_sft(tmp_path):
    file,kwargs=fixture(tmp_path)
    with pytest.raises(ValueError,match='parent does not match'):validate_continuation(file,**{**kwargs,'manifest_hash':'different'})
    (tmp_path/'training/sft/summary.json').unlink()
    with pytest.raises(ValueError,match='completion summary'):validate_continuation(file,**kwargs)


def test_continuation_rejects_kl_only_parent_and_altered_summary(tmp_path):
    file,kwargs=fixture(tmp_path);d=json.loads(file.read_text());d['results']['reward_signal']['task_signal_verified']=False;file.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='task-reward variation'):validate_continuation(file,**kwargs)
    d['results']['reward_signal']['task_signal_verified']=True;d['results']['grpo']['optimizer_updates']=20;file.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='summary disagree'):validate_continuation(file,**kwargs)
