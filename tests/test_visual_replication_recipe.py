import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('visual_replication', Path(__file__).resolve().parents[1] / 'cloud/replicate_visual_recipe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def parent():
    topology = dict(lora_scope='language_visual_projection', microbatch=1, gradient_accumulation=4)
    return dict(status='completed', specification=dict(sft_steps=480, grpo_steps=48, learning_rate=3e-5),
                results={'sft': dict(topology), 'grpo': dict(topology)})


def test_replication_preserves_successful_recipe_and_declares_no_test_selection():
    original = parent();before = copy.deepcopy(original)
    result = module.recipe(original, 43)
    assert (result['seed'], result['sft_steps'], result['grpo_steps'], result['continuation_steps']) == (43, 480, 48, 96)
    assert result['microbatch'] * result['gradient_accumulation'] == 4
    assert not result['test_access'] and not result['automatic_promotion']
    assert original == before
    original['results']['grpo']['lora_scope'] = 'language'
    with pytest.raises(ValueError, match='scope'):
        module.recipe(original, 43)


def test_control_matches_update_count_without_claiming_compute_matching():
    result = module.recipe(parent(), 17, control=True)
    assert (result['sft_steps'], result['grpo_steps'], result['learning_rate']) == (96, 0, 1e-6)
    assert not result['compute_matching_claimed']
    with pytest.raises(ValueError, match='seed-17'):
        module.recipe(parent(), 91, control=True)
    changed = parent();changed['specification']['sft_steps'] = 32
    with pytest.raises(ValueError, match='recipe differs'):
        module.recipe(changed, 43)


def test_releasing_one_gpu_preserves_other_reservations_and_dispatch_settings(tmp_path):
    path = tmp_path / 'artifacts/pipeline/controls.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'pause_dispatch': False, 'external_gpu_leases': [{'id': 'live-service', 'gpu_ids': [0]}]}))
    module.lease(tmp_path, {'id': 'replication', 'gpu_ids': [2]})
    module.lease(tmp_path, {'id': 'other-replication', 'gpu_ids': [3]})
    module.lease(tmp_path, {'id': 'replication'}, remove=True)
    result = json.loads(path.read_text())
    assert [r['id'] for r in result['external_gpu_leases']] == ['live-service', 'other-replication']
    assert result['pause_dispatch'] is False


def test_no_reward_variation_prevents_continuation_and_still_releases_gpu(tmp_path, monkeypatch):
    out = tmp_path / 'experiment';out.mkdir()
    calls = []
    def run(command, **kwargs):
        calls.append(command[-1])
        module.save(out / 'initial/result.json', {'results': {'reward_signal': {'task_signal_verified': False}}})
    released = []
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module, 'lease', lambda *args, **kwargs: released.append(kwargs.get('remove')))
    module.worker({'root': str(tmp_path), 'output': str(out), 'seed': 43,
                   'input_hashes': {}, 'continuation_steps': 96, 'source_initial': '/frozen/source', 'lease': {'id': 'test'}})
    assert calls == ['initial'] and released == [True]
    assert (out / 'continuation-not-admitted.json').exists()
