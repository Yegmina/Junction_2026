import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('submission_packager',Path(__file__).resolve().parents[1]/'cloud/package_harbor_submission.py')
packager=importlib.util.module_from_spec(spec);spec.loader.exec_module(packager)


def fixture(tmp_path):
    task=tmp_path/'task'
    for name in ('instruction.md','task.toml','environment/Dockerfile','tests/test.sh','solution/solve.sh',
                 'environment/backend/runtime/scene_factory.py','environment/backend/assets/provenance/response-kernels.json'):
        path=task/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    job=tmp_path/'job';trial=job/'trial';(trial/'verifier').mkdir(parents=True);(trial/'agent').mkdir()
    (trial/'result.json').write_text(json.dumps(dict(config=dict(agent=dict(name='opencode',model_name='test/model')),verifier_result=dict(rewards=dict(reward=.5)))))
    (trial/'verifier/reward.txt').write_text('0.50000000\n');(trial/'verifier/details.json').write_text('{"completed":true}')
    (trial/'agent/opencode.txt').write_text('\n'.join(json.dumps(x) for x in [dict(type='reasoning',part=dict(type='reasoning',text='private-reasoning-fixture')),dict(type='tool_use',part=dict(type='tool',tool='bash',state=dict(input=dict(command='drone status'))))]))
    return task,job


@pytest.mark.parametrize('extension',['.tar.gz','.zip'])
def test_bundle_keeps_real_score_and_actions_without_private_reasoning(tmp_path,extension):
    task,job=fixture(tmp_path);output=tmp_path/('bundle'+extension)
    receipt=packager.package(task,[job],output)
    assert receipt['clean_opencode_playthrough'] and receipt['runs'][0]['reward']==.5
    with (zipfile.ZipFile(output) if extension=='.zip' else tarfile.open(output)) as archive:
        read=archive.read if extension=='.zip' else lambda name:archive.extractfile(name).read()
        log=read('aerorozum-submission/run-evidence/job/trial/agent/opencode-public.jsonl')
        assert b'drone status' in log and b'private-reasoning-fixture' not in log
        assert read('aerorozum-submission/run-evidence/job/trial/verifier/reward.txt')==b'0.50000000\n'


def test_bundle_rejects_secret_and_score_disagreement(tmp_path):
    task,job=fixture(tmp_path);output=tmp_path/'bundle.tar.gz'
    (task/'instruction.md').write_text('private-credential-fixture')
    with pytest.raises(ValueError,match='Credential scan'):packager.package(task,[job],output,['private-credential-fixture'])
    assert not output.exists()
    (task/'instruction.md').write_text('public task')
    (job/'trial/verifier/reward.txt').write_text('1.0')
    with pytest.raises(ValueError,match='disagree'):packager.package(task,[job],output)
    assert not output.exists()
