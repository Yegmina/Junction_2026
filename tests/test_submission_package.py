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
    compiled=task/'environment/backend/__pycache__/module.pyc';compiled.parent.mkdir();compiled.write_bytes(b'compiled-fixture')
    receipt=packager.package(task,[job],output)
    assert receipt['clean_opencode_playthrough'] and receipt['runs'][0]['reward']==.5
    with (zipfile.ZipFile(output) if extension=='.zip' else tarfile.open(output)) as archive:
        read=archive.read if extension=='.zip' else lambda name:archive.extractfile(name).read()
        log=read('aerorozum-submission/run-evidence/job/trial/agent/opencode-public.jsonl')
        assert b'drone status' in log and b'private-reasoning-fixture' not in log
        assert read('aerorozum-submission/run-evidence/job/trial/verifier/reward.txt')==b'0.50000000\n'
        assert read('aerorozum-submission/thermal-investigation/environment/backend/__pycache__/module.pyc')==b'compiled-fixture'


def test_bundle_rejects_secret_and_score_disagreement(tmp_path):
    task,job=fixture(tmp_path);output=tmp_path/'bundle.tar.gz'
    (task/'instruction.md').write_text('private-credential-fixture')
    with pytest.raises(ValueError,match='Credential scan'):packager.package(task,[job],output,['private-credential-fixture'])
    assert not output.exists()
    (task/'instruction.md').write_text('public task')
    (job/'trial/verifier/reward.txt').write_text('1.0')
    with pytest.raises(ValueError,match='disagree'):packager.package(task,[job],output)
    assert not output.exists()


def test_comparison_audit_is_bound_to_packaged_environment_and_actual_run(tmp_path):
    task,job=fixture(tmp_path)
    (task/'task.toml').write_text('[task]\nname="fixture"\nversion="1"\n')
    trial=job/'trial';instance=dict(seed=17,budget_s=180)
    (trial/'verifier/ledger.json').write_text(json.dumps(dict(instance=instance)))
    import importlib.util
    spec=importlib.util.spec_from_file_location('audit_fixture',Path(packager.__file__).with_name('audit_harbor_contract.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    contract_hash=module.fingerprint(task)['sha256']
    entry=dict(job=job.name,result_sha256=packager.sha(trial/'result.json'),instance=instance)
    audit=dict(pair_comparable=True,contract_identical=True,native_stack_identical=True,instance_identical=True,
        left_contract_sha256=contract_hash,right_contract_sha256=contract_hash,trials=dict(left=entry,right=entry))
    proof=tmp_path/'audit.json';proof.write_text(json.dumps(audit))
    receipt=packager.package(task,[job],tmp_path/'valid.zip',audits=[proof])
    assert len(receipt['comparison_audits'])==1
    (task/'instruction.md').write_text('Changed task instruction')
    with pytest.raises(ValueError,match='does not describe the bundled task'):
        packager.package(task,[job],tmp_path/'invalid.zip',audits=[proof])


def test_inline_camera_payload_is_preserved_as_hashed_image_instead_of_base64_log_text(tmp_path):
    import base64,hashlib
    source=tmp_path/'events.jsonl';output=tmp_path/'public.jsonl'
    payload=b'\x89PNG\r\n\x1a\n'+b'image-fixture'
    url='data:image/png;base64,'+base64.b64encode(payload).decode()
    source.write_text(json.dumps(dict(type='tool_use',part=dict(type='tool',state=dict(attachments=[dict(url=url)]))))+'\n')
    result=packager.public_events(source,output);record=json.loads(output.read_text())
    relative=record['part']['state']['attachments'][0]['url']
    assert relative=='inline-images/'+hashlib.sha256(payload).hexdigest()+'.png'
    assert (tmp_path/relative).read_bytes()==payload
    assert result['externalized_image_payloads']==1 and 'base64' not in output.read_text()


def test_exact_private_value_inside_binary_image_is_still_rejected(tmp_path):
    task,job=fixture(tmp_path);(task/'environment/frame.png').write_bytes(b'\x89PNG\r\n\x1a\nprivate-binary-fixture')
    with pytest.raises(ValueError,match='Credential scan'):
        packager.package(task,[job],tmp_path/'unsafe.zip',secrets=['private-binary-fixture'])


def test_separate_task_folder_survives_packaging_without_renaming_contract(tmp_path):
    task,job=fixture(tmp_path);output=tmp_path/'budgeted.zip'
    packager.package(task,[job],output,task_folder='thermal-investigation-budgeted')
    with zipfile.ZipFile(output) as archive:
        prefix='aerorozum-submission/'
        assert archive.read(prefix+'thermal-investigation-budgeted/instruction.md')==b'fixture'
        assert b'./thermal-investigation-budgeted' in archive.read(prefix+'README.md')
    with pytest.raises(ValueError,match='single safe directory'):
        packager.package(task,[job],tmp_path/'unsafe.zip',task_folder='../outside')
