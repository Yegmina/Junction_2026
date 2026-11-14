import importlib.util
from pathlib import Path
import shutil

spec=importlib.util.spec_from_file_location('contract_audit',Path(__file__).resolve().parents[1]/'cloud/audit_harbor_contract.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def pair(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b'
    for name,text in [('instruction.md','Inspect and return'),('environment/Dockerfile','FROM test'),
                      ('environment/backend/evaluator.py','frozen grading'),('tests/test.sh','verify')]:
        p=a/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    (a/'task.toml').write_text('[task]\nversion="0.3"\n[agent]\ntimeout_sec=1800\n')
    shutil.copytree(a,b);return a,b


def test_reference_only_revision_preserves_contract_without_claiming_unfinished_runs_comparable(tmp_path):
    a,b=pair(tmp_path);(b/'task.toml').write_text('[task]\nversion="0.3.1"\n[agent]\ntimeout_sec=1800\n')
    (b/'solution').mkdir();(b/'solution/reference.py').write_text('A new public-image reference')
    report=module.audit(a,b)
    assert report['contract_identical'] and report['pair_comparable'] is None


def test_changed_evaluator_asset_or_executable_permission_invalidates_comparison(tmp_path):
    a,b=pair(tmp_path);p=b/'environment/backend/evaluator.py';p.write_text('different grading')
    assert not module.audit(a,b)['contract_identical']
    p.write_text('frozen grading');p.chmod(0o755)
    assert not module.audit(a,b)['contract_identical']


def test_resource_limit_changes_cannot_be_ignored_as_version_metadata(tmp_path):
    a,b=pair(tmp_path);(b/'task.toml').write_text('[task]\nversion="0.3.1"\n[agent]\ntimeout_sec=3600\n')
    report=module.audit(a,b)
    assert not report['contract_identical'] and not report['configuration_identical']
