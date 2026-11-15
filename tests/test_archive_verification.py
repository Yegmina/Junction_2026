import hashlib,importlib.util,sys,zipfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'cloud'))
spec=importlib.util.spec_from_file_location('archive_verification',ROOT/'cloud/verify_submission_archive.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_extraction_verifies_hashes_and_restores_executable_permissions(tmp_path):
    archive=tmp_path/'bundle.zip';payload=b'#!/bin/sh\nexit 0\n';name='task/solution/solve.sh'
    with zipfile.ZipFile(archive,'w') as z:
        item=zipfile.ZipInfo('aerorozum-submission/'+name);item.external_attr=0o100755<<16;z.writestr(item,payload)
        z.writestr('aerorozum-submission/SHA256SUMS',hashlib.sha256(payload).hexdigest()+'  '+name+'\n')
    assert module.extract_verified(archive,tmp_path/'extract')==1
    assert (tmp_path/'extract/aerorozum-submission'/name).stat().st_mode&0o777==0o755


@pytest.mark.parametrize('name,payload,manifest',[
    ('../escape',b'x',hashlib.sha256(b'x').hexdigest()),
    ('task/file',b'changed',hashlib.sha256(b'original').hexdigest())])
def test_extraction_rejects_path_traversal_or_corrupted_payload(tmp_path,name,payload,manifest):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('aerorozum-submission/'+name,payload)
        z.writestr('aerorozum-submission/SHA256SUMS',manifest+'  '+name+'\n')
    with pytest.raises(ValueError):module.extract_verified(archive,tmp_path/'extract')
    assert not (tmp_path/'escape').exists()
