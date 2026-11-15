import pytest
from aerorozum import common
from aerorozum.jobs import reserve,settle,cost_status
from aerorozum.experiment import freeze
from aerorozum.cloud import plan_schedule


def test_secret_dotenv_is_literal_and_never_shell(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path/'project');(tmp_path/'project').mkdir()
    monkeypatch.delenv('TEST_SECRET',raising=False)
    (tmp_path/'.env').write_text('TEST_SECRET="$(touch DONT_EXECUTE);literal"\n')
    assert common.secret('TEST_SECRET')=='$(touch DONT_EXECUTE);literal'
    assert not (tmp_path/'DONT_EXECUTE').exists()


def test_budget_counts_existing_reservations_and_settlements(tmp_path):
    a=reserve('google_cloud',600,{},root=tmp_path)
    with pytest.raises(ValueError):reserve('google_cloud',401,{},root=tmp_path)
    settle(a,100,root=tmp_path)
    reserve('google_cloud',900,{},root=tmp_path)
    with pytest.raises(ValueError):reserve('google_cloud',1,{},root=tmp_path)
    with pytest.raises(ValueError):settle(a,0,root=tmp_path)
    assert cost_status(tmp_path)[0]['spent_eur']==100


def test_frozen_manifest_preserves_all_counts_and_pairs(tmp_path,monkeypatch):
    monkeypatch.setenv('AEROROZUM_DATA_ROOT',str(tmp_path))
    d=freeze();assert len(d['physics'])==184
    assert len(d['campaigns'])==512+64+72+24
    assert len({r['id'] for r in d['campaigns']})==len(d['campaigns'])
    test=[r for r in d['campaigns'] if r['split']=='fast_test']
    assert len({r['background_group'] for r in test})==24
    assert {r['layout'] for r in test}=={'A','relocated_B','target_free'}
    assert freeze()['config_hash']==d['config_hash']


def test_no_eta_without_full_measured_inputs():
    assert plan_schedule({})['estimated_finish_s'] is None


def test_provider_fails_without_key_instead_of_adc(monkeypatch):
    import aerorozum.providers as p
    monkeypatch.setattr(p,'secret',lambda *a:None)
    with pytest.raises(RuntimeError,match='fallback is disabled'):p.GeminiProvider()


def test_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):common.save(tmp_path/'x.json',{'bad':float('nan')})
    assert not (tmp_path/'x.json').exists()
