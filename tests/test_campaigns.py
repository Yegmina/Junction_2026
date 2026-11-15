import asyncio
import copy
import json
from pathlib import Path
import pytest
from aerorozum.campaigns import run_campaign,evaluate_run,export_policy_trajectories,run_campaign_batch
from aerorozum.environment import Campaign


CFG={'campaign':{'max_actions':16},'camera':{'width':64,'height':64},'gemini':{'teacher_model':'fake-teacher','bulk_model':'fake-bulk'}}


class Provider:
    def __init__(self,actions):self.actions=iter(actions);self.calls=[]
    async def generate(self,role,prompt,images,model):
        self.calls.append({'prompt':prompt,'images':list(images),'model':model})
        assert all(Path(p).is_file() for p in images)
        value=next(self.actions)
        if isinstance(value,Exception):raise value
        return value


def action(name,**arguments):return {'action':name,'arguments':arguments,'evidence_note':'Visible evidence only'}


def test_teacher_uses_public_last_three_images_and_no_evaluator(tmp_path,monkeypatch):
    def forbidden(*args):raise AssertionError('Teacher must never request evaluator truth')
    monkeypatch.setattr(Campaign,'evaluator_snapshot',forbidden)
    provider=Provider([action('launch'),action('capture'),action('capture'),action('capture'),action('capture'),action('submit_report')])
    original=copy.deepcopy(CFG)
    result=asyncio.run(run_campaign(CFG,run_dir=tmp_path,provider=provider,allow_demo=True))
    assert result['status']=='completed' and result['demo']
    assert CFG==original
    assert max(len(c['images']) for c in provider.calls)==3
    assert len(provider.calls[-1]['images'])==3
    assert all('private-' not in c['prompt'] and 'targets"' not in c['prompt'] for c in provider.calls)
    assert not (tmp_path/'evaluator.json').exists()
    monkeypatch.undo()
    assert evaluate_run(tmp_path)['demo']


def test_resume_replays_sensor_ids_and_does_not_repeat_provider_calls(tmp_path):
    first=Provider([action('launch'),action('capture'),RuntimeError('temporary unavailable')])
    failed=asyncio.run(run_campaign(CFG,seed=4,run_dir=tmp_path,provider=first,allow_demo=True))
    assert failed['status']=='failed' and failed['actions']==2
    before=json.loads((tmp_path/'state.json').read_text())
    observation=before['trajectory'][1]['result']['observation']
    second=Provider([action('submit_report')])
    done=asyncio.run(run_campaign(CFG,seed=4,run_dir=tmp_path,provider=second,allow_demo=True))
    assert done['status']=='completed' and len(second.calls)==1
    assert done['campaign']['observations'][0]['id']==observation['id']
    third=Provider([])
    assert asyncio.run(run_campaign(CFG,seed=4,run_dir=tmp_path,provider=third,allow_demo=True))['resumed']
    assert not third.calls
    with pytest.raises(ValueError,match='differs'):
        asyncio.run(run_campaign(CFG,seed=5,run_dir=tmp_path,provider=third,allow_demo=True))


def test_invalid_config_actions_cannot_bypass_decision_cap(tmp_path):
    provider=Provider([action('launch',config={'max_actions':999})]*3)
    result=asyncio.run(run_campaign(CFG,run_dir=tmp_path,provider=provider,max_actions=4,allow_demo=True))
    assert result['status']=='completed' and result['actions']==4
    assert not result['campaign']['airborne']
    state=json.loads((tmp_path/'state.json').read_text())
    assert state['trajectory'][-1]['forced']


def test_research_gate_before_any_api_calls(tmp_path):
    provider=Provider([])
    result=asyncio.run(run_campaign(CFG,run_dir=tmp_path,provider=provider))
    assert result['status']=='blocked' and not provider.calls
    with pytest.raises(ValueError,match='completed'):
        evaluate_run(tmp_path)


def test_export_demo_requires_explicit_opt_in_and_preserves_prefix(tmp_path):
    provider=Provider([action('launch'),action('capture'),action('submit_report')])
    asyncio.run(run_campaign(CFG,run_dir=tmp_path/'run',provider=provider,allow_demo=True))
    with pytest.raises(ValueError,match='Demo'):
        export_policy_trajectories([tmp_path/'run'],tmp_path/'rows.jsonl')
    result=export_policy_trajectories([tmp_path/'run'],tmp_path/'rows.jsonl',allow_demo=True)
    rows=[json.loads(line) for line in (tmp_path/'rows.jsonl').read_text().splitlines()]
    assert result['rows']==3 and not result['research_accepted']
    assert rows[1]['prefix_actions']==[{'action':'launch','arguments':{}}]
    assert rows[2]['images'] and rows[2]['campaign_id']
    heldout={**CFG,'campaign_split':'fast_test'}
    asyncio.run(run_campaign(heldout,run_dir=tmp_path/'test',provider=Provider([action('submit_report')]),allow_demo=True))
    with pytest.raises(ValueError,match='Held-out'):
        export_policy_trajectories([tmp_path/'test'],tmp_path/'leaked.jsonl',allow_demo=True)


def test_bad_action_schema_is_explicitly_invalid_and_bounded(tmp_path):
    provider=Provider([{'action':'evaluator_snapshot','arguments':{}},{'action':'launch','arguments':{},'truth':[]}])
    result=asyncio.run(run_campaign(CFG,run_dir=tmp_path,provider=provider,max_actions=3,allow_demo=True))
    assert result['status']=='completed' and not result['campaign']['airborne']
    state=json.loads((tmp_path/'state.json').read_text())
    assert state['trajectory'][0]['action']=='invalid_policy_action'


def test_full_batch_blocks_before_provider_or_quota_use(tmp_path,monkeypatch):
    from aerorozum import providers
    def forbidden(*args,**kwargs):raise AssertionError('Blocked batch must never construct provider')
    monkeypatch.setattr(providers,'GeminiProvider',forbidden)
    frozen={'config_hash':'frozen','counts':{'images':30000},'campaigns':[{'id':'c1','split':'train','seed':1}]}
    result=asyncio.run(run_campaign_batch(frozen,{'config_hash':'frozen','fields':[]},cfg=CFG,run_root=tmp_path))
    assert result['status']=='blocked' and result['completed_campaigns']==0
