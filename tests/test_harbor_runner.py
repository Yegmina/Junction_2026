import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('harbor_runner',Path(__file__).resolve().parents[1]/'cloud/run_harbor_trial.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('flags,expected',[
    ([],{}),(['--phase','dawn','--budget','75'],dict(AEROROZUM_TASK_PHASE='dawn',AEROROZUM_TASK_BUDGET='75'))])
def test_runner_preserves_task_preset_unless_explicitly_overridden(monkeypatch,flags,expected):
    monkeypatch.delenv('AEROROZUM_TASK_PHASE',raising=False);monkeypatch.delenv('AEROROZUM_TASK_BUDGET',raising=False)
    monkeypatch.setattr(sys,'argv',['runner','--task','task','--job-name','fixture','--local-gemma',*flags]);calls=[]
    def invoke(args,env):
        calls.append(dict(env=env,configuration=json.loads(Path(args[-1]).read_text())));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',invoke)
    assert runner.main()==0
    assert {k:v for k,v in calls[0]['env'].items() if k in ('AEROROZUM_TASK_PHASE','AEROROZUM_TASK_BUDGET')}==expected
    assert calls[0]['configuration']['agents'][0]['name']=='opencode'


def test_invalid_flight_budget_fails_before_starting_harbor(monkeypatch):
    monkeypatch.setattr(sys,'argv',['runner','--task','task','--job-name','fixture','--local-gemma','--budget','59'])
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('Harbor must not start'))
    with pytest.raises(SystemExit):runner.main()


def test_frontier_output_allowance_is_explicit_in_saved_harness_configuration(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','credential-fixture')
    monkeypatch.setattr(sys,'argv',['runner','--task','task','--job-name','fixture','--output-tokens','32768'])
    captured=[]
    def invoke(args,env):
        option=next(x for x in args if x.startswith('opencode_config='))
        captured.append(json.loads(option.split('=',1)[1]));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',invoke)
    assert runner.main()==0
    assert captured[0]['provider']['google']['models']['gemini-4-argon']['limit']['output']==32768


def test_explicit_thinking_level_is_saved_in_model_options(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','credential-fixture')
    monkeypatch.setattr(sys,'argv',['runner','--task','task','--job-name','fixture','--thinking-level','low'])
    captured=[]
    def invoke(args,env):
        option=next(x for x in args if x.startswith('opencode_config='))
        captured.append(json.loads(option.split('=',1)[1]));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',invoke)
    assert runner.main()==0
    assert captured[0]['provider']['google']['models']['gemini-4-argon']['options']=={'thinkingConfig': {'thinkingLevel':'low'}}


def test_bounded_flight_agent_does_not_inherit_automatic_high_thinking(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','credential-fixture')
    monkeypatch.setattr(sys,'argv',['runner','--task','task','--job-name','fixture','--thinking-budget','1024','--flight-agent'])
    captured=[]
    def invoke(args,env):
        option=next(x for x in args if x.startswith('opencode_config='))
        captured.append(json.loads(option.split('=',1)[1]));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',invoke)
    assert runner.main()==0
    model=captured[0]['provider']['google']['models']['gemini-4-argon']
    assert model['reasoning'] is False
    assert model['options']['thinkingConfig']['thinkingBudget']==1024
    assert 'drone collect' in captured[0]['agent']['build']['prompt']
