import pytest
from aerorozum.common import digest
from aerorozum.providers import request_identity, generation_settings


def test_reasoning_level_cannot_reuse_paid_low_effort_answer():
    args=dict(role='study',prompt='frame',images=['sha'],model='gemini-robotics-er-2-preview',max_output_tokens=65536)
    assert request_identity({'thinking_level':'high'},**args)!=request_identity({'thinking_level':'low'},**args)
    assert request_identity({'thinking_level':'HIGH'},**args)==request_identity({'thinking_level':'high'},**args)
    old=digest(dict(role='study',prompt='frame',images=['sha'],model=args['model'],max_output_tokens=65536,temperature=.1,api_version='v1beta',thinking_budget=0))
    assert request_identity({'thinking_budget':0},**args)==old


def test_high_reasoning_uses_level_and_large_shared_output_allowance():
    result=generation_settings({'thinking_level':'high'},65536)
    assert result['thinking_config']=={'thinking_level':'HIGH'} and result['max_output_tokens']==65536
    assert generation_settings({'thinking_budget':0},2048)['thinking_config']=={'thinking_budget':0}
    with pytest.raises(ValueError,match='not both'):
        generation_settings({'thinking_level':'high','thinking_budget':0},65536)


def test_label_cache_keeps_native_reasoning_levels_separate():
    from types import SimpleNamespace
    from aerorozum.labelling import inference_cache_key
    args=({'source':{'generation':'123'}},'pixels')
    high=SimpleNamespace(cfg={'thinking_level':'high','reserve_output_tokens':65536})
    minimal=SimpleNamespace(cfg={'thinking_level':'minimal','reserve_output_tokens':65536})
    assert inference_cache_key(*args,high)!=inference_cache_key(*args,minimal)
