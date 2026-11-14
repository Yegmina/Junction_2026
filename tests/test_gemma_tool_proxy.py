import importlib.util,io,json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('gemma_tool_proxy',Path(__file__).resolve().parents[1]/'cloud/gemma_openai_proxy.py')
proxy=importlib.util.module_from_spec(spec);spec.loader.exec_module(proxy)


def test_model_chosen_tool_is_translated_and_raw_data_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(proxy,'OUTPUT',tmp_path)
    calls=[]
    def generate(request,timeout):
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps(dict(output=dict(content='Read telemetry first.',tool_calls=[dict(name='bash',arguments=dict(command='drone status'))]),input_tokens=30,output_tokens=20)).encode())
    monkeypatch.setattr(proxy.urllib.request,'urlopen',generate)
    message,raw=proxy.infer(dict(messages=[dict(role='user',content='Check the flight status')],tools=[dict(type='function',function=dict(name='bash'))]))
    assert message['tool_calls'][0]['function']['name']=='bash'
    assert json.loads(message['tool_calls'][0]['function']['arguments'])=={'command':'drone status'}
    assert 'Check the flight status' in calls[0]['prompt'] and calls[0]['images']==[]
    assert len(list(tmp_path.glob('*-response.json')))==1
    assert raw['input_tokens']==30


def test_undeclared_model_tool_is_rejected_instead_of_executed(tmp_path,monkeypatch):
    monkeypatch.setattr(proxy,'OUTPUT',tmp_path)
    monkeypatch.setattr(proxy.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(json.dumps(dict(output=dict(tool_calls=[dict(name='private_truth',arguments={})]))).encode()))
    with pytest.raises(ValueError,match='invalid tool call'):
        proxy.infer(dict(messages=[],tools=[dict(type='function',function=dict(name='bash'))]))
    assert len(list(tmp_path.glob('*-response.json')))==1


def test_standard_openai_tool_envelope_preserves_the_model_arguments(tmp_path,monkeypatch):
    monkeypatch.setattr(proxy,'OUTPUT',tmp_path)
    monkeypatch.setattr(proxy.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(json.dumps(dict(output=dict(tool_calls=[dict(type='function',function=dict(name='bash',arguments='{"command":"drone wait --seconds 2"}'))]))).encode()))
    message,_=proxy.infer(dict(messages=[],tools=[dict(type='function',function=dict(name='bash'))]))
    assert json.loads(message['tool_calls'][0]['function']['arguments'])=={'command':'drone wait --seconds 2'}
