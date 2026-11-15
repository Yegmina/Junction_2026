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


def test_native_proxy_preserves_conversation_roles_without_the_flattened_prompt(tmp_path,monkeypatch):
    monkeypatch.setenv('AEROROZUM_NATIVE_CHAT','1');monkeypatch.setattr(proxy,'OUTPUT',tmp_path)
    original=dict(messages=[dict(role='user',content='Inspect'),dict(role='tool',tool_call_id='one',content='camera result')],tools=[])
    received=[]
    def generate(request,timeout):
        received.append(json.loads(request.data));assert request.get_header('X-aero-native-session')
        return io.BytesIO(json.dumps(dict(id='native-fixture',choices=[dict(message=dict(role='assistant',content='Done.'))],usage=dict(prompt_tokens=42,completion_tokens=8))).encode())
    monkeypatch.setattr(proxy.urllib.request,'urlopen',generate)
    message,raw=proxy.infer(original)
    assert received[0]['messages']==original['messages'] and 'prompt' not in received[0]
    assert message['content']=='Done.' and raw['input_tokens']==42


def test_native_upstream_context_error_keeps_request_and_machine_readable_diagnostic(tmp_path,monkeypatch):
    monkeypatch.setenv('AEROROZUM_NATIVE_CHAT','1');monkeypatch.setattr(proxy,'OUTPUT',tmp_path)
    detail=dict(message='This model\'s maximum context length is 24576 tokens.',code='context_length_exceeded')
    def fail(request,timeout):
        raise proxy.urllib.error.HTTPError(request.full_url,400,'Bad Request',{},io.BytesIO(json.dumps(dict(detail=detail)).encode()))
    monkeypatch.setattr(proxy.urllib.request,'urlopen',fail)
    with pytest.raises(proxy.UpstreamError) as error:
        proxy.infer(dict(messages=[dict(role='user',content='public fixture')],tools=[]))
    assert error.value.status==400 and error.value.error['code']=='context_length_exceeded'
    assert 'maximum context length' in error.value.error['message']
    assert json.loads(next(tmp_path.glob('*-request.json')).read_text())['request']['messages'][0]['content']=='public fixture'
    assert json.loads(next(tmp_path.glob('*-error.json')).read_text())['status']==400
