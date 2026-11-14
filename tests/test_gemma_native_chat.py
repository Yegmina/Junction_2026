import base64,io,json
from PIL import Image
import pytest
from cloud.gemma_native_chat import normalize_messages,public_message


def test_native_history_keeps_images_with_their_tool_response_and_restores_private_state():
    buffer=io.BytesIO();Image.new('RGB',(3,2),(10,20,30)).save(buffer,format='PNG')
    image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
    source=[dict(role='user',content='Inspect the saved frame'),
        dict(role='assistant',content='',tool_calls=[dict(id='prior',type='function',function=dict(name='read',arguments='{"filePath":"/workspace/frame.png"}'))]),
        dict(role='tool',tool_call_id='prior',content=[dict(type='text',text='Saved camera image'),dict(type='image_url',image_url=dict(url=image))])]
    messages,hashes=normalize_messages(source,{'prior':{'thinking':'private fixture'}})
    assert messages[2]['role']=='tool' and messages[2]['tool_call_id']=='prior'
    assert messages[2]['content'][1]['image'].getpixel((0,0))==(10,20,30)
    assert messages[1]['tool_calls'][0]['function']['arguments']=={'filePath':'/workspace/frame.png'}
    assert messages[1]['thinking']=='private fixture' and len(hashes)==1
    assert 'thinking' not in source[1] and isinstance(source[1]['tool_calls'][0]['function']['arguments'],str)


def test_native_public_output_contains_exact_actions_and_excludes_thinking():
    parsed=dict(role='assistant',thinking='private fixture',reasoning_content='another private fixture',
        content=[dict(type='reasoning',text='hidden fixture'),dict(type='text',text='Read current telemetry.')],
        tool_calls=[dict(function=dict(name='bash',arguments=dict(command='drone status')))])
    message=public_message(parsed);serialized=json.dumps(message)
    assert 'fixture' not in serialized and message['content']=='Read current telemetry.'
    assert json.loads(message['tool_calls'][0]['function']['arguments'])=={'command':'drone status'}


def test_native_codec_rejects_remote_image_fetch_and_invalid_argument_types():
    with pytest.raises(ValueError,match='inline'):
        normalize_messages([dict(role='tool',content=[dict(type='image_url',image_url=dict(url='https://example.org/a.png'))])])
    with pytest.raises(ValueError,match='Invalid native tool'):
        public_message(dict(tool_calls=[dict(function=dict(name='bash',arguments=[]))]))
