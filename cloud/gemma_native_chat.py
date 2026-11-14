"""Native Gemma chat/tool protocol over the existing private inference model.

The canonical processor owns tool serialization and response parsing. No tool
executes here. Thinking stays private and can be restored only through this
process's previously issued tool-call IDs; public responses contain decisions
and tool calls, never the thinking channel.
"""
import asyncio
import base64
from collections import OrderedDict
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid
from fastapi import HTTPException,Request
from fastapi.responses import StreamingResponse
from PIL import Image

PROTOCOL='gemma4-native-tools-v1'
THINKING_KEYS=('thinking','reasoning','reasoning_content')


def normalize_messages(messages,thoughts=None):
    if not isinstance(messages,list) or not 1<=len(messages)<=256:raise ValueError('Expected 1..256 chat messages')
    normalized=[];hashes=[]
    for item in messages:
        if not isinstance(item,dict):raise ValueError('Invalid chat message')
        role=item.get('role')
        if role not in ('system','developer','user','assistant','tool'):raise ValueError('Unsupported message role')
        message=dict(role=role);content=item.get('content') or ''
        if isinstance(content,str):message['content']=content
        elif isinstance(content,list):
            parts=[]
            for part in content:
                if part.get('type') in ('text','input_text','output_text'):parts.append(dict(type='text',text=part.get('text','')))
                elif part.get('type')=='image_url':
                    url=part['image_url']['url']
                    if not url.startswith('data:image/') or ';base64,' not in url:raise ValueError('Only inline images are supported')
                    encoded=url.split(';base64,',1)[1]
                    if len(encoded)>4_000_000:raise ValueError('Image payload too large')
                    payload=base64.b64decode(encoded,validate=True)
                    with Image.open(io.BytesIO(payload)) as image:
                        if image.width*image.height>2_000_000:raise ValueError('Image dimensions too large')
                        parts.append(dict(type='image',image=image.convert('RGB')))
                    hashes.append(hashlib.sha256(payload).hexdigest())
                    if len(hashes)>16:raise ValueError('At most sixteen images fit this inference profile')
                else:raise ValueError('Unsupported content part')
            message['content']=parts
        else:raise ValueError('Unsupported message content')
        for name in ('tool_call_id','name'):
            if name in item:message[name]=item[name]
        if item.get('tool_calls'):
            calls=copy.deepcopy(item['tool_calls'])
            for call in calls:
                args=call['function'].get('arguments')
                if isinstance(args,str):args=json.loads(args)
                if args is None:args={}
                if not isinstance(args,dict):raise ValueError('Tool arguments must be an object')
                call['function']['arguments']=args
            message['tool_calls']=calls
            if thoughts:
                for call in calls:
                    if call.get('id') in thoughts:
                        message.update(thoughts[call['id']]);break
        normalized.append(message)
    return normalized,hashes


def public_message(parsed):
    content=parsed.get('content')
    if isinstance(content,list):content=''.join(p.get('text','') for p in content if isinstance(p,dict) and p.get('type') in ('text','output_text'))
    if content is not None and not isinstance(content,str):raise ValueError('Invalid public response content')
    message=dict(role='assistant',content=content or None);calls=[]
    for item in parsed.get('tool_calls') or []:
        function=item['function'];arguments=function.get('arguments')
        if arguments is None:arguments={}
        if not isinstance(function.get('name'),str) or not function['name'] or not isinstance(arguments,dict):raise ValueError('Invalid native tool response')
        calls.append(dict(id='call_'+uuid.uuid4().hex[:24],type='function',function=dict(name=function['name'],arguments=json.dumps(arguments))))
    if calls:message['tool_calls']=calls
    return message


def infer(app,body,session):
    import torch
    processor=app.state.processor;tokenizer=processor.tokenizer
    thoughts={call:thought for (scope,call),thought in app.state.native_thoughts.items() if scope==session}
    messages,image_hashes=normalize_messages(body.get('messages'),thoughts)
    tools=copy.deepcopy(body.get('tools') or [])
    for tool in tools:
        if tool.get('type')!='function':raise ValueError('Only function tools are supported')
        tool['function'].setdefault('description','');tool['function'].setdefault('parameters',dict(type='object',properties={}))
    thinking=os.environ.get('AEROROZUM_NATIVE_THINKING','1')=='1'
    limit=min(4096,max(64,int(body.get('max_tokens') or body.get('max_completion_tokens') or 4096)))
    inputs=processor.apply_chat_template(messages,tools=tools or None,enable_thinking=thinking,
        add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
    length=inputs['input_ids'].shape[-1]
    if length+limit>24576:raise ValueError('Native chat context exceeds the declared 24576-token inference profile')
    prefix=inputs['input_ids'][0].detach().cpu().tolist();device=app.state.model.get_input_embeddings().weight.device
    inputs={k:(v.to(device=device,dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(device)) if torch.is_tensor(v) else v for k,v in inputs.items()}
    stops=app.state.model.generation_config.eos_token_id
    stops=list(stops) if isinstance(stops,(list,tuple)) else [stops] if stops is not None else []
    for token in ('<turn|>','<|tool_response>'):
        index=tokenizer.convert_tokens_to_ids(token)
        if index is not None and index!=tokenizer.unk_token_id:stops.append(index)
    began=time.monotonic()
    with torch.inference_mode():generated=app.state.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=limit,use_cache=True,eos_token_id=sorted(set(stops)))
    tokens=generated[0,length:].detach().cpu().tolist();identifier='chatcmpl-'+uuid.uuid4().hex
    folder=Path(os.environ.get('AEROROZUM_NATIVE_LOGS','/opt/aerorozum/artifacts/private/native-inference'));folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    record=dict(id=identifier,protocol=PROTOCOL,model=os.environ.get('AEROROZUM_MISSION_MODEL','google/gemma-4-12B-it'),
        adapter_sha256=app.state.adapter_sha256,image_sha256=image_hashes,prompt_token_ids=prefix,output_token_ids=tokens,
        output_text=tokenizer.decode(tokens,skip_special_tokens=False),input_tokens=length,output_tokens=len(tokens),max_output_tokens=limit,
        thinking_enabled=thinking,elapsed_s=time.monotonic()-began)
    path=folder/(identifier+'.json');path.write_text(json.dumps(record))
    try:
        parsed=tokenizer.parse_response(tokens,prefix=prefix,tools=tools or None);message=public_message(parsed)
    except Exception as error:
        record['parse_error_type']=type(error).__name__;path.write_text(json.dumps(record))
        raise ValueError('Native response parsing failed; private generation record '+identifier) from None
    thought={key:parsed[key] for key in THINKING_KEYS if parsed.get(key)}
    if thought:
        for call in message.get('tool_calls',[]):app.state.native_thoughts[(session,call['id'])]=thought
        while len(app.state.native_thoughts)>256:app.state.native_thoughts.popitem(last=False)
    finish='tool_calls' if message.get('tool_calls') else 'length' if len(tokens)>=limit else 'stop'
    usage=dict(prompt_tokens=length,completion_tokens=len(tokens),total_tokens=length+len(tokens))
    record.update(public_message=message,finish_reason=finish);path.write_text(json.dumps(record))
    return dict(id=identifier,object='chat.completion',created=int(time.time()),model=record['model'],
        choices=[dict(index=0,message=message,finish_reason=finish)],usage=usage,
        aerorozum=dict(protocol=PROTOCOL,adapter_sha256=record['adapter_sha256'],thinking_enabled=thinking))


def attach(app):
    app.state.native_thoughts=OrderedDict();app.state.native_protocol=PROTOCOL
    @app.get('/v1/models')
    def models():return dict(object='list',data=[dict(id=os.environ.get('AEROROZUM_MISSION_MODEL','google/gemma-4-12B-it'),object='model',owned_by='aerorozum')])
    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        if not 0<int(request.headers.get('content-length','0'))<=12_000_000:raise HTTPException(413,'Invalid chat payload size')
        body=await request.json()
        session=request.headers.get('x-aero-native-session','')
        if not 32<=len(session)<=128:raise HTTPException(400,'A private proxy session is required')
        async with app.state.lock:
            try:result=await asyncio.to_thread(infer,app,body,session)
            except ValueError as error:raise HTTPException(422,str(error)) from None
        if not body.get('stream'):return result
        def events():
            message=result['choices'][0]['message'];delta=dict(message)
            if delta.get('tool_calls'):delta['tool_calls']=[dict(index=i,**call) for i,call in enumerate(delta['tool_calls'])]
            for choice in [dict(index=0,delta=delta,finish_reason=None),dict(index=0,delta={},finish_reason=result['choices'][0]['finish_reason'])]:
                chunk=dict(id=result['id'],object='chat.completion.chunk',created=result['created'],model=result['model'],choices=[choice])
                if choice['finish_reason'] is not None:chunk['usage']=result['usage']
                yield 'data: '+json.dumps(chunk)+'\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(events(),media_type='text/event-stream')
