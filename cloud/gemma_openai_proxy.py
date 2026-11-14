"""OpenAI tool-call adapter to the existing private GPU inference endpoint.

The model chooses every tool and argument. This adapter only translates the
protocol; it executes no drone actions and contains no target/reference data.
"""
import hashlib,json,os,time,uuid,urllib.request,urllib.error,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

MODEL=os.environ.get('AEROROZUM_PROXY_MODEL','gemma-4-12b-refined-grpo')
INFERENCE_URL=os.environ.get('AEROROZUM_INFERENCE_URL','http://127.0.0.1:18940/generate')
OUTPUT=Path(os.environ.get('AEROROZUM_PROXY_LOGS','/opt/aerorozum/harbor/gemma-proxy'))
INFERENCE_LOCK=threading.Lock()
NATIVE_SESSION=uuid.uuid4().hex
INFERENCE_TIMEOUT_S=float(os.environ.get('AEROROZUM_INFERENCE_TIMEOUT_S','900'))


class UpstreamError(ValueError):
    def __init__(self,status,body):
        self.status=status
        try:detail=json.loads(body).get('detail',{})
        except ValueError:detail={}
        if isinstance(detail,str):detail=dict(message=detail)
        if not isinstance(detail,dict):detail={}
        self.error=dict(message=detail.get('message') or f'Private inference returned HTTP {status}',
            type='invalid_request_error',code=detail.get('code'))
        super().__init__(self.error['message'])

def infer(request,peer='local-probe'):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if os.environ.get('AEROROZUM_NATIVE_CHAT')=='1':
        # Preserve roles, tool results and image placement for Gemma's canonical
        # template. One private proxy process belongs to one benchmark episode.
        identifier=uuid.uuid4().hex
        payload={**request,'stream':False}
        # Preserve the public request before inference, including failed calls.
        # Authentication headers and provider keys are never part of this body.
        request_path=OUTPUT/(identifier+'-request.json')
        request_path.write_text(json.dumps(dict(protocol='native-chat',model=MODEL,request=payload,upstream_timeout_s=INFERENCE_TIMEOUT_S)))
        request_path.chmod(0o600)
        session=hashlib.sha256((NATIVE_SESSION+peer).encode()).hexdigest()
        req=urllib.request.Request(INFERENCE_URL,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Aero-Native-Session':session})
        with INFERENCE_LOCK:
            try:
                with urllib.request.urlopen(req,timeout=INFERENCE_TIMEOUT_S) as response:result=json.load(response)
            except urllib.error.HTTPError as error:
                body=error.read(100000).decode(errors='replace')
                (OUTPUT/(identifier+'-error.json')).write_text(json.dumps(dict(status=error.code,body=body)))
                raise UpstreamError(error.code,body) from None
        (OUTPUT/(identifier+'-response.json')).write_text(json.dumps(result))
        usage=result.get('usage',{});limit=min(4096,max(64,request.get('max_tokens') or request.get('max_completion_tokens') or 4096))
        return result['choices'][0]['message'],dict(input_tokens=usage.get('prompt_tokens',0),output_tokens=usage.get('completion_tokens',0),requested_max_tokens=limit,finish_reason=result['choices'][0].get('finish_reason'))
    messages=[];images=[]
    tools=request.get('tools',[])
    for message in request.get('messages',[]):
        content=message.get('content') or '';texts=[]
        if isinstance(content,str):texts=[content]
        elif isinstance(content,list):
            for item in content:
                if item.get('type')=='text':texts.append(item.get('text',''))
                elif item.get('type')=='image_url':
                    url=item['image_url']['url']
                    if not url.startswith('data:image/') or ';base64,' not in url:raise ValueError('Only inline image payloads are supported')
                    images.append(url.split(';base64,',1)[1]);texts.append('[Image attached in chronological order]')
        messages.append(dict(role=message.get('role'),content='\n'.join(texts),tool_calls=message.get('tool_calls'),tool_call_id=message.get('tool_call_id')))
    if len(images)>16:raise ValueError('Image context exceeds the private inference endpoint limit of sixteen images')
    prompt=('You are the OpenCode coding agent. Choose tools to complete the user task from the conversation. '
        'Return ONLY JSON {"content":"brief operational decision summary","tool_calls":[{"name":"one of the supplied tool names","arguments":{}}]}. '
        'Use no tool_calls when the task is finished. Arguments must match the supplied tool JSON schema. '
        'This adapter does not execute tools; OpenCode will execute your exact calls and provide feedback. '
        'Do not emit detections unless the user is asking for a detection report.\nTOOLS: '+json.dumps(tools)+'\nCONVERSATION: '+json.dumps(messages))
    if len(prompt)>128000:raise ValueError('Context exceeds the private inference endpoint limit')
    payload=dict(prompt=prompt,images=images,max_output_tokens=min(4096,max(64,request.get('max_tokens') or request.get('max_completion_tokens') or 4096)))
    began=time.time();identifier=uuid.uuid4().hex
    # Requests contain public task context, never provider authentication headers.
    (OUTPUT/(identifier+'-request.json')).write_text(json.dumps(dict(model=MODEL,messages=messages,tools=tools,image_count=len(images))))
    req=urllib.request.Request(INFERENCE_URL,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with INFERENCE_LOCK:
        with urllib.request.urlopen(req,timeout=180) as response:raw=json.load(response)
    raw['requested_max_tokens']=payload['max_output_tokens']
    (OUTPUT/(identifier+'-response.json')).write_text(json.dumps(raw))
    result=raw.get('output')
    if not isinstance(result,dict):return dict(role='assistant',content=raw.get('text','')),raw
    calls=[];names={t['function']['name'] for t in tools if t.get('type')=='function'}
    for item in result.get('tool_calls',[]):
        # Models sometimes echo the standard OpenAI envelope instead of our
        # shorter JSON spelling. Both carry the same exact chosen tool/args.
        function=item.get('function',item) if isinstance(item,dict) else None
        arguments=function.get('arguments') if isinstance(function,dict) else None
        if isinstance(arguments,str):
            try:arguments=json.loads(arguments)
            except ValueError:pass
        if not isinstance(function,dict) or function.get('name') not in names or not isinstance(arguments,dict):raise ValueError('Model emitted an invalid tool call')
        calls.append(dict(id='call_'+uuid.uuid4().hex[:20],type='function',function=dict(name=function['name'],arguments=json.dumps(arguments))))
    message=dict(role='assistant',content=str(result.get('content','')))
    if calls:message['tool_calls']=calls
    return message,raw

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send_json(self,value,status=200):
        data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path.rstrip('/')=='/v1/models':return self.send_json(dict(object='list',data=[dict(id=MODEL,object='model',owned_by='aerorozum')]))
        self.send_json(dict(error='Unknown route'),404)
    def do_POST(self):
        try:
            if self.path!='/v1/chat/completions':return self.send_json(dict(error='Unknown route'),404)
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<12000000:raise ValueError('Invalid payload size')
            request=json.loads(self.rfile.read(size));message,raw=infer(request,self.client_address[0])
            finish=raw.get('finish_reason') or ('tool_calls' if message.get('tool_calls') else 'length' if raw.get('output_tokens',0)>=raw.get('requested_max_tokens',4096) else 'stop');identifier='chatcmpl-'+uuid.uuid4().hex
            usage=dict(prompt_tokens=raw.get('input_tokens',0),completion_tokens=raw.get('output_tokens',0),total_tokens=raw.get('input_tokens',0)+raw.get('output_tokens',0))
            result=dict(id=identifier,object='chat.completion',created=int(time.time()),model=MODEL,choices=[dict(index=0,message=message,finish_reason=finish)],usage=usage)
            if not request.get('stream'):return self.send_json(result)
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.end_headers()
            delta={**message}
            if delta.get('tool_calls'):delta['tool_calls']=[dict(index=i,**call) for i,call in enumerate(delta['tool_calls'])]
            for item in [dict(index=0,delta=delta,finish_reason=None),dict(index=0,delta={},finish_reason=finish)]:
                chunk=dict(id=identifier,object='chat.completion.chunk',created=result['created'],model=MODEL,choices=[item])
                if item['finish_reason'] is not None:chunk['usage']=usage
                self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode());self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
        except UpstreamError as error:self.send_json(dict(error=error.error),error.status)
        except Exception as error:self.send_json(dict(error=dict(message=str(error),type='model_protocol_error')),400)

if __name__=='__main__':ThreadingHTTPServer((os.environ.get('AEROROZUM_PROXY_HOST','127.0.0.1'),int(os.environ.get('AEROROZUM_PROXY_PORT','18941'))),Handler).serve_forever()
