"""Read-only live research dashboard. No inference calls or experiment controls."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from score_tolerant import score as tolerant_score
from urllib.parse import urlsplit


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if fallback is None else fallback


def tail(path, size=18000):
    try:
        with Path(path).open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell()-size))
            return f.read().decode(errors='replace')
    except OSError:
        return ''


def clamp(value, lower=0, upper=100):
    return max(lower, min(upper, value))


class Results:
    def __init__(self):
        self.folder = None
        self.records = {}
        self.seen = set()

    def scan(self, folder):
        if folder != self.folder:
            self.folder, self.records, self.seen = folder, {}, set()
        for p in (folder/'results').glob('*.json'):
            if p.name in self.seen:
                continue
            r = read_json(p)
            if not r.get('id'):
                continue
            # Keep compact metrics and drawing boxes, never provider prompts/raw logs.
            self.records[p.name] = {k:r.get(k) for k in ('id','sha256','prompt_version','thinking_level','replica','score','status','valid_output','reused_paid_completion','usage','at','predictions')}
            self.seen.add(p.name)
        return list(self.records.values())


class Monitor:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.snapshot = {'ready':False,'updated_at':None}
        self.images = {}
        self.history = defaultdict(lambda:deque(maxlen=90))
        self.rates = defaultdict(lambda:deque(maxlen=50))
        self.results = Results()
        self.cpu_previous = None
        self.stop = threading.Event()
        self.errors = []
        self.unit_states = {}

    def rate(self, key, count, now):
        q = self.rates[key]
        if q and count < q[-1][1]:
            q.clear()
        q.append((now,count))
        if len(q)<2 or now-q[0][0]<15:
            return None
        # History is a rolling sample; startup-only estimates are withheld.
        return max(0,(count-q[0][1])/(now-q[0][0]))

    def register_image(self, path, boxes=None, gold=None):
        path = Path(path).resolve()
        allowed = [self.root/'artifacts', self.root/'private-api-review', self.root/'private-model-comparison']
        if not any(path.is_relative_to(p) for p in allowed) or not path.is_file() or path.suffix.lower() not in ('.jpg','.jpeg','.png'):
            return None
        spec = dict(path=str(path),boxes=boxes or [],gold=gold or [])
        key = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:24]
        with self.lock:
            self.images[key] = spec
            while len(self.images)>512:
                self.images.pop(next(iter(self.images)))
        return 'image/'+key+'.jpg'

    def refresh_units(self):
        units=[]
        receipts=sorted((self.root/'private-api-review').glob('gpu-resume-*.json'))
        units += [r['unit'] for r in read_json(receipts[-1],[]) if r.get('unit')] if receipts else []
        for folder in ['private-api-review','private-prompt-study','private-model-comparison','private-height-prompt-study']:
            row=read_json(self.root/folder/'current.json')
            if row.get('unit'):units.append(row['unit'])
        if not units:return
        try:
            p=subprocess.run(['systemctl','show','--property=Id,ActiveState','--no-pager',*units],capture_output=True,text=True,timeout=4)
            for block in p.stdout.strip().split('\n\n'):
                values=dict(line.split('=',1) for line in block.splitlines() if '=' in line)
                if values.get('Id'):self.unit_states[values['Id'].removesuffix('.service')]=values.get('ActiveState')
        except (OSError,subprocess.SubprocessError):pass

    def worker_status(self, status, unit):
        active=self.unit_states.get(unit.removesuffix('.service'))
        return 'stopped' if status=='running' and active in ('inactive','failed') else status

    def infrastructure(self, now):
        hardware = {'gpus':[]}
        try:
            proc = subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
            for line in proc.stdout.splitlines():
                values=[s.strip() for s in line.split(',')]
                if len(values)!=7:
                    continue
                idx,name,util,used,total,temp,power=values
                number=lambda s:float(s) if re.fullmatch(r'\d+(\.\d+)?',s) else None
                util=number(util)
                if util is not None:self.history['gpu'+idx].append(util)
                hardware['gpus'].append(dict(index=int(idx),name=name,utilization=util,memory_used_mib=number(used),memory_total_mib=number(total),temperature=number(temp),power_w=number(power),history=list(self.history['gpu'+idx]),role='Live demo inference' if idx=='0' else 'Gemma 12B · seed '+{'1':'17','2':'43','3':'91'}.get(idx,idx)))
        except (OSError,subprocess.SubprocessError):
            hardware['gpu_error']='GPU metrics temporarily unavailable'
        values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
        total=sum(values[:8]);idle=values[3]+values[4]
        previous=self.cpu_previous;self.cpu_previous=(total,idle)
        cpu=None if previous is None or total==previous[0] else clamp(100*(1-(idle-previous[1])/(total-previous[0])))
        cores=sum(1 for l in Path('/proc/stat').read_text().splitlines() if re.match(r'cpu\d+ ',l))
        mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
        disk=shutil.disk_usage(self.root)
        hardware.update(cpu_percent=cpu,cpu_cores=cores,cpu_cores_used=None if cpu is None else cpu*cores/100,memory_total_bytes=mem['MemTotal'],memory_used_bytes=mem['MemTotal']-mem['MemAvailable'],disk_total_bytes=disk.total,disk_free_bytes=disk.free)
        return hardware

    def training(self, now):
        receipts=sorted((self.root/'private-api-review').glob('gpu-resume-*.json'))
        jobs=[]
        for row in read_json(receipts[-1],[]) if receipts else []:
            seed=row['seed'];folder=self.root/f'artifacts/postsubmission-high-teacher-visual-v1/seed-{seed}'
            spec=read_json(folder/'specification.json');status=read_json(folder/'status.json');phase=read_json(folder/'initial/status.json')
            base=self.root/f'artifacts/training/{spec.get("namespace","")}/initial/google_gemma-4-12B-it'
            sft=read_json(base/f'stream_sft/seed-{seed}/gradient_checks.json');grpo=read_json(base/f'stream_grpo/seed-{seed}/gradient_checks.json')
            result=read_json(folder/'initial/result.json');step=int(sft.get('checks',0));target=spec.get('sft_steps',3751)
            grpo_step=int(grpo.get('checks',0));stage=phase.get('stage') or status.get('stage') or 'starting'
            log=tail(row['log']);losses=re.findall(r"'loss':\s*'?([-+\deE.]+)",log);norms=re.findall(r"'grad_norm':\s*'?([-+\deE.]+)",log)
            loss=float(losses[-1]) if losses else None
            if loss is not None and math.isfinite(loss):self.history['loss'+str(seed)].append(loss)
            speed=self.rate('sft'+str(seed),step,now)
            events=[]
            for match in re.finditer(r"\{'loss':[^\n]+",log):
                a=match.group(0);l=re.search(r"'loss':\s*'?([-+\deE.]+)",a);g=re.search(r"'grad_norm':\s*'?([-+\deE.]+)",a)
                if l:events.append(f'Seed {seed} · loss {float(l.group(1)):.5f}'+(f' · gradient {float(g.group(1)):.3f}' if g else ''))
            jobs.append(dict(seed=seed,gpu=row['gpu'],status=self.worker_status(status.get('status','unknown'),row['unit']),stage=stage,sft_step=min(step,target),sft_target=target,grpo_step=grpo_step,grpo_target=spec.get('grpo_steps',48),finite_gradients=sft.get('finite'),last_loss=loss,gradient_norm=float(norms[-1]) if norms else None,loss_history=list(self.history['loss'+str(seed)]),sft_eta_seconds=(target-step)/speed if speed and step<target else None,updates_per_minute=speed*60 if speed else None,events=events[-3:],result_available=bool(result)))
        return jobs

    def api_review(self, now):
        current=read_json(self.root/'private-api-review/current.json')
        if not current:return {}
        work=Path(current['work_dir']);out=work/'output';protocol=read_json(work/'protocol.json');progress=read_json(out/'progress.json');settled=read_json(out/'settled.json');recent=read_json(out/'recent.json')
        complete=progress.get('completed',0);failed=progress.get('failed',0);target=protocol.get('batch_frames',0);speed=self.rate('review'+work.name,complete+failed,now)
        images=[]
        for row in recent.get('images',[])[:6]:
            picture=row.get('annotated') or row.get('image')
            url=self.register_image(picture) if picture else None
            if url:images.append(dict(url=url,kind='Model annotations',confidence=max([b.get('confidence',0) for b in row.get('boxes',[])+row.get('uncertain_boxes',[])] or [0]),accepted=len(row.get('boxes',[])),uncertain=len(row.get('uncertain_boxes',[])),caption='V3 · HIGH · airborne thermal'))
        return dict(status='completed' if settled else self.worker_status(progress.get('status','starting'),current.get('unit','')),completed=complete,failed=failed,total=target,uploaded=progress.get('cloud_sync',{}).get('uploaded',0),upload_pending=progress.get('cloud_sync',{}).get('upload_pending',0),per_minute=speed*60 if speed else None,eta_seconds=max(0,target-complete-failed)/speed if speed and not settled else None,model=protocol.get('model'),thinking=protocol.get('thinking_level'),images=images)

    def study(self, now):
        current=read_json(self.root/'private-prompt-study/current.json')
        if not current:return {}
        work=Path(current['work_dir']);out=work/'output';protocol=read_json(out/'protocol.json');progress=read_json(out/'progress.json');records=self.results.scan(out)
        imgs=protocol.get('images',[]);levels=protocol.get('thinking_levels',[]);prompts=list(protocol.get('prompts',{}));replicas=protocol.get('replicas',0)
        grouped=defaultdict(list)
        catalog={row['id']:row for row in imgs}
        for r in records:
            gold=catalog.get(r['id'],{}).get('boxes',[])
            r['score_tolerant']=tolerant_score(r.get('predictions') or [],gold,edge_tolerance=.75)
            grouped[(r['prompt_version'],r['thinking_level'])].append(r)
        cells=[]
        for prompt in prompts:
            for level in levels:
                group=grouped[(prompt,level)];scored=[r for r in group if r.get('score_tolerant')];tp=sum(r['score_tolerant']['tp'] for r in scored);fp=sum(r['score_tolerant']['fp'] for r in scored);fn=sum(r['score_tolerant']['fn'] for r in scored)
                cells.append(dict(prompt=prompt,level=level,completed=len(group),total=len(imgs)*replicas,tp=tp,fp=fp,fn=fn,f2=5*tp/(5*tp+4*fn+fp) if scored and 5*tp+4*fn+fp else (0 if scored else None),precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,invalid=sum(r.get('status')=='invalid_paid_response' for r in group),transport_errors=sum(r.get('status')=='transport_error' for r in group)))
        cached=read_json(out/'paid-cache-reuse.json');receipt=read_json(out/'throughput-resume-receipt.json');count=max(len(records),int(progress.get('completed',0)));total=len(imgs)*len(levels)*len(prompts)*replicas
        issued=int(receipt.get('issued',0))+int(progress.get('issued',0));speed=self.rate('study'+work.name,issued,now)
        reuse=cached.get('reused_count',0);expected_new=max(0,total-reuse);throughput=read_json(out/'throughput-addendum.json')
        examples=[];catalog={r['id']:r for r in imgs}
        candidates=sorted([r for r in records if r.get('predictions') and r.get('score')],key=lambda r:r.get('at') or '',reverse=True)
        seen=set()
        for row in candidates:
            if row['id'] in seen:continue
            source=catalog.get(row['id'],{});url=self.register_image(self.root/'artifacts'/source['image'],row.get('predictions'),[b.get('bbox',b) for b in source.get('boxes',[])]) if source.get('image') else None
            if url:
                score75=tolerant_score(row.get('predictions') or [],source.get('boxes',[]),edge_tolerance=.75)
                examples.append(dict(url=url,kind='Aki-validated reference',caption=short_prompt(row['prompt_version'])+' · '+row['thinking_level'].upper(),f2=score75['f2'],tp=score75['tp'],fp=score75['fp'],fn=score75['fn']));seen.add(row['id'])
            if len(examples)==3:break
        thoughts=sum((r.get('usage') or {}).get('thoughts_token_count',0) or 0 for r in records if not r.get('reused_paid_completion'))
        report=read_json(out/'report.json');done=report.get('status')=='completed'
        addendum=read_json(out/'scoring-addendum.json')
        return dict(status='completed' if done else self.worker_status(progress.get('status','starting'),current.get('unit','')),completed=count,total=total,images=len(imgs),replicas=replicas,levels=levels,prompts=prompts,cells=cells,cache_reused=reuse,new_requests=issued,expected_new_requests=expected_new,new_requests_per_minute=speed*60 if speed else None,configured_rpm=throughput.get('study_rpm',60),combined_configured_rpm=throughput.get('combined_configured_rpm'),concurrency=throughput.get('concurrency',8),eta_seconds=max(0,expected_new-issued)/speed if speed and not done else None,thinking_tokens_new=thoughts,archive_failures=progress.get('archive_failures',0),examples=examples,annotation_origins=protocol.get('annotation_origin_counts',{}),scoring_rule=addendum.get('rule','IoU ≥0.5 (default)'),errors=len(progress.get('errors',[])))

    def model_comparison(self, now):
        current=read_json(self.root/'private-model-comparison/current.json')
        if not current:return {}
        work=Path(current['work_dir']);out=work/'output';progress=read_json(out/'progress.json');report=read_json(out/'report.json');throughput=read_json(out/'throughput.json');quota=read_json(out/'quota-events.json',{'events':[]})
        completed=int(progress.get('completed',report.get('completed',0)));total=int(progress.get('total',report.get('total',0)));issued=int(progress.get('issued',0));speed=self.rate('model-comparison'+work.name,issued,now)
        rates=dict(progress.get('rate_by_model',{}))
        for model in throughput.get('models',[]):rates.setdefault(model,throughput.get('per_model_initial_rpm',0))
        for event in quota.get('events',[]):
            if event.get('model') in rates:rates[event['model']]=event.get('rpm',event.get('rpm_after',rates[event['model']]))
        protocol=read_json(out/'protocol.json');catalog={row['id']:row for row in protocol.get('images',[])};examples=[];seen=set()
        paths=sorted((out/'results').glob('*.json'),key=lambda path:path.stat().st_mtime,reverse=True)
        for path in paths:
            row=read_json(path);identity=(row.get('model'),row.get('prompt_version'),row.get('thinking_level'))
            if identity in seen or row.get('status')!='completed':continue
            source=catalog.get(row.get('id'))
            if source is None:continue
            image=(self.root/'artifacts'/source['image']).resolve()
            url=self.register_image(image,boxes=row.get('predictions') or [],gold=source.get('boxes') or [])
            if not url:continue
            score=tolerant_score(row.get('predictions') or [],source.get('boxes') or [],edge_tolerance=.75);examples.append(dict(url=url,kind=row.get('model','Gemini 4'),caption=f"{row.get('prompt_version','')} · {str(row.get('thinking_level','')).upper()}",f2=score['f2'],tp=score['tp'],fp=score['fp'],fn=score['fn']));seen.add(identity)
            if len(examples)>=6:break
        rescored=read_json(out/'tolerant-comparison-75.json')
        return dict(status=self.worker_status(progress.get('status','starting'),current.get('unit','')),completed=completed,total=total,issued=issued,issued_by_model=progress.get('issued_by_model',report.get('issued_by_model',{})),rate_by_model=rates,rpm_ceiling=throughput.get('per_model_rpm_ceiling'),per_model_tpm=throughput.get('per_model_tpm'),concurrency=throughput.get('per_model_concurrency'),models=report.get('models',[]),arms=rescored.get('arms',report.get('arms',[])),examples=examples,scoring_rule=rescored.get('scoring_rule','aki-box-tolerant-v4-75pct'),errors=progress.get('failed',0),eta_seconds=max(0,total-completed)/speed if speed and progress.get('status')=='running' else None,archive=read_json(out/'archive-state.json'))

    def height_study(self, now):
        current=read_json(self.root/'private-height-prompt-study/current.json')
        if not current:return {}
        work=Path(current['work_dir']);out=work/'output';progress=read_json(out/'progress.json');protocol=read_json(out/'protocol.json');rescored=read_json(out/'tolerant-comparison-75.json')
        completed=int(progress.get('completed',0));total=int(progress.get('total',0));issued=int(progress.get('issued',0));speed=self.rate('height-study'+work.name,issued,now)
        baseline_path=Path(protocol.get('comparison_baseline','')).resolve();allowed=(self.root/'private-prompt-study').resolve()
        baseline=read_json(baseline_path/'tolerant-comparison-75.json') if baseline_path.is_relative_to(allowed) else {}
        base={(row.get('prompt'),row.get('thinking_level')):row for row in baseline.get('arms',[])}
        arms=[]
        for row in rescored.get('arms',[]):
            old=base.get((row.get('prompt'),row.get('thinking_level')),{})
            arms.append(dict(**row,without_height_f2=old.get('f2'),with_height_f2=row.get('f2'),delta_f2=None if old.get('f2') is None else row.get('f2',0)-old.get('f2',0)))
        catalog={row['id']:row for row in protocol.get('images',[])};examples=[];seen=set()
        paths=sorted((out/'results').glob('*.json'),key=lambda path:path.stat().st_mtime,reverse=True)
        for path in paths:
            row=read_json(path);identity=(row.get('prompt_version'),row.get('thinking_level'))
            if identity in seen or row.get('status')!='completed':continue
            source=catalog.get(row.get('id'))
            if source is None:continue
            image=(self.root/'artifacts'/source['image']).resolve();url=self.register_image(image,boxes=row.get('predictions') or [],gold=source.get('boxes') or [])
            if not url:continue
            score=tolerant_score(row.get('predictions') or [],source.get('boxes') or [],edge_tolerance=.75)
            caption=f"{row.get('prompt_version','')} · {str(row.get('thinking_level','')).upper()} · {source.get('capture_relative_altitude_m',0):.1f} m rel."
            examples.append(dict(url=url,kind='Gemini Robotics ER-2 · height included',caption=caption,f2=score['f2'],tp=score['tp'],fp=score['fp'],fn=score['fn']));seen.add(identity)
            if len(examples)>=6:break
        return dict(status=self.worker_status(progress.get('status','starting'),current.get('unit','')),completed=completed,total=total,issued=issued,images=len(protocol.get('images',[])),thinking_levels=protocol.get('thinking_levels',[]),prompts=list(protocol.get('prompts',{})),rpm=protocol.get('limits',{}).get('requests_per_minute',60),eta_seconds=max(0,total-completed)/speed if speed and progress.get('status')=='running' else None,arms=arms,examples=examples,errors=progress.get('archive_failures',0),scoring_rule=rescored.get('scoring_rule','aki-box-tolerant-v4-75pct'))

    def sample(self):
        self.refresh_units()
        now=time.time();errors=[];data={'ready':True,'updated_at':datetime.now(timezone.utc).isoformat(),'refresh_seconds':8}
        for key,method in [('hardware',self.infrastructure),('training',self.training),('review',self.api_review),('study',self.study),('model_comparison',self.model_comparison),('height_study',self.height_study)]:
            try:data[key]=method(now)
            except Exception:
                # The public API never exposes exception text, paths or secrets.
                errors.append(key)
                with self.lock:data[key]=self.snapshot.get(key,[] if key=='training' else {})
        data['unavailable_sections']=errors
        with self.lock:self.snapshot=data

    def run(self):
        while not self.stop.is_set():
            try:self.sample()
            except Exception:pass
            self.stop.wait(8)

    def render_image(self, key):
        with self.lock:spec=self.images.get(key)
        if spec is None:return None
        from PIL import Image, ImageDraw
        with Image.open(spec['path']) as original:
            image=original.convert('RGB');image.thumbnail((1008,768));draw=ImageDraw.Draw(image)
            for boxes,color in [(spec['gold'],'#48d8ff'),(spec['boxes'],'#b4ff4d')]:
                for box in boxes:
                    box=box.get('bbox',box);x1,y1,x2,y2=[float(box[n]) for n in ('x_min','y_min','x_max','y_max')]
                    draw.rectangle((x1*image.width/1000,y1*image.height/1000,x2*image.width/1000,y2*image.height/1000),outline=color,width=2)
            buffer=io.BytesIO();image.save(buffer,'JPEG',quality=88);return buffer.getvalue()


def short_prompt(value):
    return {'thermal-v1-conservative':'Conservative','thermal-v2-recall-calibrated':'Recall focused','thermal-v3-two-scale-review':'Two scale'}.get(value,value)


def handler(monitor, webroot):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path in ('/','/index.html'):
                return self.reply(200,(webroot/'index.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/api/status':
                with monitor.lock:payload=json.dumps(monitor.snapshot,allow_nan=False).encode()
                return self.reply(200,payload,'application/json')
            if path=='/healthz':
                return self.reply(200,b'{"ok":true}','application/json')
            match=re.fullmatch(r'/image/([0-9a-f]{24})\.jpg',path)
            if match:
                try:payload=monitor.render_image(match.group(1))
                except (OSError,ValueError):payload=None
                if payload is not None:return self.reply(200,payload,'image/jpeg')
            self.reply(404,b'Not found','text/plain')

        def reply(self, status, payload, content_type):
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'; base-uri 'self'");self.end_headers()
            try:self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError):pass

        def log_message(self, *_):pass
    return Handler


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/opt/aerorozum');parser.add_argument('--port',type=int,default=18960);args=parser.parse_args()
    monitor=Monitor(args.root);threading.Thread(target=monitor.run,daemon=True).start()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(monitor,Path(__file__).parent));server.daemon_threads=True
    try:server.serve_forever()
    finally:monitor.stop.set()
