import asyncio,json,threading,time
from aerorozum.common import data_root,save,redacted_error
from aerorozum.jobs import job,update
from aerorozum.labelling import label_video_sources
r=data_root();source=r/'source-videos/labelling-sources.json';d=json.loads(source.read_text())
d['videos'].sort(key=lambda v:(v['session_id'],-v['segment']),reverse=True)
priority=r/'source-videos/labelling-sources-priority.json';save(priority,d)
record=job('robotics_er2_5fps',total_sources=len(d['videos']),completed=0,model='gemini-robotics-er-2-preview',sample_fps=5,user_threshold=.6,cloud_archive='gs://aerorozum-jetson-testdata-smart-bird/robotics-er2-labels/5fps-v1/')
stop=threading.Event()
def monitor():
 while not stop.wait(5):
  p=r/'labelling/robotics-er2-5fps/labels/progress.json'
  if p.exists():
   try:
    state=json.loads(p.read_text());update(record,completed=state.get('completed',state.get('labelled',0)),failed=state.get('failed',0),label_progress=state)
   except (OSError,ValueError):pass
threading.Thread(target=monitor,daemon=True).start()
try:
 result=asyncio.run(label_video_sources(priority,r/'labelling/robotics-er2-5fps',fps=5,threshold=.6,concurrency=16))
 update(record,result['status'],result=result);print(json.dumps(result,indent=2),flush=True)
except Exception as ex:
 update(record,'failed',error=redacted_error(ex));raise
finally:stop.set()
