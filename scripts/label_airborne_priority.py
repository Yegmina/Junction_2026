import asyncio,json
from aerorozum.common import data_root,save,redacted_error
from aerorozum.labelling import label_video_sources
from aerorozum.jobs import job,update
r=data_root();d=json.loads((r/'source-videos/labelling-sources.json').read_text())
# Prioritize airborne portions while the complete 5 Hz job covers every segment.
rows=[x for x in d['videos'] if x['session_id']=='2026-11-11_134911' and x['segment'] in (4,5,6)]
rows.sort(key=lambda x:x['segment']);p=r/'source-videos/airborne-priority.json';save(p,{'videos':rows})
j=job('robotics_er2_airborne_priority',total_sources=len(rows),model='gemini-robotics-er-2-preview',sample_fps=5)
try:
 result=asyncio.run(label_video_sources(p,r/'labelling/robotics-er2-airborne-priority',fps=5,threshold=.6,concurrency=8))
 update(j,result['status'],result=result);print(json.dumps(result,indent=2))
except Exception as e:update(j,'failed',error=redacted_error(e));raise
