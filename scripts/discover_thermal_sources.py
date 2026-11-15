"""Generation-pinned raw thermal source discovery; never issues model calls."""
import base64,concurrent.futures,hashlib,json,re,urllib.request,urllib.parse,urllib.error
from pathlib import Path
from aerorozum.common import data_root,save,digest,utcnow
from aerorozum.cloud_cache import GCSRecordStore
from aerorozum.labelling import _probe,sample_indices,freeze_session_splits
r=data_root();bucket='aerorozum-jetson-testdata-smart-bird';auth=GCSRecordStore(bucket);out=r/'labelling/additional-bucket-sources';out.mkdir(parents=True,exist_ok=True)
def get(url):
 for attempt in range(2):
  headers=auth._headers()
  try:return urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90)
  except urllib.error.HTTPError as e:
   if e.code==401 and attempt==0:auth._invalidate(headers);continue
   raise
url='https://storage.googleapis.com/storage/v1/b/'+bucket+'/o?prefix=jetson%2F&maxResults=1000';objects=[]
while url:
 with get(url) as response:d=json.load(response)
 objects.extend(d.get('items',[]));page=d.get('nextPageToken');url=url.split('&pageToken=')[0]+('&pageToken='+urllib.parse.quote(page,safe='')) if page else None
original=r/'source-videos/labelling-sources-original-150.json';old=json.loads((original if original.exists() else r/'source-videos/labelling-sources.json').read_text())['videos'];known={(x['object'],str(x['generation'])) for x in old};normal=[x for x in objects if re.search(r'/thermal-\d+\.avi$',x['name'])];extra=[x for x in normal if (x['name'],str(x['generation'])) not in known];recovered=[x for x in objects if re.search(r'/recovered/thermal-\d+-recovered\.avi$',x['name'])];selected=extra+recovered
save(out/'generation-pinned-inventory.json',{'at':utcnow(),'bucket':bucket,'normal_count':len(normal),'original_count':len(old),'new_normal_objects':extra,'recovered_objects':recovered,'annotated_derivatives_excluded':[x['name'] for x in objects if x['name'].endswith('annotated_thermal.mp4')],'no_model_calls':True})
def download(meta):
 folder=out/'media'/digest([bucket,meta['name'],meta['generation']]);folder.mkdir(parents=True,exist_ok=True);path=folder/Path(meta['name']).name
 if not path.exists() or path.stat().st_size!=int(meta['size']):
  url='https://storage.googleapis.com/storage/v1/b/'+bucket+'/o/'+urllib.parse.quote(meta['name'],safe='')+'?alt=media&generation='+str(meta['generation']);temp=path.with_suffix(path.suffix+'.partial')
  with get(url) as response,temp.open('wb') as stream:
   while True:
    block=response.read(1048576)
    if not block:break
    stream.write(block)
  if temp.stat().st_size!=int(meta['size']):raise ValueError('Pinned source size mismatch')
  temp.replace(path)
 raw=path.read_bytes()
 if meta.get('md5Hash') and base64.b64encode(hashlib.md5(raw).digest()).decode()!=meta['md5Hash']:raise ValueError('Pinned source MD5 mismatch')
 record={'bucket':bucket,'object':meta['name'],'generation':str(meta['generation']),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gcs_MD5_verified':bool(meta.get('md5Hash')),'path':str(path)};save(folder/'source.json',record);return record

def inspect(meta):
 src=download(meta);cached=Path(src['path']).parent/'probe.json'
 if cached.exists():
  value=json.loads(cached.read_text())
  if value.get('sha256')==src['sha256']:return value
 sid=re.search(r'/recordings/([^/]+)/',src['object']).group(1);segment=int(re.search(r'thermal-(\d+)',Path(src['object']).name).group(1))
 try:
  w,h,pts=_probe(src['path']);mono=bool(pts) and all(b>a for a,b in zip(pts,pts[1:]));row={**src,'session_id':sid,'segment':segment,'width':w,'height':h,'native_frames':len(pts),'pts_strictly_monotonic':mono,'duration_s':pts[-1]-pts[0] if pts else 0,'video_clock_5fps_frames':len(sample_indices(pts,5)) if mono else None,'recovered_variant':'/recovered/' in src['object']}
 except Exception as error:row={**src,'session_id':sid,'segment':segment,'pts_strictly_monotonic':False,'video_clock_5fps_frames':None,'error':str(error)[:300],'recovered_variant':'/recovered/' in src['object']}
 save(cached,row);return row

with concurrent.futures.ThreadPoolExecutor(4) as pool:
 probes=[]
 for result in pool.map(inspect,selected):
  probes.append(result);save(out/'probe-progress.json',{'at':utcnow(),'completed':len(probes),'total':len(selected),'known_5fps_frames':sum(x['video_clock_5fps_frames'] or 0 for x in probes),'sources':probes});print(json.dumps({'probed':len(probes),'total':len(selected),'frames':result['video_clock_5fps_frames']}),flush=True)
freeze_session_splits([x['session_id'] for x in probes],r/'labelling/robotics-er2-5fps/session-splits.json',17)
save(out/'native-probe.json',{'at':utcnow(),'sources':probes,'video_clock_5fps_frames':sum(x['video_clock_5fps_frames'] or 0 for x in probes),'no_model_calls':True})
