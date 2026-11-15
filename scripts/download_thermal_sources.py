import base64,concurrent.futures,hashlib,json,os,subprocess,time,urllib.request,urllib.parse,urllib.error
from aerorozum.common import data_root,save,utcnow
root=data_root();inventory=json.loads((root/'source-videos/bucket-inventory.json').read_text());bucket=inventory['bucket']
sdk='/home/yehormsi/.local/google-cloud-cli-install/google-cloud-sdk/bin/gcloud'
token=subprocess.check_output([sdk,'auth','print-access-token','--quiet'],text=True).strip()
def get(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'Bearer '+token}),timeout=60)
def fetch(row):
 name=row['name'];dest=root/'source-bucket'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and str(dest.stat().st_size)==str(row['size']):return name,'cached'
 url='https://storage.googleapis.com/storage/v1/b/'+bucket+'/o/'+urllib.parse.quote(name,safe='')+'?alt=media&generation='+row['generation']
 for attempt in range(4):
  try:
   h=hashlib.md5();temp=dest.with_suffix(dest.suffix+'.download')
   with get(url) as response,temp.open('wb') as out:
    while True:
     block=response.read(1024*1024)
     if not block:break
     out.write(block);h.update(block)
   if row.get('md5Hash') and base64.b64encode(h.digest()).decode()!=row['md5Hash']:raise ValueError('GCS MD5 mismatch')
   if str(temp.stat().st_size)!=str(row['size']):raise ValueError('GCS size mismatch')
   os.replace(temp,dest);return name,'downloaded'
  except Exception:
   if attempt==3:raise
   time.sleep(2**attempt)
rows=[r for group in inventory['sessions'].values() for r in group]
results=[]
with concurrent.futures.ThreadPoolExecutor(12) as pool:
 for i,result in enumerate(pool.map(fetch,rows),1):
  results.append(result)
  if i%10==0:print('Downloaded/verified',i,'/',len(rows),'segments',flush=True)
videos=[];errors=[]
for prefix,objects in inventory['sessions'].items():
 name=prefix+'/thermal-timestamps.jsonl';url='https://storage.googleapis.com/storage/v1/b/'+bucket+'/o/'+urllib.parse.quote(name,safe='')
 try:
  meta=json.load(get(url));fetch(meta);timestamps=str(root/'source-bucket'/name)
 except Exception as e:timestamps=None;errors.append({'session':prefix,'timestamps_available':False,'error':str(e)[:150]})
 for row in objects:
  item={'session_id':prefix.rsplit('/',1)[-1],'path':str(root/'source-bucket'/row['name']),'bucket':bucket,'object':row['name'],'generation':row['generation'],'md5Hash':row.get('md5Hash'),'segment':int(row['name'].rsplit('-',1)[-1].split('.')[0])}
  if timestamps:item['timestamps_path']=timestamps
  videos.append(item)
save(root/'source-videos/labelling-sources.json',{'created_at':utcnow(),'videos':videos,'timestamp_gaps':errors})
print('Ready:',len(videos),'native segments from',len(inventory['sessions']),'sessions',flush=True)
