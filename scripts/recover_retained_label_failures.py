"""Read-only CPython3.12 probe. Only known source IDs and allowlisted errors escape; no memory dump, pause or modification."""
import json,os,struct,sys
from pathlib import Path
import numpy as np
pid=int(sys.argv[1]) if len(sys.argv)>1 else 36477;r=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/artifacts/labelling/robotics-er2-5fps');known={x['id'] for x in json.loads((r/'sampling-manifest.json').read_text())['images']};error=b'GCS cache read failed: HTTP 401; paid inference was not issued';regions=[]
for sample in (r/'frames').glob('*/sampling.json'):
 known.update(row['id'] for row in json.loads(sample.read_text()).get('images',[]))
for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
 f=line.split();name=f[-1] if len(f)>5 else ''
 if f[1].startswith('rw') and (not name or name.startswith('[')):regions.append(tuple(int(x,16) for x in f[0].split('-')))
fd=os.open(f'/proc/{pid}/mem',os.O_RDONLY)
def chunks():
 for a,b in regions:
  for p in range(a,b,4194304):
   try:yield p,os.pread(fd,min(4194304,b-p),p)
   except OSError:continue
ptrs=[]
for p,buf in chunks():
 start=0
 while True:
  i=buf.find(error,start)
  if i<0:break
  candidate=p+i-40
  try:
   header=os.pread(fd,40,candidate)
   if len(header)==40 and struct.unpack_from('<Q',header,16)[0]==len(error):ptrs.append(candidate)
  except OSError:pass
  start=i+len(error)
ptrs=np.array(sorted(set(ptrs)),dtype=np.uint64)
if len(ptrs)<100:raise RuntimeError('Unexpected Unicode layout; process remains unchanged')
def text(p,n):
 try:
  b=os.pread(fd,40+n,p)
  if len(b)==40+n and struct.unpack_from('<Q',b,16)[0]==n:return b[40:].decode('ascii')
 except (OSError,UnicodeError,OverflowError):pass
 return None
found=set();error_keys=set()
for p,buf in chunks():
 arr=np.frombuffer(buf[:len(buf)//8*8],dtype='<u8');idx=np.flatnonzero((arr>=ptrs[0])&(arr<=ptrs[-1]))
 if not len(idx):continue
 pos=np.searchsorted(ptrs,arr[idx]);mask=pos<len(ptrs);idx=idx[mask];pos=pos[mask];idx=idx[ptrs[pos]==arr[idx]]
 for i in idx:
  offset=int(i)*8
  if offset<24:continue
  a,b,c=struct.unpack_from('<QQQ',buf,offset-24)
  if text(a,2)=='id' and text(c,5)=='error':
   identity=text(b,64)
   if identity in known:found.add(identity);error_keys.add(c)
records={}
for p,buf in chunks():
 arr=np.frombuffer(buf[:len(buf)//8*8],dtype='<u8')
 for key in error_keys:
  for i in np.flatnonzero(arr==key):
   offset=int(i)*8
   if offset<16 or offset+16>len(buf):continue
   a,b,c,e=struct.unpack_from('<QQQQ',buf,offset-16)
   if text(a,2)!='id':continue
   identity=text(b,64)
   if identity not in known:continue
   try:
    header=os.pread(fd,40,e);length=struct.unpack_from('<Q',header,16)[0]
    message=text(e,length) if 0<length<=2048 else None
   except (OSError,OverflowError,struct.error):message=None
   records[identity]=message or 'Retained error text unavailable in supported ASCII layout'
os.close(fd);out=r/'diagnosis'/('retained-failures-'+str(pid)+'.json');out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({'pid':pid,'method':'read_only_retained_failure_dicts_allowlisted_source_IDs','records':[{'id':i,'error':e} for i,e in sorted(records.items())]},indent=2)+'\n')
print(json.dumps({'recovered_frame_failures':len(records),'output':str(out),'process_untouched':True}))
