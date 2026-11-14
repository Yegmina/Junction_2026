#!/usr/bin/python3
"""Public client; flight, image provenance and report state live in the sidecar."""
import argparse,json,os,urllib.request,urllib.error
from pathlib import Path
p=argparse.ArgumentParser(description='Native Gazebo investigation tools. Coordinates are local east/north metres; altitude is above flat ground.')
p.add_argument('command',choices=['status','launch','goto','observe','wait','land','submit','result'])
p.add_argument('--x',type=float);p.add_argument('--y',type=float);p.add_argument('--altitude',type=float,default=5)
p.add_argument('--seconds',type=float,default=2);p.add_argument('--report',help='JSON file: {"candidates":[{"evidence":[{"observation_id":"...","bbox":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000}}],"notes":"..."}]}')
a=p.parse_args();body={k:v for k,v in vars(a).items() if v is not None}
if a.report:body['report']=json.loads(Path(a.report).read_text())
base=os.environ.get('DRONE_API','http://simulator:8080')
try:
 req=urllib.request.Request(base+'/tool',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 result=json.load(urllib.request.urlopen(req,timeout=240))
 for item in result.get('images',[]):
  destination=Path('/workspace/evidence')/item['filename'];destination.parent.mkdir(parents=True,exist_ok=True)
  destination.write_bytes(urllib.request.urlopen(base+item['url'],timeout=30).read());item['local_file']=str(destination)
 print(json.dumps(result,indent=2))
except urllib.error.HTTPError as error:
 print(error.read().decode());raise SystemExit(1)
