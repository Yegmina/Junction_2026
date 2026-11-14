"""Public-observation baseline: grid coverage and small thermal residual peaks.

This is a reference policy, not an oracle; it never reads simulator state,
target positions, field arrays, verifier internals, or scoring endpoints.
"""
import json,subprocess,time,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,maximum_filter

def call(command,*args):
    process=subprocess.run(['drone',command,*map(str,args)],capture_output=True,text=True)
    if process.returncode:raise RuntimeError(process.stdout+process.stderr)
    return json.loads(process.stdout)

state=call('launch','--altitude',5)
candidates=[]
for x,y in [(7.5,-2.5),(12.5,-2.5),(12.5,2.5),(7.5,2.5),(-2.5,9.5),(2.5,9.5),(2.5,14.5),(-2.5,14.5)]:
    state=call('status')
    if state['remaining_s']<45:break
    call('goto','--x',x,'--y',y,'--altitude',5)
    for _ in range(25):
        state=call('wait','--seconds',2);p=state['pose']
        if math.hypot(p['x_m']-x,p['y_m']-y)<.5:break
        if state['remaining_s']<40:break
    if state['remaining_s']<40:break
    observation=call('observe');item=next(i for i in observation['images'] if i['kind']=='thermal')
    pixels=np.asarray(Image.open(item['local_file']).convert('L'),dtype=np.float32)
    residual=gaussian_filter(pixels,1.5)-gaussian_filter(pixels,8)
    threshold=max(12,float(np.std(residual))*4)
    peaks=np.argwhere((abs(residual)==maximum_filter(abs(residual),size=31))&(abs(residual)>threshold))
    ranked=sorted(peaks,key=lambda p:float(abs(residual[tuple(p)])),reverse=True)[:2]
    for v,u in ranked:
        if not 15<u<pixels.shape[1]-15 or not 15<v<pixels.shape[0]-15:continue
        bbox=dict(x_min=(u-12)/pixels.shape[1]*1000,y_min=(v-12)/pixels.shape[0]*1000,x_max=(u+12)/pixels.shape[1]*1000,y_max=(v+12)/pixels.shape[0]*1000)
        candidates.append(dict(notes='Compact thermal residual; provisional public-image heuristic',evidence=[dict(observation_id=observation['observation_id'],bbox=bbox)]))
call('land');path=Path('/workspace/report.json');path.write_text(json.dumps(dict(candidates=candidates[:32])))
print(json.dumps(call('submit','--report',path),indent=2))
