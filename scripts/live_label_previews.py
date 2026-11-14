"""Publish previews from saved paid results; never issue inference or rewrite them."""
import json,time
from pathlib import Path
from aerorozum.common import data_root,save,utcnow
from aerorozum.labelling import _live_preview
r=data_root();folder=r/'labelling/robotics-er2-5fps/labels';seen=set();recent=[]
while True:
 for path in sorted((folder/'results').glob('*.json'),key=lambda p:p.stat().st_mtime):
  if path.name in seen:continue
  try:
   result=json.loads(path.read_text())
   if 'detections' not in result or not result.get('image'):continue
   source=Path(result['image']);source=source if source.is_absolute() else r/source
   _live_preview(result,source,folder,path.stem)
   public={k:result[k] for k in ['id','image','annotated','model','boxes','uncertain_boxes','source','timestamp_s','source_offset_s','frame_index','split','user_threshold'] if k in result}
   public['cloud_sync_status']=result.get('cloud_sync',{}).get('status','unknown')
   recent.append(public);recent=recent[-12:];seen.add(path.name)
  except (OSError,ValueError,KeyError):continue
 if recent:save(folder/'recent.json',{'updated_at':utcnow(),'model':'gemini-robotics-er-2-preview','images':recent})
 time.sleep(5)
