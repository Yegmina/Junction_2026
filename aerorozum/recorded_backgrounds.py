"""Display textures quilted from original thermal frames, never physical truth.

Source grayscale intensities are retained. Frames are unlabelled and may
contain targets, people or other objects. World scale is assumed.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import os
import re
import numpy as np
from PIL import Image
SOURCE='recorded_background_composite_unvalidated'
DEFAULT_DIRECTORY=Path('/home/yehormsi/Downloads/gemini_parallel_flight_test/evidence')


def original_frames(directory):
    paths=sorted(p for p in Path(directory).expanduser().resolve().glob('frame_*.jpg') if re.fullmatch(r'frame_\d+',p.stem))
    if not paths:raise ValueError('No original thermal frames found; set recorded_background_dir to raw frame_<digits>.jpg directory')
    return paths


@lru_cache(maxsize=3)
def _source_bank(directory,maximum,screened_manifest="",screened_fingerprint=""):
    paths=original_frames(directory);arrays=[];provenance=[]
    indices=np.linspace(0,len(paths)-1,min(maximum,len(paths))).round().astype(int)
    for index in indices:
        path=paths[int(index)]
        with Image.open(path) as image:
            if image.size!=(640,512):continue
            arrays.append(np.asarray(image.convert('L'),dtype=np.uint8))
        provenance.append({'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':640,'height':512})
    if screened_manifest:
        for path,row in screened_crops(screened_manifest,maximum=max(1,maximum//4)):
            if row.get('split')!='train':continue
            with Image.open(path) as image:arrays.append(np.asarray(image.convert('L'),dtype=np.uint8))
            provenance.append({**row,'filename':path.name,'source_type':row.get('background_source','annotation_excluded_crop'),'screening_provenance':'model_screened_not_human_verified'})
    if not arrays:raise ValueError('Recorded textures require original 640×512 thermal frames')
    return tuple(arrays),tuple(provenance)


@lru_cache(maxsize=3)
def _quilt(directory,seed,resolution,maximum,patch,overlap,screened_manifest="",screened_fingerprint=""):
    frames,sources=_source_bank(directory,maximum,screened_manifest,screened_fingerprint);rng=np.random.default_rng(seed)
    atlas=np.zeros((resolution,resolution),np.float32);stride=patch-overlap;used=set();placements=[]
    for y in range(0,resolution,stride):
        for x in range(0,resolution,stride):
            h=min(patch,resolution-y);w=min(patch,resolution-x);best=None;cost_best=float('inf');key_best=None
            for _ in range(12):
                index=int(rng.integers(len(frames)));source=frames[index]
                if source.shape[0]<patch or source.shape[1]<patch:
                    source=np.asarray(Image.fromarray(source).resize((max(patch,source.shape[1]),max(patch,source.shape[0])),Image.Resampling.BILINEAR))
                sx=int(rng.integers(0,source.shape[1]-patch+1));sy=int(rng.integers(0,source.shape[0]-patch+1));key=(index,sx,sy)
                if key in used:continue
                candidate=source[sy:sy+h,sx:sx+w].astype(np.float32);cost=0.
                if x:
                    delta=candidate[::8,:min(overlap,w):4]-atlas[y:y+h:8,x:x+min(overlap,w):4];cost+=float(np.mean(delta*delta))
                if y:
                    delta=candidate[:min(overlap,h):4,::8]-atlas[y:y+min(overlap,h):4,x:x+w:8];cost+=float(np.mean(delta*delta))
                if cost<cost_best:best=candidate;cost_best=cost;key_best=key
            if best is None:raise RuntimeError('Could not select a unique recorded patch')
            alpha=np.ones((h,w),np.float32)
            if x:
                length=min(overlap,w);alpha[:,:length]*=np.linspace(0,1,length,dtype=np.float32)[None,:]
            if y:
                length=min(overlap,h);alpha[:length,:]*=np.linspace(0,1,length,dtype=np.float32)[:,None]
            atlas[y:y+h,x:x+w]=best*alpha+atlas[y:y+h,x:x+w]*(1-alpha)
            used.add(key_best);placements.append({'x_px':x,'y_px':y,'width_px':w,'height_px':h,'source_index':key_best[0],'source_x_px':key_best[1],'source_y_px':key_best[2]})
    atlas.setflags(write=False)
    metadata={'source':SOURCE,'validation_state':'recorded_display_composite_not_physics','research_accepted':False,'source_frames':list(sources),'unique_source_frames_used':len({p['source_index'] for p in placements}),'patch_count':len(placements),'patches':placements,'texture_width_px':resolution,'texture_height_px':resolution,'source_pixel_min':int(min(a.min() for a in frames)),'source_pixel_max':int(max(a.max() for a in frames)),'normalization':'none; original grayscale values with feathered overlaps','rotation_or_mirroring':'none','world_scale_calibrated':False,'radiometric':False,'ground_truth':'unknown_recorded_background','target_free_verified':False,'synthetic_targets_injected':False,'limitations':['Unlabelled recorded frames can contain real targets or other objects.','Texture composite is not a spatial reconstruction or validated temperature field.','World scale is an explicit visualization assumption.']}
    return atlas,metadata


class RecordedBackground:
    def __init__(self,config,seed):
        directory=Path(config.get('recorded_background_dir') or os.environ.get('AEROROZUM_RECORDED_BACKGROUND_DIR') or DEFAULT_DIRECTORY).expanduser().resolve()
        resolution=int(config.get('recorded_background_resolution',2048));maximum=int(config.get('recorded_background_max_frames',64))
        patch=int(config.get('recorded_background_patch',256));overlap=int(config.get('recorded_background_overlap',48))
        if not 256<=resolution<=4096 or not 1<=maximum<=256 or not 32<=patch<=256 or not 1<=overlap<patch//2:raise ValueError('Invalid recorded-background quilting configuration')
        self.width_m=float(config['width_m']);self.height_m=float(config['height_m'])
        if self.width_m<=0 or self.height_m<=0:raise ValueError('World dimensions must be positive')
        screened=str(config.get('recorded_screened_crop_manifest',''))
        if not screened and 'recorded_background_dir' not in config and config.get('include_screened_zip_crops',True):
            import json
            from .common import data_root
            for candidate in (data_root()/'pipeline/inputs/backgrounds.json',data_root()/'backgrounds/annotation-excluded-train/manifest.json'):
                if candidate.is_file():
                    available=json.loads(candidate.read_text())
                    if available.get('status')=='screened' and available.get('accepted_crops',0)>0:
                        screened=str(candidate)
                        break
        fingerprint=hashlib.sha256(Path(screened).read_bytes()).hexdigest() if screened else ''
        expected=config.get('recorded_screened_crop_manifest_sha256')
        if expected and expected!=fingerprint:raise ValueError('Screened background manifest differs from frozen input')
        self.pixels,self.metadata=_quilt(str(directory),int(seed),resolution,maximum,patch,overlap,screened,fingerprint)
        self.metadata={**self.metadata,'screened_crop_manifest_sha256':fingerprint or None}
        self.x_m=np.linspace(0,self.width_m,resolution);self.y_m=np.linspace(0,self.height_m,resolution)

    def sample(self,xs,ys):
        """Return recorded grayscale intensities, never degrees kelvin."""
        xs,ys=np.broadcast_arrays(np.asarray(xs,dtype=float),np.asarray(ys,dtype=float))
        xi=np.clip(xs/self.width_m*(self.pixels.shape[1]-1),0,self.pixels.shape[1]-1);yi=np.clip(ys/self.height_m*(self.pixels.shape[0]-1),0,self.pixels.shape[0]-1)
        x0=np.floor(xi).astype(int);y0=np.floor(yi).astype(int);x1=np.minimum(x0+1,self.pixels.shape[1]-1);y1=np.minimum(y0+1,self.pixels.shape[0]-1)
        wx=xi-x0;wy=yi-y0
        return (1-wx)*(1-wy)*self.pixels[y0,x0]+wx*(1-wy)*self.pixels[y0,x1]+(1-wx)*wy*self.pixels[y1,x0]+wx*wy*self.pixels[y1,x1]


def export_annotation_excluded_crops(archive,output_dir,*,margin_px=16,min_side=96,maximum=512,allowed_splits=("train",)):
    """Export only annotation-excluded crops; none is accepted until model screening.

    Every source image is positive. Exclusion is a spatial annotation check,
    not proof that an unannotated object does not exist in a crop.
    """
    import json
    from io import BytesIO
    from zipfile import ZipFile
    output_dir=Path(output_dir).resolve();(output_dir/'crops').mkdir(parents=True,exist_ok=True)
    rows=[]
    with ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.json'):continue
            annotation=json.loads(z.read(name))
            if not isinstance(annotation,dict) or 'shapes' not in annotation:continue
            pieces=Path(name).stem.split('__');zone=pieces[1] if len(pieces)>1 else None
            split='test' if zone in ('zone8','zone9') else 'validation' if zone=='zone7' else 'train' if zone in ('zone1','zone2','zone3','zone4','zone5') else 'unknown'
            if split not in allowed_splits:continue
            source=str(Path(name).parent/annotation['imagePath'])
            if '..' in Path(source).parts:raise ValueError('Unsafe archive image path')
            raw=z.read(source)
            with Image.open(BytesIO(raw)) as opened:image=opened.convert('L')
            width,height=image.size;expanded=[]
            for shape in annotation['shapes']:
                points=shape.get('points',[])
                if not points:continue
                xs=[float(p[0]) for p in points];ys=[float(p[1]) for p in points]
                mx=max(margin_px,(max(xs)-min(xs))*.25);my=max(margin_px,(max(ys)-min(ys))*.25)
                expanded.append([max(0,int(np.floor(min(xs)-mx))),max(0,int(np.floor(min(ys)-my))),min(width,int(np.ceil(max(xs)+mx))),min(height,int(np.ceil(max(ys)+my)))])
            if not expanded:continue  # Do not reinterpret incomplete annotations as negative frames.
            x_edges=sorted({0,width,*[v for b in expanded for v in (b[0],b[2])]})
            y_edges=sorted({0,height,*[v for b in expanded for v in (b[1],b[3])]})
            candidates=[]
            for x1 in x_edges:
                for x2 in x_edges:
                    if x2-x1<min_side:continue
                    for y1 in y_edges:
                        for y2 in y_edges:
                            if y2-y1<min_side:continue
                            if any(x1<b[2] and x2>b[0] and y1<b[3] and y2>b[1] for b in expanded):continue
                            candidates.append((x1,y1,x2,y2))
            candidates.sort(key=lambda b:((b[2]-b[0])*(b[3]-b[1])),reverse=True)
            if not candidates:continue
            box=candidates[0];parent_hash=hashlib.sha256(raw).hexdigest()
            identity=hashlib.sha256((parent_hash+str(box)).encode()).hexdigest()[:24]
            target=output_dir/'crops'/f'{identity}.png';image.crop(box).save(target)
            rows.append({'id':identity,'image':target.relative_to(output_dir).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source':'annotation_excluded_crop','parent_image':Path(source).name,'parent_zone':zone,'split':split,'parent_sha256':parent_hash,'crop_xyxy':list(box),'excluded_boxes_with_margin_px':expanded,'minimum_margin_px':margin_px,'width':box[2]-box[0],'height':box[3]-box[1],'screening':{'status':'pending'},'eligible_for_background_use':False,'target_free_verified':False,'provenance':'annotation_exclusion_not_identity_verification'})
            if len(rows)>=maximum:break
    result={'schema_version':1,'source_archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),'status':'awaiting_model_screening','allowed_splits':list(allowed_splits),'crops':rows,'accepted_crops':0,'limitations':['All parent images contain positive annotations.','Excluding every annotated box does not prove the remaining crop is mine-free.','Only crops with an accepted model-screen record may enter the optional crop background bank.']}
    (output_dir/'manifest.json').write_text(json.dumps(result,indent=2))
    return result


def screened_crops(manifest,expected_model="gemini-robotics-er-2-preview",maximum=None):
    import json
    manifest=Path(manifest).resolve();document=json.loads(manifest.read_text());rows=[]
    artifact_bank=document.get('schema_version')==2 and document.get('image_root')=='artifact_root'
    if artifact_bank:
        from .common import data_root
        allowed_root=data_root().resolve()
    else:allowed_root=manifest.parent
    for row in document.get('crops',[]):
        screen=row.get('screening',{})
        if (screen.get('status')!='accepted' or screen.get('decision')!='no_mine_detected' or screen.get('model')!=expected_model or screen.get('rule')!='zero_candidates_at_any_confidence' or screen.get('detections')!=[] or row.get('detections')!=[]):continue
        if artifact_bank and row.get('split')!='train':continue
        if Path(row['image']).is_absolute():raise ValueError('Screened crop image must be relative')
        target=(allowed_root/row['image']).resolve()
        if not target.is_relative_to(allowed_root):raise ValueError('Screened crop escapes its archive')
        if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Screened crop checksum changed')
        rows.append((target,row))
        if maximum is not None and len(rows)>=int(maximum):break
    return rows


async def screen_annotation_crops(manifest,concurrency=8):
    """Screen exclusion crops with the exact Robotics ER 2 endpoint and retain provenance."""
    import asyncio
    import json
    from .common import config,data_root,save,utcnow,redacted_error,ROOT
    from .jobs import job,update
    from .providers import GeminiProvider
    from .labelling import label_frames,MODEL
    from .cloud_cache import GCSRecordStore
    manifest=Path(manifest).resolve();document=json.loads(manifest.read_text())
    rows=document['crops'];destination=manifest.parent/'labels'
    cfg={**config()['gemini'],**json.loads((ROOT/'configs/labelling.json').read_text()),'concurrency':concurrency}
    provider=GeminiProvider(cfg,output_dir=destination/'provider')
    task=job('recorded_background_crop_screening',model=MODEL,completed=0,total=len(rows),accepted_background_crops=0,human_reviewed=False)
    def merge():
        results={}
        for path in (destination/'results').glob('*.json'):
            try:
                item=json.loads(path.read_text())
                if item.get('model')==MODEL:results[item['id']]=item
            except (ValueError,OSError,KeyError):continue
        for row in rows:
            result=results.get(row['id'])
            if result is None:continue
            row['detections']=result['detections']
            if result.get('zero_candidates') and result['detections']==[] and result.get('screening',{}).get('rule')=='zero_candidates_at_any_confidence':
                row['screening']=result['screening'];row['eligible_for_background_use']=True
            else:
                row['screening']={'status':'rejected','decision':'candidate_or_uncertainty_present','model':MODEL,'human_reviewed':False};row['eligible_for_background_use']=False
            row['screened_at']=result['labelled_at'];row['target_free_verified']=False
        accepted=sum(bool(row.get('eligible_for_background_use')) for row in rows)
        document.update(status='screening_in_progress',screened_crops=len(results),accepted_crops=accepted,screening_model=MODEL,updated_at=utcnow())
        save(manifest,document);update(task,completed=len(results),accepted_background_crops=accepted)
    async def monitor():
        while True:
            merge();await asyncio.sleep(3)
    watcher=asyncio.create_task(monitor())
    try:
        store=GCSRecordStore(cfg.get('gcs_bucket','aerorozum-jetson-testdata-smart-bird'),prefix='robotics-er2-labels/5fps-v1/zip-crops')
        result=await label_frames(rows,destination,provider=provider,model=MODEL,threshold=.6,concurrency=concurrency,image_root=manifest.parent,record_store=store)
        watcher.cancel()
        try:await watcher
        except asyncio.CancelledError:pass
        merge();document['status']='screened' if result['status']=='completed' else 'partially_screened'
        save(manifest,document);update(task,result['status'],completed=result['labelled'],failed=result['failed'],accepted_background_crops=document['accepted_crops'])
        return {**result,'accepted_background_crops':document['accepted_crops'],'screened_crop_manifest':str(manifest),'human_reviewed':False,'verified_negative':False}
    except Exception as error:
        update(task,'failed',error=redacted_error(error));raise
    finally:
        watcher.cancel();await provider.close()


def archive_crop_cache(manifest,bucket='aerorozum-jetson-testdata-smart-bird',*,once=False,interval_s=10,max_seconds=1800):
    """Archive already-paid crop records without issuing any model requests."""
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from .cloud_cache import GCSRecordStore
    from .common import save,utcnow,redacted_error
    from .jobs import job,update
    manifest=Path(manifest).resolve();folder=manifest.parent/'labels';destination=manifest.parent/'cloud_archive'
    destination.mkdir(exist_ok=True);state_path=destination/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'records':{},'raw_text_missing':0}
    store=GCSRecordStore(bucket,prefix='robotics-er2-labels/5fps-v1/zip-crops')
    record=job('zip_crop_cloud_archive',provider='google_cloud_storage',bucket=bucket,completed=len(state['records']),detail='Archiving existing paid labels and available prompt/response logs; no inference calls')
    started=time.monotonic()
    def upload(path,response_index):
        payload=json.loads(path.read_text());raw=response_index.get(payload['sha256'])
        if raw is None:raw={'raw_text_available':False,'reason':'No saved provider response record matched this crop'}
        else:raw={**raw,'raw_text_available':bool(raw.get('raw_text')),'verbatim_response_limit':None if raw.get('raw_text') else 'Provider version running when this request was made saved parsed output, prompt and usage but not the original response text. No replacement inference was issued.'}
        uris=store.write_frame(path.stem,payload,raw_log=raw)
        return path.stem,{'id':payload['id'],'image_sha256':payload['sha256'],'parent_sha256':payload.get('parent_sha256'),'uris':uris,'raw_text_available':bool(raw.get('raw_text_available'))}
    try:
        while True:
            response_index={}
            for path in (folder/'provider/responses').glob('*.json'):
                try:
                    response=json.loads(path.read_text())
                    if response.get('model')=='gemini-robotics-er-2-preview':
                        for image_hash in response.get('image_hashes',[]):response_index[image_hash]=response
                except (ValueError,OSError):continue
            pending=[p for p in (folder/'results').glob('*.json') if p.stem not in state['records']]
            with ThreadPoolExecutor(max_workers=8) as pool:
                for future in as_completed([pool.submit(upload,path,response_index) for path in pending]):
                    key,value=future.result();state['records'][key]=value
                    state.update(updated_at=utcnow(),bucket=bucket,prefix=store.prefix,raw_text_missing=sum(not r['raw_text_available'] for r in state['records'].values()))
                    save(state_path,state);update(record,completed=len(state['records']),raw_text_missing=state['raw_text_missing'])
            progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {}
            done=progress.get('status') in ('completed','partial') and len(state['records'])>=int(progress.get('labelled',progress.get('completed',0)))
            if once or done or time.monotonic()-started>=max_seconds:break
            time.sleep(interval_s)
        document={'updated_at':utcnow(),'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'archived_frames':len(state['records']),'raw_text_missing':state.get('raw_text_missing',0),'records':state['records'],'no_replacement_inference':True,'complete':done}
        summary_uri=store.put(hashlib.sha256(json.dumps(document,sort_keys=True).encode()).hexdigest(),document,'progress')
        save(destination/'summary.json',{**document,'cloud_manifest':summary_uri})
        update(record,'completed' if done else 'partial',completed=len(state['records']),cloud_manifest=summary_uri)
        return {'archived_frames':len(state['records']),'raw_text_missing':state.get('raw_text_missing',0),'cloud_manifest':summary_uri,'complete':done}
    except Exception as error:
        update(record,'failed',error=redacted_error(error));raise
