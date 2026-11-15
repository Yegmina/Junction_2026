"""Accepted-pair, bounded-memory physical patch rendering.

This module never promotes a solver result or stretches a patch into a survey
world. Larger-field composition must have a separate held-out validation report.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import shutil
from zipfile import ZipFile
import hashlib

import numpy as np
from PIL import Image
from scipy.ndimage import label,find_objects,gaussian_filter
from .common import digest,save,utcnow


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_pair(entry):
    if entry.get('status')!='accepted':raise ValueError('Pair is not accepted by the physics coordinator')
    if entry.get('split') not in ('train','validation','test','fast_test','direct_test') or not entry.get('campaign_id'):raise ValueError('Explicit frozen physical campaign identity and split required')
    present=Path(entry['field_path']).resolve();absent=Path(entry['counterfactual_path']).resolve()
    report=json.loads(Path(entry['acceptance_report_path']).read_text())
    if report.get('accepted') is not True or report.get('research_accepted') is not True or report.get('status')!='numerically_accepted_pair':
        raise ValueError('Executed numerical pair-acceptance certificate required')
    hashes={}
    for name,path in (('present',present),('absent',absent)):
        actual=file_hash(path);expected=report.get(name+'_sha256')
        if expected!=actual:raise ValueError(f'{name} field hash differs from accepted pair')
        metadata=json.loads(path.with_suffix('.json').read_text())
        if metadata.get('research_accepted') is not True or metadata.get('validation_state')!='numerically_accepted':
            raise ValueError('Physical field is still experimental/unaccepted')
        if metadata.get('field_sha256')!=actual:raise ValueError('Field metadata does not bind the numerical acceptance to these bytes')
        if 'dumux' not in str(metadata.get('backend',metadata.get('solver',''))).lower():raise ValueError('Accepted direct DuMux provenance required')
        hashes[name]=actual
    return {'present':present,'absent':absent,'report':report,'hashes':hashes,
            'certificate_sha256':file_hash(entry['acceptance_report_path'])}


class SurfaceField:
    def __init__(self,path,cache_root):
        self.path=Path(path);self.sha=file_hash(self.path);folder=Path(cache_root)/self.sha;folder.mkdir(parents=True,exist_ok=True)
        self.data={}
        with ZipFile(self.path) as archive:
            for name in ('times_s','x_m','y_m','temperature_k','moisture'):
                target=folder/(name+'.npy')
                if not target.is_file():
                    temporary=target.with_suffix('.partial')
                    with archive.open(name+'.npy') as source,temporary.open('wb') as output:shutil.copyfileobj(source,output,1024*1024)
                    temporary.replace(target)
                self.data[name]=np.load(target,mmap_mode='r',allow_pickle=False)
        for key in ('times_s','x_m','y_m'):
            axis=self.data[key]
            if axis.ndim!=1 or len(axis)<2 or not np.isfinite(axis).all() or not np.all(np.diff(axis)>0):raise ValueError('Invalid physical coordinate/time axis')
        shape=(len(self.data['times_s']),len(self.data['y_m']),len(self.data['x_m']))
        if self.data['temperature_k'].shape!=shape or self.data['moisture'].shape!=shape:raise ValueError('Invalid surface field shape')
        self.metadata=json.loads(self.path.with_suffix('.json').read_text())

    def sample(self,xs,ys,time_s):
        times=self.data['times_s']
        if not times[0]<=time_s<=times[-1]:raise ValueError('Physical-time extrapolation is forbidden')
        j=min(max(1,int(np.searchsorted(times,time_s))),len(times)-1)
        weight=(time_s-times[j-1])/(times[j]-times[j-1])
        x=np.interp(xs,self.data['x_m'],np.arange(len(self.data['x_m'])))
        y=np.interp(ys,self.data['y_m'],np.arange(len(self.data['y_m'])))
        x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
        x1=np.minimum(x0+1,len(self.data['x_m'])-1);y1=np.minimum(y0+1,len(self.data['y_m'])-1)
        wx=x-x0;wy=y-y0
        result=0.
        for k,alpha in ((j-1,1-weight),(j,weight)):
            a=self.data['temperature_k'][k]
            result=result+alpha*((1-wx)*(1-wy)*a[y0,x0]+wx*(1-wy)*a[y0,x1]+(1-wx)*wy*a[y1,x0]+wx*wy*a[y1,x1])
        if not np.isfinite(result).all():raise ValueError('Nonfinite physical temperatures')
        return result


def render_pair_shard(entry,output_root,artifact_root,*,count=64,seed=17,shape=(640,512)):
    """Emit a numerically accepted single-scene shard as soon as its pair arrives."""
    if int(count)<2 or int(count)%2:raise ValueError('Matched present/absent shard needs an even image count')
    checked=validate_pair(entry);base=Path(artifact_root).resolve();root=Path(output_root).resolve()
    identity=digest([entry.get('campaign_id'),checked['hashes'],checked['certificate_sha256'],count,seed,shape,'native-gsd-crop-v4-shared-transfer'])[:24]
    folder=root/identity;manifest=folder/'manifest.json'
    if manifest.exists():return json.loads(manifest.read_text())
    present=SurfaceField(checked['present'],root/'.field-cache');absent=SurfaceField(checked['absent'],root/'.field-cache')
    for axis in ('x_m','y_m'):
        if not np.array_equal(present.data[axis],absent.data[axis]):raise ValueError('Counterfactual spatial grids differ')
    start=max(present.data['times_s'][0],absent.data['times_s'][0]);end=min(present.data['times_s'][-1],absent.data['times_s'][-1])
    if end<=start:raise ValueError('No common physical time coverage')
    domain=present.metadata.get('domain',{});w=float(domain.get('width_m',present.data['x_m'][-1]));h=float(domain.get('height_m',present.data['y_m'][-1]))
    sensor_width,sensor_height=map(int,shape);rng=np.random.default_rng(seed);rows=[]
    altitude=4.;gsd_x=2*altitude*math.tan(math.radians(100)/2)/sensor_width
    gsd_y=2*altitude*math.tan(math.radians(82)/2)/sensor_height
    for pair_index,time_s in enumerate(np.linspace(start,end,int(count)//2)):
        # Render the actual native-GSD crop of a 4m-altitude camera. A1m patch
        # yields only ~50–70 native pixels, never a stretched640px full frame.
        fraction=float(rng.uniform(.65,.95));foot_w=w*fraction;foot_h=h*fraction
        width=min(sensor_width,int(foot_w/gsd_x));height=min(sensor_height,int(foot_h/gsd_y))
        if width<3 or height<3:raise ValueError('Physical patch has fewer than3 native pixels at the declared camera GSD')
        foot_w=width*gsd_x;foot_h=height*gsd_y
        cx=float(rng.uniform(foot_w/2,w-foot_w/2));cy=float(rng.uniform(foot_h/2,h-foot_h/2))
        xs,ys=np.meshgrid(cx+((np.arange(width)+.5)/width-.5)*foot_w,cy+(.5-(np.arange(height)+.5)/height)*foot_h)
        a=gaussian_filter(present.sample(xs,ys,float(time_s)),.65)
        b=gaussian_filter(absent.sample(xs,ys,float(time_s)),.65);difference=np.abs(a-b)
        components,_=label(difference>=.12);boxes=[]
        for index,slices in enumerate(find_objects(components),1):
            if slices is None:continue
            yy,xx=slices
            if xx.stop-xx.start<3 or yy.stop-yy.start<3 or np.sum(components[slices]==index)<9:continue
            boxes.append({'label':'provisional thermal signature','bbox':{'x_min':xx.start/width*1000,'y_min':yy.start/height*1000,'x_max':xx.stop/width*1000,'y_max':yy.stop/height*1000}})
        noise=rng.normal(0,.04,(height,width));low,high=np.percentile(np.stack([a,b]),(2,98))
        for target_present,temperatures in ((True,a),(False,b)):
            sample_id=digest([identity,pair_index,target_present]);path=folder/'images'/(sample_id+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            pixels=np.clip((temperatures+noise-low)/max(float(high-low),.5)*230+12,0,255).astype(np.uint8)
            Image.fromarray(pixels).convert('RGB').save(path)
            if not path.is_relative_to(base):raise ValueError('Shard must be stored under artifact root')
            rows.append({'id':sample_id,'image':str(path.relative_to(base)),'sha256':file_hash(path),'width':width,'height':height,
                'split':entry.get('split','train'),'source_group':entry.get('campaign_id',identity),'provenance':'direct_physics',
                'research_accepted':True,'fidelity':'numerically_accepted_direct_physical_patch','boxes':boxes if target_present else [],
                'training_role':'auxiliary_physics_cue_only','appearance_validation':'nominal camera transfer, not calibrated to field imagery',
                'acceptance_scope':'numerical field and matched-counterfactual visibility; realistic appearance not implied',
                'physical_field_sha256':checked['hashes']['present' if target_present else 'absent'],
                'counterfactual_sha256':checked['hashes']['absent' if target_present else 'present'],
                'paired_present_sha256':checked['hashes']['present'],'paired_absent_sha256':checked['hashes']['absent'],'target_present_in_physical_patch':target_present,
                'acceptance_certificate_sha256':checked['certificate_sha256'],
                'capture_time_s':float(time_s),'field_domain_m':[w,h],'camera_footprint_m':[foot_w,foot_h],
                'implied_nadir_altitude_m':altitude,'native_ground_sample_distance_m':[gsd_x,gsd_y],
                'full_sensor_resolution':[sensor_width,sensor_height],
                'effective_crop_fov_deg':[math.degrees(2*math.atan(foot_w/(2*altitude))),math.degrees(2*math.atan(foot_h/(2*altitude)))],
                'observation_scope':'native-GSD crop of physical patch at4m altitude; not a full-flight scene',
                'observable_label_protocol':'matched post-PSF physical difference >=0.12K (3x nominal NETD), at least3x3 visible pixels',
                'camera_transfer':{'agc':'shared matched-pair 2nd/98th percentile, minimum0.5K range','agc_window_k':[float(low),float(low+max(high-low,.5))],'gaussian_noise_std_k':.04,'gaussian_blur_sigma_pixels':.65},
                'physical_mine_identity_confirmed':False})
    doc={'metadata':{'status':'passed','research_accepted':True,'namespace':'incremental-stream-v1','created_at':utcnow(),
                     'pair_hashes':checked['hashes'],'certificate_sha256':checked['certificate_sha256'],'count':len(rows),
                     'training_role':'auxiliary_physics_cue_only','recorded_background_compositor_status':'not validated; no hybrid training images generated',
                     'limitation':'Numerically accepted simulation patch; field validation and 64m composition not implied'},'images':rows}
    save(manifest,doc)
    # Raw compressed solver artifacts remain authoritative. Expanded mmap arrays
    # are only a render scratchpad, not another unbounded copy of the library.
    for field in (present,absent):
        for array in field.data.values():
            mapped=getattr(array,'_mmap',None)
            if mapped is not None:mapped.close()
    for field_hash in set(checked['hashes'].values()):shutil.rmtree(root/'.field-cache'/field_hash,ignore_errors=True)
    return doc
