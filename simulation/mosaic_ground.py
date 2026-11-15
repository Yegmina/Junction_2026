"""Blend training-only recorded thermal crops into a native material atlas."""
import hashlib,json,random
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

BASE=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl')

def main():
    doc=json.loads((BASE/'artifacts/pipeline/inputs/backgrounds.json').read_text())
    rows=[r for r in doc['crops'] if r.get('split')=='train' and r.get('row_kind')=='zip_screened_background' and r.get('screening',{}).get('status')=='accepted' and r.get('detections')==[] and r.get('image_statistics',{}).get('gray_stddev',0)>12]
    if not rows:raise ValueError('No approved training-only textures')
    rng=random.Random(20261114);rng.shuffle(rows);size=4608;tile=384;stride=256
    total=np.zeros((size,size),np.float32);weights=np.zeros_like(total)
    ramp=np.maximum(.001,np.sin(np.linspace(0,np.pi,tile))**2).astype('float32');window=ramp[:,None]*ramp[None,:]
    sources={};index=0
    for y in range(0,size-tile+1,stride):
        for x in range(0,size-tile+1,stride):
            row=rows[index%len(rows)];index+=1;path=BASE/'artifacts'/row['image'];raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Texture hash mismatch')
            sources[row['sha256']]={k:row.get(k) for k in ('sha256','image','split','source','screening')}
            image=Image.open(path).convert('L').resize((tile,tile),Image.Resampling.BICUBIC).rotate(rng.choice([0,90,180,270]))
            if rng.random()<.5:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            a=np.asarray(image,dtype=np.float32);a=(a-float(a.mean()))/max(8,float(a.std()))*rng.uniform(18,25)+rng.uniform(120,136)
            total[y:y+tile,x:x+tile]+=a*window;weights[y:y+tile,x:x+tile]+=window
    atlas=total/np.maximum(weights,.000001);atlas=atlas[128:4224,128:4224]
    # Retain recorded spatial texture; a broad display-only trend avoids a flat
    # photometric mean. Neither this normalization nor K mapping is calibration.
    low=gaussian_filter(np.random.default_rng(20261114).normal(size=(64,64)),3)
    low=(low-low.mean())/max(.001,low.std())
    trend=np.asarray(Image.fromarray(low.astype('float32')).resize((4096,4096),Image.Resampling.BICUBIC))
    atlas=np.clip(atlas+trend*5,0,255).astype('uint8')
    output=BASE/'simulation/models/recorded_soil/materials/textures/recorded_ground.png';Image.fromarray(atlas).save(output)
    receipt=dict(kind='recorded_texture_mosaic',seed=20261114,width=4096,height=4096,world_extent_m=[70,70],assigned_display_pixel_m=70/4096,scale_calibrated=False,source_count=len(sources),patch_count=index,patch_px=tile,stride_px=stride,blend='overlapping cosine feather; randomized quarter turns and reflection',photometry='per-patch display normalization plus broad display trend; not radiometric',source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),target_free_verified=False,screening_status='teacher_screened_no_proposals_not_independently_target_free',temperature_mapping=dict(minimum_k=293,maximum_k=297,assumed_not_measured=True),sources=list(sources.values()))
    manifest=BASE/'simulation/provenance/recorded-ground-mosaic.json';manifest.write_text(json.dumps(receipt,indent=2))
    compact={k:v for k,v in receipt.items() if k!='sources'};compact.update(source_manifest=str(manifest),source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),reprojection='Native moving Gazebo camera rendering a recorded texture material, not frame replay',radiometric=False)
    (BASE/'simulation/provenance/recorded-ground.json').write_text(json.dumps(compact,indent=2));print(compact)

if __name__=='__main__':main()
