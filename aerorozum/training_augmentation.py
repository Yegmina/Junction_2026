"""Deterministic image/box flips for already admitted training rows only."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from PIL import Image
from .common import digest,save


def flip_box(box,horizontal=False,vertical=False):
    result=dict(box)
    if horizontal:result.update(x_min=1000-box['x_max'],x_max=1000-box['x_min'])
    if vertical:result.update(y_min=1000-box['y_max'],y_max=1000-box['y_min'])
    return result


def augment_training_rows(rows,cache):
    """Expand each original into identity, horizontal, vertical and both flips.

    Original labels/provenance are never edited. Returned derivatives are
    materialized training examples, not additional independent observations.
    This function is called after the frozen source manifest has been checked.
    """
    if not rows:raise ValueError('Training rows required')
    # Reject held-out inputs before creating a derivative or cache directory.
    for row in rows:
        zone=row.get('zone') or row.get('parent_zone') or row.get('source',{}).get('zone')
        if row.get('split')!='train' or zone in ('zone7','zone8','zone9'):
            raise ValueError('Only training rows can be augmented')
    output=[];identities=[];cache=Path(cache)
    transforms=[('horizontal',True,False),('vertical',False,True),('both',True,True)]
    for row in rows:
        source=Path(row['image_path']);payload=source.read_bytes()
        sha=hashlib.sha256(payload).hexdigest()
        if sha!=row['sha256']:raise ValueError('Augmentation source checksum changed')
        output.append(row)
        with Image.open(io.BytesIO(payload)) as original:
            original=original.convert('RGB')
            for name,horizontal,vertical in transforms:
                folder=cache/sha;folder.mkdir(parents=True,exist_ok=True)
                path=folder/(name+'.png');receipt=folder/(name+'.json')
                if path.exists() and receipt.exists():
                    identity=json.loads(receipt.read_text())
                    if identity.get('parent_sha256')!=sha or identity.get('transform')!=name or hashlib.sha256(path.read_bytes()).hexdigest()!=identity.get('sha256'):
                        raise ValueError('Augmentation cache checksum changed')
                else:
                    image=original
                    if horizontal:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if vertical:image=image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    stream=io.BytesIO();image.save(stream,format='PNG');data=stream.getvalue()
                    fd,temp=tempfile.mkstemp(dir=folder,prefix='.'+name,suffix='.tmp')
                    try:
                        with os.fdopen(fd,'wb') as f:f.write(data)
                        os.replace(temp,path)
                    finally:Path(temp).unlink(missing_ok=True)
                    identity=dict(parent_sha256=sha,transform=name,sha256=hashlib.sha256(data).hexdigest())
                    save(receipt,identity)
                boxes=[flip_box(box,horizontal,vertical) for box in row['gold_boxes']]
                output.append({**row,'image_path':str(path),'gold_boxes':boxes,'augmentation':identity})
                identities.append({**identity,'gold_boxes':boxes})
    return output,dict(protocol='training-hv-flips-v1',source_images=len(rows),materialized_examples=len(output),
        independent_observations=len(rows),derivative_manifest_sha256=digest(identities),
        scope='Geometry-preserving training augmentation only; no new labels or physical fields')
