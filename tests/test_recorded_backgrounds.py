import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
import pytest
from aerorozum.recorded_backgrounds import RecordedBackground,original_frames,export_annotation_excluded_crops,screened_crops
from aerorozum.renderer import ThermalRenderer


def frames(tmp_path):
    rng=np.random.default_rng(51)
    for i in range(3):
        a=np.clip(rng.normal(96+i*3,9,(512,640)),45,150).astype('uint8')
        Image.fromarray(a).save(tmp_path/f'frame_{i:06d}.jpg')
    Image.new('L',(640,512),255).save(tmp_path/'frame_000000_boxes.jpg')
    return tmp_path


def cfg(tmp_path):
    return {'width_m':64,'height_m':64,'camera':{'width':128,'height':128,'hfov_deg':100,'vfov_deg':82},'recorded_background_dir':str(frames(tmp_path)),'recorded_background_resolution':512,'recorded_background_max_frames':3}


def test_originals_only_and_deterministic_nonperiodic_quilt(tmp_path):
    c=cfg(tmp_path);assert len(original_frames(tmp_path))==3
    a=RecordedBackground(c,3);b=RecordedBackground(c,3)
    assert np.array_equal(a.pixels,b.pixels)
    assert a.metadata['source']=='recorded_background_composite_unvalidated'
    assert a.metadata['target_free_verified'] is False
    assert a.pixels.min()>=a.metadata['source_pixel_min']
    assert a.pixels.max()<=a.metadata['source_pixel_max']
    assert not np.array_equal(a.pixels[:200,:200],a.pixels[208:408,208:408])
    assert all(len(s['sha256'])==64 for s in a.metadata['source_frames'])


def test_render_preserves_raw_tone_and_does_not_inject_targets(tmp_path):
    c=cfg(tmp_path);pose={'x_m':32,'y_m':32,'altitude_m':4,'yaw_deg':0}
    a=ThermalRenderer(c,3,[{'id':'not-real','x_m':32,'y_m':32,'radius_m':5}])
    b=ThermalRenderer(c,3,[])
    a.render(pose,21600,tmp_path/'a.png');b.render(pose,80000,tmp_path/'b.png')
    assert np.array_equal(np.asarray(Image.open(tmp_path/'a.png')),np.asarray(Image.open(tmp_path/'b.png')))
    assert a.evidence_targets(pose,21600)==[]
    xs,ys=a.ground_grid(pose)
    expected=np.rint(a.recorded_background.sample(xs,ys)).astype('uint8')
    assert np.array_equal(np.asarray(Image.open(tmp_path/'a.png').convert('L')),expected)
    assert json.loads((tmp_path/'a.source.json').read_text())['ground_truth']=='unknown_recorded_background'


def test_missing_real_sources_never_silently_fall_back_to_noise(tmp_path):
    with pytest.raises(ValueError,match='original thermal frames'):
        RecordedBackground({'width_m':64,'height_m':64,'recorded_background_dir':str(tmp_path)},1)


def test_positive_archive_only_exports_box_excluded_pending_crops(tmp_path):
    import io
    raw=io.BytesIO();Image.new('L',(336,256),90).save(raw,format='JPEG')
    annotation={'imagePath':'03032020__zone1__frame.jpg','shapes':[{'points':[[130,80],[190,145]],'shape_type':'rectangle','label':'mine'}]}
    archive=tmp_path/'positive.zip'
    with ZipFile(archive,'w') as z:z.writestr('data/03032020__zone1__frame.jpg',raw.getvalue());z.writestr('data/03032020__zone1__frame.json',json.dumps(annotation))
    out=tmp_path/'crops';d=export_annotation_excluded_crops(archive,out)
    assert d['status']=='awaiting_model_screening' and d['crops']
    row=d['crops'][0];x1,y1,x2,y2=row['crop_xyxy']
    assert all(not(x1<b[2] and x2>b[0] and y1<b[3] and y2>b[1]) for b in row['excluded_boxes_with_margin_px'])
    assert screened_crops(out/'manifest.json')==[]
    row['detections']=[]
    row['screening']={'status':'accepted','decision':'no_mine_detected','model':'gemini-robotics-er-2-preview','detections':[],'rule':'zero_candidates_at_any_confidence'}
    (out/'manifest.json').write_text(json.dumps(d))
    assert len(screened_crops(out/'manifest.json'))==1


def test_paid_crop_archive_never_regenerates_missing_raw_text(tmp_path,monkeypatch):
    from aerorozum.recorded_backgrounds import archive_crop_cache
    from aerorozum import cloud_cache
    monkeypatch.setenv('AEROROZUM_DATA_ROOT',str(tmp_path/'artifacts'))
    root=tmp_path/'crops';(root/'labels/results').mkdir(parents=True);(root/'labels/provider/responses').mkdir(parents=True)
    (root/'manifest.json').write_text('{"crops":[]}')
    payload={'id':'crop-one','sha256':'a'*64,'model':'gemini-robotics-er-2-preview','model_output':{'detections':[]},'parent_sha256':'b'*64}
    (root/'labels/results'/('c'*64+'.json')).write_text(json.dumps(payload))
    (root/'labels/provider/responses'/('d'*64+'.json')).write_text(json.dumps({'image_hashes':['a'*64],'model':'gemini-robotics-er-2-preview','prompt':'original saved prompt','output':{'detections':[]},'usage':{'total_tokens':42}}))
    (root/'labels/progress.json').write_text('{"status":"completed","labelled":1}')
    archived=[]
    class Store:
        def __init__(self,bucket,prefix):self.prefix=prefix
        def write_frame(self,key,record,raw_log=None):
            archived.append((record,raw_log));return {'result':'gs://fixture/result.json','raw_log':'gs://fixture/log.json','labels':'gs://fixture/labels.json'}
        def put(self,key,record,kind):return 'gs://fixture/progress.json'
    monkeypatch.setattr(cloud_cache,'GCSRecordStore',Store)
    result=archive_crop_cache(root/'manifest.json',once=True)
    assert result['archived_frames']==1 and result['raw_text_missing']==1
    assert archived[0][1]['raw_text_available'] is False
    assert archived[0][1]['prompt']=='original saved prompt'
    assert archived[0][1]['usage']['total_tokens']==42


def test_model_uncertainty_is_not_a_background_even_below_positive_threshold(tmp_path):
    from io import BytesIO
    raw=BytesIO();Image.new('L',(336,256),90).save(raw,format='JPEG')
    archive=tmp_path/'positive-zones.zip';ann={'imagePath':'03032020__zone1__frame.jpg','shapes':[{'points':[[140,90],[180,130]],'shape_type':'rectangle','label':'mine'}]}
    with ZipFile(archive,'w') as z:
        z.writestr('data/03032020__zone1__frame.jpg',raw.getvalue());z.writestr('data/03032020__zone1__frame.json',json.dumps(ann))
        ann['imagePath']='03032020__zone8__frame.jpg';z.writestr('data/03032020__zone8__frame.jpg',raw.getvalue());z.writestr('data/03032020__zone8__frame.json',json.dumps(ann))
    out=tmp_path/'candidate-crops';d=export_annotation_excluded_crops(archive,out)
    assert len(d['crops'])==1 and d['crops'][0]['parent_zone']=='zone1'
    row=d['crops'][0]
    row['screening']={'status':'accepted','decision':'no_mine_detected','model':'gemini-robotics-er-2-preview','rule':'zero_candidates_at_any_confidence','detections':[],'confidence':1}
    row['detections']=[{'confidence':.1,'bbox':{'x_min':1,'y_min':1,'x_max':20,'y_max':20}}]
    (out/'manifest.json').write_text(json.dumps(d))
    assert screened_crops(out/'manifest.json')==[]

def test_pipeline_background_bank_retains_train_only_and_checks_sha(tmp_path,monkeypatch):
 import hashlib,json
 from aerorozum.recorded_backgrounds import screened_crops
 from PIL import Image
 monkeypatch.setenv('AEROROZUM_DATA_ROOT',str(tmp_path));image=tmp_path/'real.png';Image.new('L',(128,128),90).save(image)
 screen={'status':'accepted','decision':'no_mine_detected','model':'gemini-robotics-er-2-preview','rule':'zero_candidates_at_any_confidence','detections':[]}
 row={'id':'native','image':'real.png','sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'split':'train','detections':[],'screening':screen}
 manifest=tmp_path/'pipeline/inputs/backgrounds.json';manifest.parent.mkdir(parents=True);manifest.write_text(json.dumps({'schema_version':2,'image_root':'artifact_root','crops':[row,{**row,'split':'test'}]}))
 assert len(screened_crops(manifest,maximum=16))==1
 image.write_bytes(b'modified')
 import pytest
 with pytest.raises(ValueError,match='checksum'):screened_crops(manifest)
