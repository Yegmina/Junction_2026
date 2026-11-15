import hashlib,json
import numpy as np
import pytest
from simulation.physics_camera import PhysicsCamera
from aerorozum.gazebo_bridge import GazeboBridge,GazeboUnavailable

def camera(tmp_path):
    y,x=np.mgrid[-1:1:100j,-1:1:100j];p=tmp_path/'temperature_k.npy';np.save(p,(300+.4*x+.2*y).astype('float32'))
    meta=dict(thermal_field_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),world_bounds=dict(x_min=-5,y_min=-5,x_max=5,y_max=5),physics_time_s=27000,physics=dict(backend='dumux_extended_richards_ni'))
    (tmp_path/'scene.json').write_text(json.dumps(meta));return PhysicsCamera(tmp_path)

def test_camera_ground_projection_uses_gazebo_axes_and_mount(tmp_path):
    c=camera(tmp_path);pose=dict(x_m=0,y_m=0,z_m=2.12)
    x,y,valid=c.ground_points(pose,641,513)
    assert abs(x[256,320])<1e-8 and abs(y[256,320])<1e-8 and valid.all()
    assert x[0,320]>0 and y[256,-1]<0
    xx,yy,_=c.ground_points({**pose,'z_m':4.12},641,513)
    assert np.allclose(xx,2*x) and np.allclose(yy,2*y)

def test_thermal_is_grayscale_with_sensor_range_and_field_binding(tmp_path):
    c=camera(tmp_path);pose=dict(x_m=0,y_m=0,z_m=3)
    im,meta=c.render(pose,sim_time_s=1)
    assert im.mode=='L' and len(np.unique(np.asarray(im)))>64
    assert meta['source_kind']=='dumux_field_crop' and meta['field_sha256']==c.metadata['thermal_field_sha256']
    assert meta['research_accepted'] is False and meta['display_mapping']['high_k']>meta['display_mapping']['low_k']
    again,_=c.render(pose,sim_time_s=1);assert im.tobytes()==again.tobytes()

def test_corrupted_physics_field_is_rejected(tmp_path):
    camera(tmp_path);(tmp_path/'temperature_k.npy').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):PhysicsCamera(tmp_path)

def test_pair_with_matching_id_but_different_timestamp_is_rejected():
    class BadPair(GazeboBridge):
        def _request(self,*args,**kwargs):return dict(thermal=dict(pair_id='same',sim_time_s=1,pose={}),rgb=dict(pair_id='same',sim_time_s=2,pose={}))
    with pytest.raises(GazeboUnavailable,match='timestamp'):BadPair().capture_pair('/unused','unused')
