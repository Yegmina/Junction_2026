import importlib.util,math,sys
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'harbor/thermal-investigation/solution/vision.py'
sys.path.insert(0,str(path.parents[1]/'environment'))
spec=importlib.util.spec_from_file_location('reference_vision',path);vision=importlib.util.module_from_spec(spec);spec.loader.exec_module(vision)


def calibration():
    camera=dict(width=128,height=96,intrinsics=dict(fx=100.,fy=100.),
        camera_mount=dict(x_m=0,y_m=0,z_m=-.12,roll_rad=0,pitch_rad=math.pi/2,yaw_rad=0))
    return {**camera,'pose':dict(x_m=0,y_m=0,z_m=3.),'rgb_calibration':camera}


def test_registered_identical_cameras_preserve_actual_pixel_locations():
    yy,xx=np.mgrid[:96,:128];rgb=np.stack([xx,yy,xx+yy],axis=-1).astype(np.uint8)
    aligned,valid=vision.registered_rgb(calibration(),rgb,(96,128))
    np.testing.assert_allclose(aligned[valid]*255,rgb[valid],atol=1e-10)


@pytest.mark.parametrize('phase,sign',[('warming',1),('dawn',-1)])
def test_rgb_explained_ground_is_rejected_but_compact_thermal_residual_is_proposed(phase,sign):
    yy,xx=np.mgrid[:96,:128];r=40+xx;g=30+yy;b=(xx+yy)//2
    rgb=np.stack([r,g,b],axis=-1).astype(np.uint8);ground=30+.65*r+.25*g
    assert vision.proposals(calibration(),ground,rgb,phase)==[]
    thermal=ground+sign*45*np.exp(-((xx-70)**2+(yy-45)**2)/20.)
    proposed=vision.proposals(calibration(),thermal,rgb,phase)
    assert any(math.hypot(u-70,v-45)<2 for u,v,_,strength in proposed)
