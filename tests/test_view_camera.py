import math
import numpy as np
import pytest
from simulation.view_camera import DEFAULT,settings,camera_pose,rotation


def test_native_camera_orbits_drone_without_changing_drone_pose():
    drone=dict(x_m=10.,y_m=-4.,z_m=5.)
    original=dict(drone)
    a=camera_pose(DEFAULT,drone)['sensor_pose']
    b=camera_pose(settings(DEFAULT,{'azimuth_deg':DEFAULT['azimuth_deg']+90}),drone)['sensor_pose']
    assert drone==original
    assert a['z_m']==b['z_m'] and not np.allclose([a['x_m'],a['y_m']],[b['x_m'],b['y_m']])
    forward=rotation(a['roll_rad'],a['pitch_rad'],a['yaw_rad'])[:,0]
    target=np.array([10.,-4.,5.05]);pos=np.array([a[k] for k in ['x_m','y_m','z_m']])
    assert np.allclose(forward,(target-pos)/np.linalg.norm(target-pos))


def test_model_transform_compensates_the_existing_offset_sensor():
    mount=(22.,-24.,21.,0.,.57,2.31)
    p=camera_pose(DEFAULT,dict(x_m=-3,y_m=8,z_m=4),mount)
    q=p['orientation'];x,y,z,w=[q[k] for k in ['x','y','z','w']]
    r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    actual=np.array(list(p['position'].values()))+r@np.array(mount[:3])
    assert np.allclose(actual,[p['sensor_pose'][k] for k in ['x_m','y_m','z_m']])
    assert np.allclose(r@rotation(*mount[3:]),rotation(0,p['sensor_pose']['pitch_rad'],p['sensor_pose']['yaw_rad']))


@pytest.mark.parametrize('update',[{'distance_m':float('nan')},{'elevation_deg':-1},{'mode':'fly'}, {'pan_x_m':True},{'drone_altitude':20}])
def test_view_rejects_invalid_or_aircraft_commands(update):
    with pytest.raises(ValueError):settings(DEFAULT,update)
    assert DEFAULT['distance_m']==7
