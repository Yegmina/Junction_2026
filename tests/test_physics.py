import json
from dataclasses import replace

import numpy as np
import pytest

from aerorozum.physics import PhysicsConfig, CoupledReference, export_fields, hydraulic_state, simulate, verify_numerics


def test_actual_numerical_verification():
    result=verify_numerics()
    assert result['status']=='passed', result
    assert result['research_accepted'] is False


def test_closed_water_redistribution_conserves_mass():
    c=PhysicsConfig(nx=3,ny=3,nz=8,targets=[],surface_exchange_enabled=False,gravity_enabled=False)
    solver=CoupledReference(c)
    solver.head[0]*=.5
    solver.theta,_,_=hydraulic_state(solver.head,solver.porosity,c)
    solver.initial_water_m3=float(solver.theta.sum()*solver.volume)
    solver.initial_energy_j=solver.energy_j()
    before=float(np.var(solver.theta))
    for _ in range(4):solver.step(30)
    assert solver.balances()['water_relative_residual']<1e-8
    assert float(np.var(solver.theta))<before


def test_export_schema_and_no_false_acceptance(tmp_path):
    c=PhysicsConfig(nx=4,ny=4,nz=6,duration_s=60,dt_s=20,output_interval_s=30,targets=[])
    result=export_fields(tmp_path/'field.npz',c)
    with np.load(result['field_path']) as data:
        assert data['temperature_k'].shape==(3,4,4)
        assert data['moisture'].shape==data['temperature_k'].shape
        assert np.all(np.diff(data['times_s'])>0)
    public=json.loads((tmp_path/'field.json').read_text())
    private=json.loads((tmp_path/'field.truth.json').read_text())
    assert public['research_accepted'] is False
    assert public['validation_state']=='unverified'
    assert 'targets' not in public
    assert private['targets']==[]


def test_evaporation_changes_water_and_temperature():
    c=PhysicsConfig(nx=3,ny=3,nz=6,targets=[],duration_s=300,dt_s=30,output_interval_s=300,air_diurnal_amplitude_k=0,sky_temperature_offset_k=0,solar_peak_w_m2=0)
    wet,r=simulate(c)
    dry,_=simulate(replace(c,evaporation_enabled=False))
    assert r['balances']['water_export_m3']>0
    assert wet['moisture'][-1].mean()<wet['moisture'][0].mean()
    assert wet['temperature_k'][-1].mean()<dry['temperature_k'][-1].mean()


def test_invalid_config():
    with pytest.raises(ValueError):CoupledReference(PhysicsConfig(nx=1))


def test_dumux_vtu_conversion_selects_highest_cell_layer(tmp_path):
    # Two hexahedra stacked vertically. Verify real VTK connectivity, top-layer
    # extraction, and volumetric moisture conversion rather than renaming arrays.
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('convert_dumux',Path(__file__).parents[1]/'physics/convert_dumux.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    points='0 0 0  1 0 0  0 1 0  1 1 0  0 0 1  1 0 1  0 1 1  1 1 1  0 0 2  1 0 2  0 1 2  1 1 2'
    xml=f'''<VTKFile><UnstructuredGrid><Piece NumberOfCells="2" NumberOfPoints="12"><CellData>
    <DataArray Name="T" format="ascii">280 300</DataArray><DataArray Name="S_liq" format="ascii">0.5 0.75</DataArray><DataArray Name="porosity" format="ascii">0.4 0.4</DataArray>
    </CellData><Points><DataArray format="ascii">{points}</DataArray></Points><Cells><DataArray Name="connectivity" format="ascii">0 1 2 3 4 5 6 7 4 5 6 7 8 9 10 11</DataArray><DataArray Name="offsets" format="ascii">8 16</DataArray></Cells></Piece></UnstructuredGrid></VTKFile>'''
    path=tmp_path/'field.vtu';path.write_text(xml)
    x,y,t,m=mod.surface(path)
    assert t.shape==(1,1) and t[0,0]==300
    assert m[0,0]==pytest.approx(.3)


def test_graded_mesh_refinement_and_domain_preserve_physical_nodes():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('graded_mesh',Path(__file__).parents[1]/'physics/graded_mesh.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    base={'Grid.UpperRight':'1 1 .4',**mod.make_mesh()}
    fine={'Grid.UpperRight':'1 1 .4',**mod.make_mesh(refinement=2)}
    larger={'Grid.UpperRight':'2 2 .8',**mod.make_mesh(extent=(2,2,.8),centre=(1,1))}
    for axis,shift in enumerate([.5,.5,.4]):
        a=mod.axis_nodes(base,axis);b=mod.axis_nodes(fine,axis);c=mod.axis_nodes(larger,axis)
        assert len(b)==2*(len(a)-1)+1
        assert np.allclose(b[::2],a)
        assert all(np.min(np.abs(c-(v+shift)))<1e-10 for v in a)


def test_compact_export_preserves_every_solved_surface_state(tmp_path):
    from physics.convert_compact import convert_compact
    (tmp_path/'surface-format.json').write_text(json.dumps(dict(points=4,scalar='float64',byte_order='little')))
    (tmp_path/'surface-grid.csv').write_text('x_m,y_m\n.25,.25\n.75,.25\n.25,.75\n.75,.75\n')
    records=np.array([[0,280,281,282,283,.2,.3,.4,.5],[1,284,285,286,287,.21,.31,.41,.51],[3,288,289,290,291,.22,.32,.42,.52]],dtype='<f8')
    records.tofile(tmp_path/'surface-frames.bin')
    (tmp_path/'conservation.csv').write_text('time_s,water_kg\n1,1\n3,1\n')
    result=convert_compact(tmp_path,tmp_path/'field.npz')
    with np.load(result['field_path']) as data:
        assert np.array_equal(data['times_s'],records[:,0])
        assert np.array_equal(data['temperature_k'].reshape(3,4),records[:,1:5].astype(np.float32))
    (tmp_path/'conservation.csv').write_text('time_s,water_kg\n1,1\n')
    with pytest.raises(ValueError,match='every solved ledger timestep'):convert_compact(tmp_path,tmp_path/'bad.npz')


def test_convergence_requires_absolute_and_relative_physics_accuracy(tmp_path):
    from physics.verify_dumux_cases import compare
    axes=dict(times_s=np.array([0.,1.,10.]),x_m=np.array([.25,.75]),y_m=np.array([.25,.75]))
    temperature=np.broadcast_to(np.array([300.,300.01,300.1])[:,None,None],(3,2,2)).copy()
    moisture=np.full((3,2,2),.2)
    reference=tmp_path/'reference.npz';candidate=tmp_path/'candidate.npz'
    np.savez(reference,**axes,temperature_k=temperature,moisture=moisture)
    np.savez(candidate,**axes,temperature_k=temperature+.015,moisture=moisture)
    result=compare(reference,candidate)
    assert result['max_temperature_error_k']<.02
    assert result['temperature_relative_error']>.02 and result['status']=='failed'
    np.savez(candidate,**axes,temperature_k=temperature,moisture=moisture+.005)
    result=compare(reference,candidate)
    assert result['moisture_relative_error']==pytest.approx(.025)
    assert result['status']=='failed'
    np.savez(candidate,**axes,temperature_k=temperature+.0001,moisture=moisture+.0001)
    assert compare(reference,candidate)['status']=='passed'


def test_axisymmetric_mesh_refines_and_preserves_domain_nodes():
    from physics.run_axisymmetric import mesh
    from physics.graded_mesh import axis_nodes
    base=mesh();fine=mesh(refinement=2);domain=mesh(radius=2,depth=1.6)
    for axis,shift in [(0,0.),(1,.8)]:
        a=axis_nodes(base,axis);b=axis_nodes(fine,axis);d=axis_nodes(domain,axis)
        assert np.allclose(b[::2],a)
        assert all(np.min(np.abs(d-(value+shift)))<1e-10 for value in a)
    assert np.min(np.abs(axis_nodes(base,0)-.12))<1e-12


def test_radial_cartesian_projection_stays_inside_proven_bound():
    from physics.radial_projection import plan,frame
    from scipy.interpolate import RegularGridInterpolator
    r=np.linspace(.0025,.9,180)
    temperature=289+np.array([.7,2.5])[:,None]*np.exp(-r[None,:]**2/.02)
    moisture=.3-.01*np.exp(-r[None,:]**2/.02)*np.array([.3,1.])[:,None]
    n,tp,mp,certificate=plan(r,temperature,moisture)
    x=(np.arange(n)+.5)/n;xx,yy=np.meshgrid(x,x);query=np.hypot(xx-.5,yy-.5)
    pixels=frame(tp,1,query).astype('float32');rng=np.random.default_rng(17)
    points=rng.uniform(x[0],x[-1],size=(10000,2));native=frame(tp,1,np.hypot(points[:,0]-.5,points[:,1]-.5))
    projected=RegularGridInterpolator((x,x),pixels)(points)
    assert np.max(np.abs(projected-native))<=certificate['temperature_error_bound_k']
    assert certificate['temperature_error_bound_k']<=.002
    assert certificate['moisture_error_bound']<=1e-4
