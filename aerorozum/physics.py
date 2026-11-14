"""Conservative coupled heat/water reference calculation and field exchange.

This is an experimental Python finite-volume implementation, NOT DuMux.  The
separate ``physics/bootstrap_dumux.py`` builds the pinned upstream reference.
No output from this module is automatically accepted as research ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

SIGMA = 5.670374419e-8
RHO_W = 1000.0
CP_W = 4180.0
T_REF = 273.15


@dataclass
class PhysicsConfig:
    """SI units; array order is depth, north/y, east/x."""

    nx: int = 20
    ny: int = 20
    nz: int = 16
    width_m: float = 1.0
    height_m: float = 1.0
    depth_m: float = 0.4
    duration_s: float = 86400.0
    dt_s: float = 120.0
    output_interval_s: float = 1800.0
    initial_temperature_k: float = 289.15
    initial_saturation: float = 0.6
    porosity: float = 0.43
    residual_saturation: float = 0.08
    saturated_conductivity_m_s: float = 2e-6
    van_genuchten_alpha_m_inv: float = 3.6
    van_genuchten_n: float = 1.56
    dry_conductivity_w_m_k: float = 0.25
    wet_conductivity_w_m_k: float = 1.6
    solid_volumetric_capacity_j_m3_k: float = 2.0e6
    air_temperature_k: float = 289.15
    air_diurnal_amplitude_k: float = 5.0
    relative_humidity: float = 0.55
    wind_m_s: float = 2.0
    solar_peak_w_m2: float = 650.0
    albedo: float = 0.22
    emissivity: float = 0.95
    sky_temperature_offset_k: float = 15.0
    start_hour: float = 6.0
    evaporation_enabled: bool = True
    gravity_enabled: bool = True
    surface_exchange_enabled: bool = True
    targets: list[dict] = field(default_factory=lambda: [dict(id="target-001", x_m=0.5, y_m=0.5, radius_m=0.12, depth_m=0.025, thickness_m=0.05, material="inert_polymer_surrogate", conductivity_w_m_k=0.2, capacity_j_m3_k=1.7e6)])

    def validate(self):
        if min(self.nx, self.ny, self.nz) < 2:
            raise ValueError("Each grid axis needs at least two cells")
        if min(self.width_m, self.height_m, self.depth_m, self.duration_s, self.dt_s, self.output_interval_s) <= 0:
            raise ValueError("Domain lengths, duration, timestep and output interval must be positive")
        if not 0 < self.porosity < 1 or not 0 < self.initial_saturation < 1:
            raise ValueError("Porosity and effective initial saturation must be inside (0, 1)")
        if not 0 <= self.relative_humidity <= 1 or self.van_genuchten_n <= 1:
            raise ValueError("Invalid humidity or van Genuchten n")
        for target in self.targets:
            if target.get("radius_m", 0) <= 0 or target.get("depth_m", -1) < 0:
                raise ValueError("Invalid target geometry")


def hydraulic_state(head_m, porosity, config: PhysicsConfig):
    """van Genuchten/Mualem retention, capacity and hydraulic conductivity."""
    h = np.minimum(np.asarray(head_m), -1e-8)
    n = config.van_genuchten_n
    m = 1.0 - 1.0 / n
    alpha = config.van_genuchten_alpha_m_inv
    a = np.maximum(-alpha * h, 1e-12)
    se = (1 + a**n) ** -m
    tr = config.residual_saturation * porosity
    theta = tr + (porosity - tr) * se
    capacity = (porosity - tr) * m * n * alpha * a**(n - 1) * (1 + a**n)**(-m - 1)
    kr = np.sqrt(se) * (1 - (1 - se**(1 / m))**m)**2
    return theta, np.maximum(capacity, 1e-16), kr


def saturation_vapour_density(temperature_k):
    tc = np.asarray(temperature_k) - T_REF
    pressure = 610.94 * np.exp(17.625 * tc / (tc + 243.04))
    return pressure / (461.5 * np.asarray(temperature_k))


class CoupledReference:
    """3D closed-bottom Richards flow coupled to conservative sensible energy.

    No gas-phase vapour transport is solved inside the soil, unlike upstream
    ExtendedRichardsNI. Surface evaporation uses a humidity/air-resistance law.
    Thus this is a separately labelled numerical reference, not an equivalent
    replacement for the pinned DuMux model.
    """

    def __init__(self, config: PhysicsConfig):
        config.validate()
        self.config = config
        c = config
        self.shape = (c.nz, c.ny, c.nx)
        self.dx, self.dy, self.dz = c.width_m / c.nx, c.height_m / c.ny, c.depth_m / c.nz
        self.volume = self.dx * self.dy * self.dz
        self.area = self.dx * self.dy
        self.x = (np.arange(c.nx) + 0.5) * self.dx
        self.y = (np.arange(c.ny) + 0.5) * self.dy
        self.z = (np.arange(c.nz) + 0.5) * self.dz
        zz, yy, xx = np.meshgrid(self.z, self.y, self.x, indexing="ij")
        self.target_mask = np.zeros(self.shape, dtype=bool)
        self.porosity = np.full(self.shape, c.porosity)
        self.ksat = np.full(self.shape, c.saturated_conductivity_m_s)
        self.solid_capacity = np.full(self.shape, c.solid_volumetric_capacity_j_m3_k)
        self.target_conductivity = np.zeros(self.shape)
        for target in c.targets:
            mask = ((xx-target["x_m"])**2 + (yy-target["y_m"])**2 <= target["radius_m"]**2) & (zz >= target["depth_m"]) & (zz <= target["depth_m"] + target.get("thickness_m", .05))
            self.target_mask |= mask
            self.porosity[mask] = 1e-6
            self.ksat[mask] = 1e-24
            self.solid_capacity[mask] = target.get("capacity_j_m3_k", 1.7e6)
            self.target_conductivity[mask] = target.get("conductivity_w_m_k", .2)
        initial_head = -((c.initial_saturation**(-1/(1-1/c.van_genuchten_n))-1)**(1/c.van_genuchten_n)) / c.van_genuchten_alpha_m_inv
        self.head = np.full(self.shape, initial_head)
        self.theta, _, _ = hydraulic_state(self.head, self.porosity, c)
        self.temperature = np.full(self.shape, c.initial_temperature_k)
        ids = np.arange(np.prod(self.shape)).reshape(self.shape)
        self.connections = []
        for axis, spacing in enumerate((self.dz, self.dy, self.dx)):
            left = [slice(None)]*3; right = left.copy()
            left[axis] = slice(None, -1); right[axis] = slice(1, None)
            self.connections.append((ids[tuple(left)].ravel(), ids[tuple(right)].ravel(), spacing, axis))
        self.top = ids[0].ravel()
        self.elapsed_s = 0.0
        self.water_export_m3 = 0.0
        self.external_energy_j = 0.0
        self.initial_water_m3 = float(self.theta.sum() * self.volume)
        self.initial_energy_j = self.energy_j()
        self.steps = 0
        self.max_water_local_residual = 0.0
        self.max_energy_local_residual = 0.0

    def capacity(self, theta=None):
        theta = self.theta if theta is None else theta
        return (1-self.porosity)*self.solid_capacity + theta*RHO_W*CP_W + (self.porosity-theta)*1200.0

    def energy_j(self):
        return float(np.sum(self.capacity() * (self.temperature-T_REF))*self.volume)

    def _laplacian(self, coefficient):
        a = np.asarray(coefficient).ravel()
        rows, cols, values = [], [], []
        for i, j, spacing, axis in self.connections:
            f = 2*a[i]*a[j]/np.maximum(a[i]+a[j], 1e-100) / spacing**2
            rows.extend((i, j, i, j)); cols.extend((i, j, j, i)); values.extend((f, f, -f, -f))
        return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(a.size, a.size)).tocsr()

    def weather(self, at_s):
        c = self.config
        hour = (c.start_hour + at_s/3600) % 24
        solar = c.solar_peak_w_m2 * max(0., math.sin(math.pi*(hour-6)/12))
        air = c.air_temperature_k + c.air_diurnal_amplitude_k * math.sin(2*math.pi*(hour-9)/24)
        return solar, air, air-c.sky_temperature_offset_k

    def _evaporation(self, air_k, dt):
        c = self.config
        if not (c.evaporation_enabled and c.surface_exchange_enabled):
            return np.zeros((c.ny, c.nx))
        # Kelvin water activity at the pore surface; empirical aerodynamic resistance.
        activity = np.exp(np.clip(9.80665*self.head[0]/(461.5*self.temperature[0]), -100, 0))
        resistance_s_m = 200.0 / (1 + .5*c.wind_m_s)
        evap = (activity*saturation_vapour_density(self.temperature[0])-c.relative_humidity*saturation_vapour_density(air_k)) / (RHO_W*resistance_s_m)
        tr = c.residual_saturation*self.porosity[0]
        upper = np.maximum(0, (self.theta[0]-tr)*self.dz*.25/dt)
        lower = -np.maximum(0, (self.porosity[0]-self.theta[0])*self.dz*.25/dt)
        return np.clip(evap, lower, upper)

    def _water(self, dt, evap):
        c = self.config
        old_theta = self.theta.copy()
        h = self.head.copy()
        boundary = np.zeros(self.shape); boundary[0] = -evap/self.dz
        final_fluxes = []
        for iteration in range(60):
            theta, cap, kr = hydraulic_state(h, self.porosity, c)
            k = self.ksat*kr
            gravity = np.zeros(h.size)
            if c.gravity_enabled:
                i, j, spacing, _ = self.connections[0]
                a = k.ravel(); face_k = 2*a[i]*a[j]/np.maximum(a[i]+a[j], 1e-100)
                np.add.at(gravity, i, -face_k/spacing)
                np.add.at(gravity, j, face_k/spacing)
            lap = self._laplacian(k)
            diag = cap.ravel()/dt
            rhs = diag*h.ravel()-(theta-old_theta).ravel()/dt + gravity + boundary.ravel()
            new_h = spsolve(diags(diag)+lap, rhs).reshape(self.shape)
            if not np.isfinite(new_h).all() or np.max(new_h) >= 0:
                raise ArithmeticError("Richards iteration left unsaturated domain; reduce timestep")
            # Under-relax difficult dry cells; the final residual is still checked.
            if iteration < 3:
                new_h = .8*new_h+.2*h
            new_theta, _, _ = hydraulic_state(new_h, self.porosity, c)
            delta = float(np.max(np.abs(new_theta-theta)))
            h = new_h
            if delta < 1e-10:
                break
        else:
            raise ArithmeticError("Richards nonlinear solve did not converge")
        theta, _, kr = hydraulic_state(h, self.porosity, c)
        a = (self.ksat*kr).ravel()
        div = np.zeros(h.size)
        for i,j,spacing,axis in self.connections:
            face_k = 2*a[i]*a[j]/np.maximum(a[i]+a[j],1e-100)
            flux = face_k*((1. if axis==0 and c.gravity_enabled else 0.)-(h.ravel()[j]-h.ravel()[i])/spacing)
            np.add.at(div,i,flux/spacing);np.add.at(div,j,-flux/spacing)
            final_fluxes.append((i,j,flux,spacing))
        residual = (theta-old_theta).ravel()/dt + div-boundary.ravel()
        if np.max(np.abs(residual))*dt > 1e-8:
            raise ArithmeticError("Water conservation residual exceeded tolerance")
        self.max_water_local_residual=max(self.max_water_local_residual,float(np.max(np.abs(residual))*dt))
        return h,theta,final_fluxes

    def step(self, dt):
        c = self.config
        solar, air, sky = self.weather(self.elapsed_s + dt/2)
        evap = self._evaporation(air,dt)
        new_h, new_theta, fluxes = self._water(dt,evap)
        old_u = self.temperature-T_REF
        old_c = self.capacity()
        new_c = self.capacity(new_theta)
        se = (new_theta-c.residual_saturation*self.porosity)/((1-c.residual_saturation)*self.porosity)
        conductivity = c.dry_conductivity_w_m_k+(c.wet_conductivity_w_m_k-c.dry_conductivity_w_m_k)*np.sqrt(np.clip(se,0,1))
        conductivity[self.target_mask] = self.target_conductivity[self.target_mask]
        lap = self._laplacian(conductivity)
        # Conservative, upwind sensible enthalpy carried by liquid water.
        adv = np.zeros(old_u.size)
        for i,j,flux,spacing in fluxes:
            energy_flux = RHO_W*CP_W*flux*np.where(flux>=0,old_u.ravel()[i],old_u.ravel()[j])
            np.add.at(adv,i,-energy_flux/spacing);np.add.at(adv,j,energy_flux/spacing)
        guess=self.temperature.copy()
        for _ in range(20):
            ts=guess[0]
            extra=np.zeros(self.shape);rhs=(old_c*old_u/dt).ravel()+adv
            if c.surface_exchange_enabled:
                hconv=5.7+3.8*c.wind_m_s
                latent=2.501e6-2370*(ts-T_REF)
                q=(1-c.albedo)*solar+c.emissivity*SIGMA*(sky**4-ts**4)+hconv*(air-ts)-RHO_W*evap*(latent+CP_W*(ts-T_REF))
                dq=c.emissivity*4*SIGMA*ts**3+hconv+RHO_W*evap*(CP_W-2370)
                extra[0]=dq/self.dz
                rhs[self.top]+=(q+dq*(ts-T_REF)).ravel()/self.dz
            solved=spsolve(diags((new_c/dt+extra).ravel())+lap,rhs).reshape(self.shape)+T_REF
            if not np.isfinite(solved).all() or np.min(solved)<200 or np.max(solved)>400:
                raise ArithmeticError("Heat solution left supported temperature range")
            if np.max(np.abs(solved-guess))<1e-8:
                guess=solved;break
            guess=solved
        else:
            raise ArithmeticError("Surface energy iteration did not converge")
        q=np.zeros((c.ny,c.nx))
        if c.surface_exchange_enabled:
            ts=guess[0];latent=2.501e6-2370*(ts-T_REF)
            q=(1-c.albedo)*solar+c.emissivity*SIGMA*(sky**4-ts**4)+(5.7+3.8*c.wind_m_s)*(air-ts)-RHO_W*evap*(latent+CP_W*(ts-T_REF))
        energy_before=self.energy_j()
        self.head=new_h;self.theta=new_theta;self.temperature=guess
        external=float(q.sum()*self.area*dt)
        residual=self.energy_j()-energy_before-external
        self.max_energy_local_residual=max(self.max_energy_local_residual,abs(residual))
        self.external_energy_j+=external
        self.water_export_m3+=float(evap.sum()*self.area*dt)
        self.elapsed_s+=dt;self.steps+=1

    def balances(self):
        water=float(self.theta.sum()*self.volume)
        water_error=water-self.initial_water_m3+self.water_export_m3
        energy_error=self.energy_j()-self.initial_energy_j-self.external_energy_j
        return dict(water_initial_m3=self.initial_water_m3,water_final_m3=water,water_export_m3=self.water_export_m3,water_residual_m3=water_error,water_relative_residual=abs(water_error)/max(self.initial_water_m3,1e-15),energy_initial_j=self.initial_energy_j,energy_final_j=self.energy_j(),external_energy_j=self.external_energy_j,energy_residual_j=energy_error,energy_relative_residual=abs(energy_error)/max(abs(self.external_energy_j),abs(self.initial_energy_j),1),max_local_water_residual=self.max_water_local_residual,max_step_energy_residual_j=self.max_energy_local_residual)


def simulate(config: PhysicsConfig | dict | None = None, progress=None):
    c=config if isinstance(config,PhysicsConfig) else PhysicsConfig(**(config or {}))
    solver=CoupledReference(c)
    times=[0.];temperatures=[solver.temperature[0].copy()];moistures=[solver.theta[0].copy()]
    next_output=min(c.output_interval_s,c.duration_s)
    started=time.monotonic()
    while solver.elapsed_s < c.duration_s-1e-8:
        dt=min(c.dt_s,c.duration_s-solver.elapsed_s,next_output-solver.elapsed_s)
        while True:
            try:solver.step(dt);break
            except ArithmeticError:
                dt/=2
                if dt<.01:raise
        if solver.elapsed_s >= next_output-1e-8:
            times.append(solver.elapsed_s);temperatures.append(solver.temperature[0].copy());moistures.append(solver.theta[0].copy())
            if progress:progress(dict(time_s=solver.elapsed_s,steps=solver.steps,elapsed_wall_s=time.monotonic()-started))
            next_output=min(next_output+c.output_interval_s,c.duration_s)
    arrays=dict(times_s=np.asarray(times),x_m=solver.x,y_m=solver.y,temperature_k=np.stack(temperatures).astype(np.float32),moisture=np.stack(moistures).astype(np.float32))
    return arrays,dict(balances=solver.balances(),steps=solver.steps,elapsed_wall_s=time.monotonic()-started,temperature_range_k=[float(solver.temperature.min()),float(solver.temperature.max())],moisture_range=[float(solver.theta.min()),float(solver.theta.max())],target_cells=int(solver.target_mask.sum()))


def export_fields(output: str | Path, config: PhysicsConfig | dict | None = None, progress=None):
    c=config if isinstance(config,PhysicsConfig) else PhysicsConfig(**(config or {}))
    arrays,results=simulate(c,progress)
    output=Path(output).with_suffix('.npz');output.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest()
    with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.npz',delete=False) as temporary:
        temp_path=Path(temporary.name)
        try:np.savez_compressed(temporary,**arrays)
        except BaseException:
            temp_path.unlink(missing_ok=True);raise
    temp_path.replace(output)
    metadata=dict(fidelity='experimental_coupled_finite_volume',backend='python_fv_richards_heat',solver_version='0.1.0',config_hash=digest,validation_state='unverified',research_accepted=False,units=dict(times_s='s',x_m='m east',y_m='m north',temperature_k='K',moisture='m3 water / m3 bulk'),domain=dict(width_m=c.width_m,height_m=c.height_m,depth_m=c.depth_m),results=results,limitations=['Python reference backend is not DuMux and has no subsurface vapour-diffusion equation.','Surface array is top-cell centre temperature, not a separately reconstructed face temperature.','Fixed reference atmosphere; no rain, root uptake, freeze/thaw or air-flow solve.','Target is regularized near-impermeable solid; regularization sensitivity not yet accepted.','No grid/time/domain convergence or field validation has been inferred from conservation alone.'])
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    output.with_suffix('.truth.json').write_text(json.dumps(dict(targets=c.targets,input_parameters=asdict(c),config_hash=digest),indent=2))
    return dict(field_path=str(output),metadata_path=str(output.with_suffix('.json')),truth_path=str(output.with_suffix('.truth.json')),**metadata)


def export_pair(output_dir: str | Path, config: PhysicsConfig | dict | None = None, progress=None):
    """Run matched target-present/absent cases, keeping evaluation truth private."""
    from dataclasses import replace
    c=config if isinstance(config,PhysicsConfig) else PhysicsConfig(**(config or {}))
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    present=export_fields(root/'present.npz',c,progress)
    absent=export_fields(root/'absent.npz',replace(c,targets=[]),progress)
    with np.load(present['field_path']) as p,np.load(absent['field_path']) as a:
        difference=p['temperature_k'].astype(float)-a['temperature_k'].astype(float)
        result=dict(backend='python_fv_richards_heat',validation_state='unverified',research_accepted=False,present=present['field_path'],absent=absent['field_path'],max_absolute_temperature_difference_k=float(np.abs(difference).max()),max_contrast_time_s=float(p['times_s'][np.argmax(np.abs(difference).reshape(len(difference),-1).max(axis=1))]))
    (root/'pair.truth.json').write_text(json.dumps(result,indent=2))
    return result


def verify_numerics():
    """Actual small numerical checks, intentionally not a research acceptance gate."""
    tests=[]
    c=PhysicsConfig(nx=4,ny=4,nz=6,duration_s=120,dt_s=30,output_interval_s=120,targets=[],surface_exchange_enabled=False,gravity_enabled=False)
    a,r=simulate(c)
    uniform_error=float(np.max(np.abs(a['temperature_k']-c.initial_temperature_k)))
    tests.append(dict(name='closed_uniform_equilibrium',passed=uniform_error<2e-5,temperature_error_k=uniform_error,**r['balances']))
    c2=PhysicsConfig(nx=4,ny=4,nz=8,duration_s=600,dt_s=30,output_interval_s=600,targets=[],air_diurnal_amplitude_k=0,solar_peak_w_m2=0,sky_temperature_offset_k=0)
    a,r=simulate(c2)
    b=r['balances']
    tests.append(dict(name='coupled_evaporation_mass_energy',passed=b['water_export_m3']>0 and b['water_relative_residual']<1e-7 and b['energy_relative_residual']<1e-7,**b))
    c3=PhysicsConfig(nx=4,ny=4,nz=8,duration_s=600,dt_s=30,output_interval_s=600,targets=[],surface_exchange_enabled=False,gravity_enabled=False)
    s=CoupledReference(c3);s.temperature[0]+=5;s.initial_energy_j=s.energy_j();before=float(np.var(s.temperature))
    for _ in range(20):s.step(30)
    b=s.balances();tests.append(dict(name='closed_heat_diffusion_energy',passed=b['energy_relative_residual']<1e-8 and float(np.var(s.temperature))<before,initial_variance=before,final_variance=float(np.var(s.temperature)),**b))
    # Exact Neumann-boundary cosine mode of homogeneous heat conduction.
    c4=PhysicsConfig(nx=2,ny=2,nz=32,duration_s=3600,dt_s=30,targets=[],surface_exchange_enabled=False,gravity_enabled=False)
    s=CoupledReference(c4)
    mode=np.cos(np.pi*s.z/c4.depth_m)[:,None,None]
    s.temperature[:]=c4.initial_temperature_k+4*mode
    s.initial_energy_j=s.energy_j()
    thermal_k=c4.dry_conductivity_w_m_k+(c4.wet_conductivity_w_m_k-c4.dry_conductivity_w_m_k)*np.sqrt(c4.initial_saturation)
    alpha=thermal_k/float(s.capacity()[0,0,0])
    for _ in range(120):s.step(30)
    exact=c4.initial_temperature_k+4*np.exp(-alpha*(np.pi/c4.depth_m)**2*3600)*mode
    error=float(np.max(np.abs(s.temperature-exact)))
    tests.append(dict(name='analytical_neumann_heat_mode',passed=error<.01,max_temperature_error_k=error,diffusivity_m2_s=alpha))
    # A hydrostatic head gradient cancels gravity exactly: zero net water flux.
    c5=PhysicsConfig(nx=2,ny=2,nz=12,targets=[],surface_exchange_enabled=False)
    s=CoupledReference(c5);s.head[:]=s.z[:,None,None]-3
    s.theta,_,_=hydraulic_state(s.head,s.porosity,c5)
    before=s.theta.copy()
    for _ in range(10):s.step(30)
    error=float(np.max(np.abs(s.theta-before)))
    tests.append(dict(name='richards_hydrostatic_equilibrium',passed=error<1e-10,max_moisture_error=error))
    return dict(status='passed' if all(t['passed'] for t in tests) else 'failed',backend='python_fv_richards_heat',research_accepted=False,tests=tests)


def staged_validation_report(upstream_root: str | Path | None = None):
    """Compute acceptance from actual evidence; missing gates remain pending."""
    evidence=verify_numerics()
    if upstream_root is None:
        upstream_root=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/physics/dumux')
    upstream_root=Path(upstream_root)
    upstream=json.loads((upstream_root/'status.json').read_text()) if (upstream_root/'status.json').exists() else {}
    checks=[dict(name='pinned_dumux_upstream_regression',status='passed' if upstream.get('status')=='upstream_test_passed' else 'pending',evidence=str(upstream_root/'status.json'))]
    checks.extend(dict(name=t['name'],status='passed' if t['passed'] else 'failed',backend=evidence['backend'],result=t) for t in evidence['tests'])
    analytical_path=upstream_root/'analytical-status.json'
    analytical=json.loads(analytical_path.read_text()) if analytical_path.exists() else {}
    checks.append(dict(name='dumux_analytical_heat_water',status='passed' if analytical.get('status')=='upstream_analytical_regressions_passed' else 'pending',evidence=str(analytical_path),result=analytical))
    for name in ['custom_dumux_mass_energy_balance','heterogeneous_interface_flux_continuity','grid_time_domain_convergence','target_impermeability_regularization']:
        checks.append(dict(name=name,status='pending',reason='No passing result has been recorded by an implemented verifier'))
    comparisons=upstream_root.parent.parent/'artifacts/physics/surface-case-comparison-verification.json'
    if not comparisons.exists():comparisons=upstream_root.parent.parent/'artifacts/physics/case-comparison-verification.json'
    screens=json.loads(comparisons.read_text()) if comparisons.exists() else None
    accepted=all(t['status']=='passed' for t in checks)
    return dict(status='numerically_accepted' if accepted else 'partial_verification',solver='DuMux 3.10.0 + separate Python finite-volume reference',research_accepted=accepted,numerical_admission=dict(accepted=accepted,checks=checks),checks=checks,upstream=upstream,custom_dumux_screens=screens,surrogate_admission=dict(accepted=False,status='pending',reason='Additional matched direct-field approximation error gate not yet passed; does not block independent direct-field numerical admission'),published_experiment_reproduction=dict(status='not_performed',informational=True),field_validation='not_performed',limitations=['Numerical acceptance is separate from physical field validation.','Passing upstream regression does not validate the custom atmospheric boundary or target geometry.','No numerical or field validation has been invented for missing checks.'])


def assess_dumux_ledger(path: str | Path, required_duration_s: float, tolerance=.005, target_required=True):
    """Check actual DuMux-integrated budgets, never a Python substitute."""
    path=Path(path)
    if not path.exists():return dict(status='pending',reason='DuMux conservation.csv missing',research_accepted=False)
    values=np.genfromtxt(path,delimiter=',',names=True)
    values=np.atleast_1d(values)
    names=set(values.dtype.names or ())
    required={'time_s','water_kg','water_relative_residual','energy_relative_residual','water_residual_kg','energy_residual_j','cumulative_water_out_kg','cumulative_energy_out_j','target_water_kg','cumulative_absolute_target_exchange_kg'}
    if not required<=names or not len(values):raise ValueError('Incomplete DuMux budget columns')
    if any(not np.isfinite(values[n]).all() for n in names):raise ValueError('Nonfinite DuMux budget')
    duration=float(values['time_s'][-1]);complete=abs(duration-required_duration_s)<1e-5*max(1,required_duration_s)
    # Independently normalize by cumulative absolute transported mass/energy,
    # not by the large absolute reference storage (which could hide errors).
    mass_transport=np.cumsum(np.abs(np.diff(np.r_[0.,values['cumulative_water_out_kg']])))
    energy_transport=np.cumsum(np.abs(np.diff(np.r_[0.,values['cumulative_energy_out_j']])))
    water=float(np.max(np.abs(values['water_residual_kg'])/np.maximum(mass_transport,1e-9)))
    energy=float(np.max(np.abs(values['energy_residual_j'])/np.maximum(energy_transport,1e-6)))
    inventory=float(np.max(values['target_water_kg']))/max(float(values['water_kg'][0]),1e-12)
    exchange=float(values['cumulative_absolute_target_exchange_kg'][-1])/max(float(values['water_kg'][0]),1e-12)
    checks=[dict(name='mass_energy_balance',status='passed' if complete and max(water,energy)<=tolerance else 'failed',water_relative_error=water,energy_relative_error=energy,tolerance=tolerance,normalization='Cumulative absolute boundary transport; absolute denominators 1e-9 kg and 1e-6 J only for roundoff near zero'),dict(name='target_fluid_storage_exchange',status='passed' if complete and max(inventory,exchange)<=1e-6 else 'failed',max_fluid_inventory_fraction=inventory,cumulative_exchange_fraction=exchange,tolerance=1e-6)]
    if {'interface_water_net_kg_s','interface_energy_net_w','interface_water_absolute_kg_s','interface_energy_absolute_w'}<=names:
        energy_flow=float(np.max(values['interface_energy_absolute_w']))
        # Mixed absolute/relative tolerances remain meaningful when interface
        # water transport is effectively zero (10^-17 kg/s in dry solid cells).
        er=float(np.max(np.abs(values['interface_energy_net_w'])/np.maximum(values['interface_energy_absolute_w'],1e-4)))
        wr=float(np.max(np.abs(values['interface_water_net_kg_s'])/np.maximum(values['interface_water_absolute_kg_s'],1e-8)))
        continuity_status=('not_applicable' if not target_required and energy_flow<1e-15 else ('passed' if complete and energy_flow>1e-10 and max(er,wr)<1e-8 else 'failed'))
        checks.append(dict(name='heterogeneous_interface_flux_continuity',status=continuity_status,max_energy_scaled_imbalance=er,max_water_scaled_imbalance=wr,max_absolute_energy_flow_w=energy_flow,relative_tolerance=1e-8,absolute_water_tolerance_kg_s=1e-16,absolute_energy_tolerance_w=1e-12))
    else:checks.append(dict(name='heterogeneous_interface_flux_continuity',status='pending',reason='This earlier instrumented binary did not log both sides of material interfaces'))
    return dict(status='passed' if all(c['status'] in ('passed','not_applicable') for c in checks) else 'partial_or_failed',backend='dumux_extended_richards_ni',ledger_path=str(path),duration_s=duration,required_duration_s=required_duration_s,checks=checks,research_accepted=False,scope='Only budget and interface checks for this executed trajectory; grid/time/domain convergence remains separate')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('generate');p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('pair');p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--output',type=Path)
    p=sub.add_parser('validate');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='verify':
        result=verify_numerics()
        if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1
    if args.command=='validate':
        result=staged_validation_report();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return 0
    config=json.loads(args.config.read_text()) if args.config else None
    generate=export_pair if args.command=='pair' else export_fields
    result=generate(args.output,config,lambda p:print(json.dumps(p),flush=True))
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())
