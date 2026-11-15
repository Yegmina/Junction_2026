#!/usr/bin/env python3
"""Evaluate actual custom DuMux refinement runs without auto-accepting gaps."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def compare(reference,candidate,tolerance_k=.02,coordinate_shift_m=0.,relative_tolerance=.02):
    reference=Path(reference);candidate=Path(candidate)
    if not reference.exists() or not candidate.exists():return dict(status='pending',reason='Required solver field not completed')
    metadata=[]
    for path in [reference,candidate]:
        sidecar=path.with_suffix('.json')
        metadata.append(json.loads(sidecar.read_text()) if sidecar.exists() else {})
    representations=[m.get('temperature_representation','top_cell_centre') for m in metadata]
    if representations[0]!=representations[1]:raise ValueError('Cannot compare cell-centre and boundary-face temperatures as matching physical observables')
    with np.load(reference) as a,np.load(candidate) as b:
        times=a['times_s'];y=a['y_m'];x=a['x_m']
        if times[0]<b['times_s'][0] or times[-1]>b['times_s'][-1]:raise ValueError('Refinement case does not cover reference time range')
        tt,yy,xx=np.meshgrid(times,y,x,indexing='ij');points=np.stack((tt,yy,xx),axis=-1)
        ti=RegularGridInterpolator((b['times_s'],b['y_m']-coordinate_shift_m,b['x_m']-coordinate_shift_m),b['temperature_k'],bounds_error=True)(points)
        mi=RegularGridInterpolator((b['times_s'],b['y_m']-coordinate_shift_m,b['x_m']-coordinate_shift_m),b['moisture'],bounds_error=True)(points)
        error=float(np.max(np.abs(ti-a['temperature_k'])))
        error_at=np.abs(ti-a['temperature_k']);where=np.unravel_index(np.argmax(error_at),error_at.shape)
        moisture_error=float(np.max(np.abs(mi-a['moisture'])))
        # Time-weighted norms avoid overweighting the tiny adaptive start-up
        # steps. Normalize temperature against its physical temporal excursion,
        # never against the arbitrary ~300 K absolute temperature offset.
        if len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('At least two strictly increasing physical times required')
        intervals=np.diff(times);weights=np.zeros_like(times);weights[:-1]+=intervals/2;weights[1:]+=intervals/2
        weights=np.sqrt(weights)[:,None,None]
        temperature_norm=float(np.linalg.norm(weights*(a['temperature_k']-a['temperature_k'][0])))
        temperature_error_norm=float(np.linalg.norm(weights*(ti-a['temperature_k'])))
        moisture_norm=float(np.linalg.norm(weights*a['moisture']))
        moisture_error_norm=float(np.linalg.norm(weights*(mi-a['moisture'])))
        temperature_relative=temperature_error_norm/max(temperature_norm,1e-12)
        moisture_relative=moisture_error_norm/max(moisture_norm,1e-12)
        passed=error<=tolerance_k and temperature_relative<=relative_tolerance and moisture_relative<=relative_tolerance
        return dict(status='passed' if passed else 'failed',reference=str(reference),candidate=str(candidate),max_temperature_error_k=error,max_error_time_s=float(times[where[0]]),max_error_position_m=[float(x[where[2]]),float(y[where[1]])],initial_temperature_error_k=float(error_at[0].max()),final_temperature_error_k=float(error_at[-1].max()),max_moisture_difference=moisture_error,temperature_relative_error=temperature_relative,moisture_relative_error=moisture_relative,relative_tolerance=relative_tolerance,relative_norm_definition='Trapezoid-time-weighted L2; temperature denominator is reference change from its initial surface field, moisture denominator is reference water volume fraction',temperature_tolerance_k=tolerance_k,temperature_representation=representations[0],comparison='Candidate interpolated at reference horizontal cell centres and times; no extrapolation',limitation='Representative-case screening is separate from full-day and parameter-range convergence certification')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/artifacts/physics'))
    p.add_argument('--output',type=Path)
    p.add_argument('--surface',action='store_true',help='Check corrected surface/component-enthalpy formulation, never mix with legacy fields')
    a=p.parse_args();root=a.data_root
    baseline=root/'dumux-surface-base/field.npz' if a.surface else root/'dumux-smoke-field.npz'
    mapping={'time-refinement':'surface-time','grid-refinement':'surface-grid','target-regularization':'surface-regularization'} if a.surface else {k:k for k in ['time-refinement','grid-refinement','target-regularization']}
    cases={name:compare(baseline,root/f'dumux-{directory}'/'field.npz') for name,directory in mapping.items()}
    domain='dumux-surface-domain' if a.surface else 'dumux-domain-refinement'
    cases['domain-refinement']=compare(baseline,root/domain/'field.npz',coordinate_shift_m=.5)
    result=dict(status='partial_verification',solver='DuMux',solver_version='3.10.0',formulation_version='surface_component_enthalpy_v2' if a.surface else 'legacy_cell_mixture_v1',research_accepted=False,cases=cases,required_pending=['custom_mass_energy_balance','heterogeneous_interface_flux','full_day_convergence'],published_experiment_reproduction=dict(status='not_performed',informational=True),field_validation='not_performed',surrogate_admission=dict(accepted=False,status='pending',separate_from_direct_admission=True),scope='Actual one-hour coarse smoke cases only; full-day convergence remains required')
    out=a.output or root/('surface-case-comparison-verification.json' if a.surface else 'case-comparison-verification.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    from aerorozum.physics import staged_validation_report
    (root/'verification.json').write_text(json.dumps(staged_validation_report(),indent=2))


if __name__=='__main__':main()
