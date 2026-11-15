#!/usr/bin/env python3
"""Convert actual sequential ASCII DuMux TPFA VTU/PVD outputs to field NPZ.

Rejects unsupported binary/MPI/missing-data output rather than inventing values.
The exported surface is the highest cell-centre layer; no face reconstruction or
research acceptance is implied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def values(node,dtype=float):
    if node is None or node.get('format','ascii')!='ascii':raise ValueError('ASCII VTK data required')
    return np.fromstring(node.text or '',sep=' ',dtype=dtype)


def surface(path):
    piece=ET.parse(path).find('.//Piece')
    if piece is None:raise ValueError('Expected sequential VTU Piece')
    needed={'T','surface_temperature','S_liq','porosity'}
    fields={n.get('Name'):values(n) for n in piece.findall('./CellData/DataArray') if n.get('Name') in needed}
    points=values(piece.find('./Points/DataArray')).reshape(-1,3)
    if np.ptp(points[:,2]) <= 0:
        raise ValueError('A 2D vertical cross-section is not a 3D thermal surface')
    cells={n.get('Name'):values(n,np.int64) for n in piece.findall('./Cells/DataArray')}
    ids=cells['connectivity'];offsets=cells['offsets']
    begin=0;centres=[]
    for end in offsets:
        centres.append(points[ids[begin:end]].mean(axis=0));begin=end
    centres=np.round(np.asarray(centres),9)
    top=np.isclose(centres[:,2],centres[:,2].max(),atol=1e-8)
    selected=centres[top];x=np.unique(selected[:,0]);y=np.unique(selected[:,1])
    if len(x)*len(y)!=int(top.sum()):raise ValueError('Expected a complete Cartesian top cell layer')
    order=np.lexsort((selected[:,0],selected[:,1]))
    temperature=fields.get('surface_temperature',fields['T'])[top][order].reshape(len(y),len(x))
    moisture=(fields['S_liq']*fields['porosity'])[top][order].reshape(len(y),len(x))
    if not np.isfinite(temperature).all() or not np.isfinite(moisture).all():raise ValueError('Nonfinite solver output')
    return x,y,temperature,moisture


def write_exchange(output,times,x,y,temperatures,moistures,source_hashes,representation,source_pvd,truth=None):
    output=Path(output).with_suffix('.npz');output.parent.mkdir(parents=True,exist_ok=True)
    if np.any(np.diff(times)<=0):raise ValueError('Times must increase strictly')
    np.savez_compressed(output,times_s=np.asarray(times),x_m=x,y_m=y,temperature_k=np.stack(temperatures).astype(np.float32),moisture=np.stack(moistures).astype(np.float32))
    metadata=dict(fidelity='experimental_coupled_dumux',backend='dumux_extended_richards_ni',solver_version='3.10.0',temperature_representation=representation,validation_state='custom_application_unverified',research_accepted=False,source_pvd=str(source_pvd),source_hashes=source_hashes,units=dict(times_s='s',x_m='m east',y_m='m north',temperature_k='K',moisture='m3 water / m3 bulk'),limitations=['Custom 3D atmospheric boundary and target regularization not research accepted.',('Surface temperature uses a local half-cell conductive reconstruction.' if representation=='reconstructed_boundary_face' else 'Highest cell-centre layer, not reconstructed physical surface temperature.'),'Moisture represents the highest cell-centre layer.','Unmodified upstream regression status is tracked separately.','No field validation or parameter calibration inferred from solver execution.'])
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    if truth:
        data=json.loads(Path(truth).read_text())
        output.with_suffix('.truth.json').write_text(json.dumps(data,indent=2))
    return dict(field_path=str(output),frames=len(times),shape=[len(times),len(y),len(x)],**metadata)


def convert(pvd,output,truth=None):
    pvd=Path(pvd)
    entries=ET.parse(pvd).findall('.//DataSet')
    entries=sorted(entries,key=lambda e:float(e.get('timestep')))
    if not entries:raise ValueError('PVD has no results')
    times=[];temperatures=[];moistures=[];source_hashes={};representations=set()
    for e in entries:
        file=pvd.parent/e.get('file')
        if file.suffix!='.vtu':raise ValueError('Sequential .vtu files required, not MPI .pvtu')
        xx,yy,temp,moisture=surface(file)
        representations.add('reconstructed_boundary_face' if ET.parse(file).find('.//CellData/DataArray[@Name="surface_temperature"]') is not None else 'top_cell_centre')
        if temperatures and (not np.array_equal(x,xx) or not np.array_equal(y,yy)):raise ValueError('Grid changed between timesteps')
        x,y=xx,yy;times.append(float(e.get('timestep')));temperatures.append(temp);moistures.append(moisture)
        source_hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    if len(representations)!=1:raise ValueError('Mixed cell-centre and face-temperature outputs')
    return write_exchange(output,times,x,y,temperatures,moistures,source_hashes,next(iter(representations)),pvd.name,truth)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pvd',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--truth',type=Path)
    a=p.parse_args();print(json.dumps(convert(a.pvd,a.output,a.truth),indent=2))


if __name__=='__main__':main()
