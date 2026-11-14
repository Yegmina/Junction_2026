"""Read every compact DuMux surface state, without temporal/spatial thinning."""
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .convert_dumux import write_exchange
except ImportError:
    from convert_dumux import write_exchange


def convert_compact(root,output,truth=None):
    root=Path(root);description=json.loads((root/'surface-format.json').read_text())
    if description['scalar']!='float64':raise ValueError('Unsupported compact scalar format')
    coordinates=np.round(np.loadtxt(root/'surface-grid.csv',delimiter=',',skiprows=1).reshape(-1,2),9)
    count=int(description['points']);columns=1+2*count
    if len(coordinates)!=count:raise ValueError('Surface coordinate count mismatch')
    dtype='<f8' if description['byte_order']=='little' else '>f8'
    raw=np.fromfile(root/'surface-frames.bin',dtype=dtype)
    if len(raw)%columns:raise ValueError('Incomplete final surface frame')
    records=raw.reshape(-1,columns)
    if not len(records) or not np.isfinite(records).all():raise ValueError('Empty or nonfinite surface output')
    x=np.unique(coordinates[:,0]);y=np.unique(coordinates[:,1]);xx,yy=np.meshgrid(x,y)
    if count!=len(x)*len(y) or not np.allclose(coordinates,np.c_[xx.ravel(),yy.ravel()],rtol=0,atol=1e-12):raise ValueError('Expected Cartesian surfaces ordered north/east')
    temperatures=records[:,1:1+count].reshape(-1,len(y),len(x));moistures=records[:,1+count:].reshape(-1,len(y),len(x))
    ledger=np.atleast_1d(np.genfromtxt(root/'conservation.csv',delimiter=',',names=True))
    if len(ledger)!=len(records)-1 or not np.allclose(ledger['time_s'],records[1:,0],rtol=0,atol=1e-8):raise ValueError('Surface snapshots do not cover every solved ledger timestep')
    volumes=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.vtu','.pvtu','.pvd') and p.resolve().is_relative_to(root.resolve()))
    files=[root/'surface-frames.bin',root/'surface-grid.csv',root/'surface-format.json',*volumes]
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result=write_exchange(output,records[:,0],x,y,temperatures,moistures,hashes,'reconstructed_boundary_face','surface-frames.bin',truth)
    meta_path=Path(output).with_suffix('.json');metadata=json.loads(meta_path.read_text())
    metadata.update(surface_serialization='native float64, exported float32 NPZ',surface_mpi_ranks=description.get('mpi_ranks',1),all_solved_surface_states_retained=True,volume_retention=dict(mode='first_last_periodic',kept_checkpoints=[str(p.relative_to(root)) for p in volumes],all_surface_samples_retained=True))
    meta_path.write_text(json.dumps(metadata,indent=2));return result
