"""Lossless surface extraction with bounded intermediate volume retention.

Every required surface sample is retained. Only already-processed reproducible
volume intermediates are pruned; first/latest/hourly volume checkpoints remain.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    from .convert_dumux import surface,write_exchange
except ImportError:
    from convert_dumux import surface,write_exchange


class SurfaceCollector:
    def __init__(self,root,prune=False):
        self.root=Path(root);self.prune=prune;self.done=set();self.times=[];self.temperatures=[];self.moistures=[];self.hashes={};self.keep=set();self.previous=None;self.next_hour=0.;self.representation=None

    def drain(self):
        pvd=self.root/'campaign.pvd'
        if not pvd.exists():return
        try:entries=ET.parse(pvd).findall('.//DataSet')
        except ET.ParseError:return
        for entry in entries:
            name=entry.get('file')
            if name in self.done:continue
            path=self.root/name
            if path.suffix!='.vtu' or not path.resolve().is_relative_to(self.root.resolve()):raise ValueError('Sequential contained VTU outputs required')
            if not path.exists():break
            try:
                xx,yy,t,m=surface(path)
                representation='reconstructed_boundary_face' if ET.parse(path).find('.//CellData/DataArray[@Name="surface_temperature"]') is not None else 'top_cell_centre'
            except ET.ParseError:break
            if self.times and (not np.array_equal(xx,self.x) or not np.array_equal(yy,self.y) or representation!=self.representation):raise ValueError('Surface grid or representation changed')
            self.x,self.y=xx,yy;self.representation=representation
            at=float(entry.get('timestep'))
            self.times.append(at);self.temperatures.append(t);self.moistures.append(m);self.hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();self.done.add(name)
            if at>=self.next_hour:
                self.keep.add(name);self.next_hour=(int(at//3600)+1)*3600
            if self.prune and self.previous and self.previous not in self.keep:
                (self.root/self.previous).unlink(missing_ok=True)
            self.previous=name
            if len(self.times)%60==0:self.checkpoint()

    def checkpoint(self):
        if not self.times:return
        temporary=self.root/'surface-partial-writing.npz'
        result=write_exchange(temporary,self.times,self.x,self.y,self.temperatures,self.moistures,self.hashes,self.representation,'campaign.pvd')
        temporary.replace(self.root/'surface-partial.npz')
        temporary.with_suffix('.json').replace(self.root/'surface-partial.json')
        return result

    def finish(self,truth):
        self.drain()
        entries=ET.parse(self.root/'campaign.pvd').findall('.//DataSet')
        if len(entries)!=len(self.times):raise ValueError('Some solver output frames were not extracted')
        result=write_exchange(self.root/'field.npz',self.times,self.x,self.y,self.temperatures,self.moistures,self.hashes,self.representation,'campaign.pvd',truth)
        metadata=json.loads((self.root/'field.json').read_text())
        metadata['volume_retention']=dict(mode='first_latest_hourly' if self.prune else 'all',all_surface_samples_retained=True,source_volume_hashes_retained=True,kept_checkpoints=sorted(self.keep|{self.previous}))
        (self.root/'field.json').write_text(json.dumps(metadata,indent=2))
        return result
