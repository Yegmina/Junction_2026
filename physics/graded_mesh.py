"""Orthogonal piecewise-uniform grids concentrated at the buried object.

These change discretization only, never the equations or material properties.
Grid admission still requires an independently solved 2x refinement.
"""
import math

import numpy as np


def make_mesh(extent=(1.,1.,.4),centre=(.5,.5),radius=.12,depth=.03,thickness=.05,refinement=1):
    if refinement<1 or int(refinement)!=refinement:raise ValueError('Positive integer refinement required')
    if min(extent)<=0 or min(depth,thickness,radius)<=0 or depth+thickness>=extent[2]:raise ValueError('Positive domain/target dimensions and a contained buried target required')
    result={}
    for axis in (0,1):
        c=centre[axis];length=extent[axis]
        if c-radius-.08<=0 or c+radius+.08>=length:raise ValueError('Target refinement zone must remain inside domain')
        positions=sorted(set([0.,length,*[v for v in [c-.5,c-radius-.08,c-radius,c+radius,c+radius+.08,c+.5] if 0<v<length]]))
        counts=[]
        for left,right in zip(positions,positions[1:]):
            distance=abs((left+right)/2-c)
            spacing=2*radius/12 if distance<radius else (.04 if distance<radius+.08 else .15)
            counts.append(max(1,math.ceil((right-left)/spacing-1e-9))*refinement)
        result['Grid.Positions'+str(axis)]=' '.join(format(v,'.12g') for v in positions)
        result['Grid.Cells'+str(axis)]=' '.join(str(v) for v in counts)
    length=extent[2]
    positions=sorted(set([0.,length,*[v for v in [length-.4,length-.2,length-.1,length-depth-thickness,length-depth] if 0<v<length]]))
    counts=[]
    for lower,upper in zip(positions,positions[1:]):
        below=length-(lower+upper)/2
        spacing=.001 if below<depth else (.005 if below<depth+thickness else (.01 if below<.1 else (.05 if below<.2 else .1)))
        counts.append(max(1,math.ceil((upper-lower)/spacing-1e-9))*refinement)
    result['Grid.Positions2']=' '.join(format(v,'.12g') for v in positions)
    result['Grid.Cells2']=' '.join(str(v) for v in counts)
    return result


def axis_nodes(parameters,axis):
    """Recover the actual supported uniform/piecewise-uniform grid nodes."""
    key='Grid.Positions'+str(axis)
    if key not in parameters:
        extent=[float(v) for v in parameters['Grid.UpperRight'].split()]
        cells=[int(v) for v in parameters['Grid.Cells'].split()]
        return np.linspace(0,extent[axis],cells[axis]+1)
    positions=[float(v) for v in parameters[key].split()]
    cells=[int(v) for v in parameters['Grid.Cells'+str(axis)].split()]
    grading=[float(v) for v in parameters.get('Grid.Grading'+str(axis),'1').split()]
    if any(g!=1 for g in grading):raise ValueError('Admission verifier currently supports piecewise-uniform tensor grids only')
    if len(positions)!=len(cells)+1:raise ValueError('Inconsistent tensor-grid positions/counts')
    return np.concatenate([np.linspace(a,b,n+1)[:-1] for a,b,n in zip(positions,positions[1:],cells)]+[np.asarray([positions[-1]])])
