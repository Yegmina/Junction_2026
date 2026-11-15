"""Conservative Cartesian interpolation bounds for a native radial field.

For a C1 radial function Hessian eigenvalues are f'' and f'/r. Bilinear
error <= H*(dx^2+dy^2)/8. The first annulus is constant at the axis.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator,PPoly


def reconstruction(radius,values):
    r=np.r_[0.,np.asarray(radius,dtype=float)]
    y=np.c_[np.asarray(values,dtype=float)[:,0],np.asarray(values,dtype=float)]
    if not np.all(np.diff(r)>0):raise ValueError('Positive increasing radial centres required')
    return PchipInterpolator(r,y,axis=1,extrapolate=False)


def hessian_bound(poly):
    a,b,c,_=poly.c;h=np.diff(poly.x)[:,None]
    second=np.maximum(np.abs(2*b),np.abs(6*a*h+2*b))
    first=np.maximum(np.abs(c),np.abs((3*a*h+2*b)*h+c))
    with np.errstate(divide='ignore',invalid='ignore'):vertex=np.where(a!=0,-b/(3*a),0.)
    vertex=np.clip(vertex,0,h);first=np.maximum(first,np.abs((3*a*vertex+2*b)*vertex+c))
    if np.any(first[0]>1e-12):raise ValueError('Reconstruction is not regular at the symmetry axis')
    ratio=np.zeros_like(first);ratio[1:]=first[1:]/poly.x[1:-1,None]
    return float(max(np.max(second),np.max(ratio)))


def plan(radius,temperature,moisture,width=1.,minimum_cells=100,maximum_cells=512,temperature_tolerance=.002,moisture_tolerance=1e-4):
    tp=reconstruction(radius,temperature);mp=reconstruction(radius,moisture)
    ht=hessian_bound(tp);hm=hessian_bound(mp)
    qt=float(np.spacing(np.float32(np.max(np.abs(temperature))))/2);qm=float(np.spacing(np.float32(np.max(np.abs(moisture))))/2)
    if temperature_tolerance<=qt or moisture_tolerance<=qm:raise ValueError('Tolerance below float32 serialization precision')
    needed=max(width*np.sqrt(ht/(4*(temperature_tolerance-qt))),width*np.sqrt(hm/(4*(moisture_tolerance-qm))))
    n=max(minimum_cells,int(np.ceil(needed)))
    if n>maximum_cells:raise ValueError(f'Projection needs {n} cells per axis, above cap {maximum_cells}; inaccurate export rejected')
    dx=width/n
    certificate=dict(status='bounded',cartesian_cells_per_axis=n,temperature_error_bound_k=ht*dx*dx/4+qt,moisture_error_bound=hm*dx*dx/4+qm,temperature_tolerance_k=temperature_tolerance,moisture_tolerance=moisture_tolerance,temperature_hessian_bound_k_m2=ht,moisture_hessian_bound_m2=hm,proof='Exact piecewise-cubic derivative bounds give H=max(abs(f_second),abs(f_prime/r)). Bilinear error <=H*(dx^2+dy^2)/8 plus float32 rounding. Convex interpolation in time preserves the maximum bound.',scope='Cartesian interpolation interior between first/last centres; no field extrapolation',native_reconstruction='C1 PCHIP; constant first annulus; reconstructed fields interpolated linearly in time',research_accepted=False)
    return n,tp,mp,certificate


def frame(poly,index,query):
    return PPoly(poly.c[:,:,index],poly.x,extrapolate=False)(query)
