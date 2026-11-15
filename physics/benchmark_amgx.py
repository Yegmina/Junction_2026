#!/usr/bin/env python3
"""Inspect AMGX fp64 convergence on an actual exported DuMux Newton system."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import mmread


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',type=Path);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tolerance',type=float,default=1e-12);p.add_argument('--amg',action='store_true');p.add_argument('--preconditioner',choices=['jacobi','amg','dilu','ilu'],default='jacobi');p.add_argument('--solver',default='FGMRES');p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();lib=ct.CDLL(str(a.library),mode=ct.RTLD_GLOBAL);H=ct.c_void_p;I=ct.c_int
    def function(name,types):
        f=getattr(lib,'AMGX_'+name);f.argtypes=types;f.restype=I
        def call(*args):
            result=f(*args)
            if result:raise RuntimeError('AMGX_'+name+' returned '+str(result))
        return call
    init=function('initialize',[]);plugins=function('initialize_plugins',[]);finish=function('finalize',[]);finishplugins=function('finalize_plugins',[])
    createconfig=function('config_create',[ct.POINTER(H),ct.c_char_p]);createresource=function('resources_create_simple',[ct.POINTER(H),H]);creatematrix=function('matrix_create',[ct.POINTER(H),H,I]);createvector=function('vector_create',[ct.POINTER(H),H,I]);createsolver=function('solver_create',[ct.POINTER(H),H,I,H])
    upload=function('matrix_upload_all',[H,I,I,I,I,H,H,H,H]);bind=function('vector_bind',[H,H]);uploadvector=function('vector_upload',[H,I,I,H]);zero=function('vector_set_zero',[H,I,I]);setup=function('solver_setup',[H,H]);solve=function('solver_solve',[H,H,H]);download=function('vector_download',[H,H]);status=function('solver_get_status',[H,ct.POINTER(I)]);iterations=function('solver_get_iterations_number',[H,ct.POINTER(I)])
    A=mmread(a.case/'jacobian-scaled.mtx').tobsr(blocksize=(2,2));A.sort_indices();b=np.asarray(mmread(a.case/'rhs-scaled.mtx')).ravel().astype('float64')
    raw=mmread(a.case/'jacobian-original.mtx').tocsr();rawb=np.asarray(mmread(a.case/'rhs-original.mtx')).ravel()
    rows=np.asarray(A.indptr,dtype='int32');cols=np.asarray(A.indices,dtype='int32');data=np.asarray(A.data,dtype='float64');answer=np.zeros_like(b);n=len(rows)-1
    preconditioner=dict(solver='BLOCK_JACOBI',max_iters=1,relaxation_factor=.9)
    if a.amg or a.preconditioner=='amg':preconditioner=dict(solver='AMG',algorithm='AGGREGATION',selector='SIZE_4',max_iters=1,presweeps=1,postsweeps=1,cycle='V',max_levels=20,smoother=dict(solver='BLOCK_JACOBI'))
    if a.preconditioner in ('dilu','ilu'):preconditioner=dict(solver='MULTICOLOR_'+('DILU' if a.preconditioner=='dilu' else 'ILU'),max_iters=1)
    config=dict(config_version=2,solver=dict(solver=a.solver,max_iters=2000,gmres_n_restart=80,tolerance=a.tolerance,norm='L2',convergence='RELATIVE_INI_CORE',monitor_residual=1,print_solve_stats=0,preconditioner=preconditioner))
    handles=[H() for _ in range(6)];cfg,res,mat,rhs,x,solver=handles
    init();plugins();began=time.perf_counter()
    createconfig(ct.byref(cfg),json.dumps(config).encode());createresource(ct.byref(res),cfg);creatematrix(ct.byref(mat),res,8193);createvector(ct.byref(rhs),res,8193);createvector(ct.byref(x),res,8193);createsolver(ct.byref(solver),res,8193,cfg)
    upload(mat,n,len(cols),2,2,H(rows.ctypes.data),H(cols.ctypes.data),H(data.ctypes.data),None);bind(rhs,mat);bind(x,mat);setup(solver,mat);uploadvector(rhs,n,2,H(b.ctypes.data));zero(x,n,2)
    timings=[]
    for repeat in range(a.repeats):
        zero(x,n,2);started=time.perf_counter();solve(solver,rhs,x);download(x,H(answer.ctypes.data));timings.append(time.perf_counter()-started)
    elapsed=timings[0]
    outcome=I();count=I();status(solver,ct.byref(outcome));iterations(solver,ct.byref(count))
    residual=raw@answer-rawb;scaled=A@answer-b
    result=dict(status_code=outcome.value,iterations=count.value,config=config,total_s=time.perf_counter()-began,solve_download_s=elapsed,scaled_relative_residual=float(np.linalg.norm(scaled)/np.linalg.norm(b)),original_relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(rawb)),water_relative_residual=float(np.linalg.norm(residual[::2])/np.linalg.norm(rawb[::2])),energy_relative_residual=float(np.linalg.norm(residual[1::2])/np.linalg.norm(rawb[1::2])))
    result['repeated_solve_download_s']=timings
    native=a.case/'correction-native-istl.mtx'
    if native.exists():
        reference=np.asarray(mmread(native)).ravel();result['relative_solution_error_to_native']=float(np.linalg.norm(answer-reference)/np.linalg.norm(reference))
    result['passed']=outcome.value==0 and all(result[k]<=1e-8 for k in ['scaled_relative_residual','original_relative_residual','water_relative_residual','energy_relative_residual'])
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    for name,h in [('solver',solver),('vector',x),('vector',rhs),('matrix',mat),('resources',res),('config',cfg)]:function(name+'_destroy',[H])(h)
    finishplugins();finish()


if __name__=='__main__':main()
