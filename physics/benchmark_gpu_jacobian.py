#!/usr/bin/env python3
"""Benchmark a genuine exported DuMux Newton matrix; this does not solve physics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.io
import scipy.sparse.linalg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as csl
    pool=cp.cuda.MemoryPool();pool.set_limit(size=4*1024**3)
    cp.cuda.set_allocator(pool.malloc)
    directory=args.case
    paths=[directory/x for x in ('jacobian-scaled.mtx','rhs-scaled.mtx','jacobian-original.mtx','rhs-original.mtx')]
    A=scipy.io.mmread(paths[0]).tocsr();b=np.asarray(scipy.io.mmread(paths[1])).reshape(-1)
    rawA=scipy.io.mmread(paths[2]).tocsr();rawb=np.asarray(scipy.io.mmread(paths[3])).reshape(-1)
    receipt=dict(status='running',purpose='isolated genuine Jacobian benchmark, not an in-Newton production backend',gpu_name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),shape=list(A.shape),nnz=int(A.nnz),source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
    native_path=directory/'correction-native-istl.mtx'
    native=np.asarray(scipy.io.mmread(native_path)).reshape(-1) if native_path.exists() else None
    baseline_path=directory/'native-linear-baseline.json'
    if baseline_path.exists():receipt['native_production_baseline']=json.loads(baseline_path.read_text())
    args.output.write_text(json.dumps(receipt,indent=2))
    start=time.perf_counter();reference=scipy.sparse.linalg.spsolve(A,b);receipt['cpu_direct_s']=time.perf_counter()-start
    receipt['cpu_scaled_relative_residual']=float(np.linalg.norm(A@reference-b)/max(np.linalg.norm(b),1e-300))
    start=time.perf_counter();gpuA=cs.csr_matrix(A);gpub=cp.asarray(b);cp.cuda.Stream.null.synchronize();receipt['host_to_device_s']=time.perf_counter()-start
    attempts=[]
    for method in ('gpu_ilu0_bicgstab','gpu_jacobi_gmres'):
        result=dict(method=method)
        try:
            start=time.perf_counter()
            if method=='gpu_ilu0_bicgstab':
                ilu=csl.spilu(gpuA,fill_factor=1)
                preconditioner=csl.LinearOperator(gpuA.shape,matvec=ilu.solve,dtype=gpuA.dtype)
            else:
                diagonal=gpuA.diagonal()
                preconditioner=csl.LinearOperator(gpuA.shape,matvec=lambda x:x/diagonal,dtype=gpuA.dtype)
            cp.cuda.Stream.null.synchronize();result['gpu_preconditioner_s']=time.perf_counter()-start
            start=time.perf_counter()
            if method=='gpu_ilu0_bicgstab':answer,info=csl.bicgstab(gpuA,gpub,rtol=1e-10,atol=0,maxiter=3000,M=preconditioner)
            else:answer,info=csl.gmres(gpuA,gpub,rtol=1e-10,atol=0,restart=80,maxiter=2000,M=preconditioner)
            cp.cuda.Stream.null.synchronize();result['gpu_solve_s']=time.perf_counter()-start
            answer=cp.asnumpy(answer);result['solver_info']=int(info)
            result['scaled_relative_residual']=float(np.linalg.norm(A@answer-b)/max(np.linalg.norm(b),1e-300))
            result['original_relative_residual']=float(np.linalg.norm(rawA@answer-rawb)/max(np.linalg.norm(rawb),1e-300))
            result['relative_solution_error_to_cpu']=float(np.linalg.norm(answer-reference)/max(np.linalg.norm(reference),1e-300))
            if native is not None:result['relative_solution_error_to_native_istl']=float(np.linalg.norm(answer-native)/max(np.linalg.norm(native),1e-300))
            for name,offset in [('water',0),('energy',1)]:
                residual=(rawA@answer-rawb)[offset::2];component_rhs=rawb[offset::2]
                result[name+'_relative_residual']=float(np.linalg.norm(residual)/max(np.linalg.norm(component_rhs),1e-300))
            result['passed']=bool(info==0 and result['scaled_relative_residual']<=1e-8 and result['original_relative_residual']<=1e-8 and result['water_relative_residual']<=1e-8 and result['energy_relative_residual']<=1e-8 and result['relative_solution_error_to_cpu']<=1e-6)
        except Exception as exc:
            result.update(passed=False,error=f'{type(exc).__name__}: {exc}')
        attempts.append(result);receipt['attempts']=attempts;receipt['status']='passed' if any(x['passed'] for x in attempts) else 'failed';args.output.write_text(json.dumps(receipt,indent=2))
        if result['passed']:break
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
