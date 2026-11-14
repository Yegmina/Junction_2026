# GPU physics execution

The user authorized preparing a genuine GPU physics backend on the existing
four-A100 VM and running it when model validation releases the GPUs.

The CPU reference is the custom DuMux 3.10 ExtendedRichardsNI campaign with
cached parameters, graded meshes, compact surface output and unchanged physical
equations. Its CPU linear backend is `ILUBiCGSTABIstlSolver`.

NVIDIA AMGX is pinned at `91a8413ef267b1c32aff4014c02820e1c5897ac2`. Source and
upstream licensing remain under `/opt/aerorozum/gpu-physics/AMGX` on the worker.
The isolated build targets A100 CUDA architecture 80, CUDA 12.9, a shared library
and no MPI. `cloud/build-gpu-physics.sh` records revision, status and logs in
`gs://aerorozum-thermal-rl-avid-relic-509408-e4/physics/gpu-backend-build/`.
It does not modify the training environment or replace the CPU binary.

Execution gates:

1. Export an actual DuMux Jacobian and matching right-hand side.
2. Solve that same system in double precision on GPU; compare residual and
   solution with CPU, including host/device transfer cost.
3. Integrate GPU linear solves into the original Newton loop, preserving
   assembly, physical parameters and residual tolerances.
4. Compare an identical complete CPU/GPU scene: surface temperatures, moisture,
   conservation ledgers, accepted timesteps and total runtime.
5. Use it for generation only if numerical agreement holds and the complete
   measured workload is faster. A library build or matrix benchmark alone does
   not establish an accepted physics field.

The latest compact CPU profile attributed approximately 62–64% of Newton time
to assembly, 26.5–28.6% to linear solves and 8.8–9.2% to state update. Accelerating
linear solves alone has a theoretical whole-step speedup ceiling near 1.4×.
That is a profiling estimate, not a measured GPU speedup. Larger acceleration
would require assembly acceleration and equivalent numerical checks.

Grid convergence is independent of device choice. The current .026 K screening
discrepancy still exceeds the required .020 K. GPU-generated output must satisfy
the same numerical admission checks as CPU output.
