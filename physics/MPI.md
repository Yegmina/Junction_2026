The optional `--mpi-compact` application keeps the native DuMux residual,
Jacobian, Newton method and distributed ISTL solver. It gathers diagnostic
surface faces owned by each rank and retains every solved physical time in the
same float64 exchange stream. Storage, external transport and interface-flux
ledgers use native MPI sums. No distributed AMGX implementation is claimed;
the current AMGX adapter is explicitly restricted to one process.

Build and run with the existing pinned DuMux/DUNE installation:

```sh
python physics/build_campaign.py --cached --ledger --surface --graded --cached-parameters --compact-output --amgx --mpi-compact --jobs 4
PYTHONPATH=. python physics/run_dumux.py --output-dir /absolute/fresh/case --duration 600 --dt 30 --threads 1 --cached --ledger --surface --graded --cached-parameters --compact-output --amgx --mpi-compact --mpi-ranks 4 --surface-refinement 2
```

On November 14, an executed 600-second graded 3D pilot at one and four ranks
retained identical Cartesian axes and adaptive time states. Native float64
maximum differences were 1.14e-13 K and 4.83e-15 volumetric moisture; both
actual mass/energy, target-fluid and interface-flux checks passed. Exported
float32 fields were identical. Observed wall times were 26.90 and 10.48 seconds
on the local pilot; these are exploratory timings, not a cloud production
speed guarantee. Evidence is stored externally under
`physics/mpi-parity-20261114/verification.json` with source hashes and native
budget snapshots.

`verify_parallel.py REFERENCE CANDIDATE --duration 600 --output REPORT`
compares both native time grids without extrapolation. It requires matching
frozen physical inputs, matching physical mesh, maximum temperature error
1e-4 K, maximum moisture error 1e-6 and actual transport/interface budgets.
For a running longer reference it snapshots complete ledger rows through the
first solved state covering the comparison endpoint. This prefix test does
not certify a full day, grid/domain convergence or physical field validation.

The cloud 1,906,688-cell reference is benchmarked separately at 16 MPI ranks
and at four ranks with four assembly threads. Existing reference evidence is
preserved until a measured candidate passes parity and offers a useful
end-to-end runtime. All full-day conservation, refinement and strict 3D
correspondence requirements continue to apply.
