# Thermal and moisture physics

Two separate backends are tracked honestly:

* **DuMux 3.10.0 / DUNE 2.10.0**: `bootstrap_dumux.py` downloads pinned upstream source, preserves its licenses, builds the unmodified `ExtendedRichardsNI` evaporation example, and runs its reference regression. `build_campaign.py` builds the separate working 3D buried-object application; actual full-day paired fields have been executed. This is a real coupled nonisothermal Richards/vapour-diffusion solver. Its numerical admission still requires the complete independent refinements; outdoor physical validation has not been performed.
* **Experimental Python finite volume**: `aerorozum.physics` is executable now. It solves three-dimensional Richards liquid flow and heat transport with conservative internal fluxes, saturation-dependent material properties, surface evaporation/condensation, solar forcing, radiation and convection. It does **not** solve DuMux's subsurface vapour equation and must never be presented as DuMux output.

## Execute

Use the project environment, with `numpy` and `scipy` installed. Keep generated fields and builds on the external SSD.

```sh
python -m aerorozum.physics verify --output /path/to/data/physics/python-verification.json
python -m aerorozum.physics generate --config physics/smoke.json --output /path/to/data/physics/smoke.npz
python -m aerorozum.physics pair --config physics/day_smoke.json --output /path/to/data/physics/day-pair
python -m aerorozum.physics validate --output /path/to/data/artifacts/physics/verification.json
python physics/bootstrap_dumux.py --root /path/to/data/physics/dumux --jobs 4
python physics/build_campaign.py --root /path/to/data/physics/dumux --jobs 4
python physics/build_campaign.py --root /path/to/data/physics/dumux --cached --ledger --surface --graded --jobs 4
python physics/run_dumux.py --root /path/to/data/physics/dumux --output-dir /path/to/data/physics/case --cached --ledger --surface --graded --duration 86400 --dt 30 --threads 8
```

`generate` writes compressed NPZ arrays `times_s[T]`, `x_m[W]`, `y_m[H]`, `temperature_k[T,H,W]`, `moisture[T,H,W]`. Coordinates are east/north metres, temperature kelvin, moisture water volume per bulk volume. The public JSON sidecar records configuration hash, units, backend, conservation residuals, limits and `research_accepted: false`. Private same-stem `.truth.json` holds targets and complete physical configuration. Do not expose private truth to an agent.

## Fidelity limits and acceptance

The Python solver uses an implicit finite-volume heat equation and modified Picard Richards iterations. Internal water and sensible-energy transport are conservative; surface energy accounts for exported water's sensible and latent enthalpy. Boundaries are sealed except the atmospheric surface. Target cells use tiny porosity and permeability as an explicitly unaccepted regularization. Top-cell centre temperatures are exported, not a reconstructed radiative face. No roots, rainfall, gas advection, freeze/thaw or hydrodynamic flight simulation is provided.

`verify` actually checks equilibrium and closed-system energy, coupled evaporative mass/energy balance, an analytical cosine heat-decay solution and Richards hydrostatic equilibrium. Those checks alone do not constitute spatial/time convergence, complete Richards verification, physical calibration, or outdoor validation. Output remains `validation_state: unverified` and research generation must remain blocked until the full configured gates pass. `validate` publishes a staged manifest with missing gates explicitly pending and automatically computes `research_accepted` from all required results.

The custom DuMux campaign build is separate from the upstream regression and retains its GPL headers. It extends the pinned evaporation example to 3D, adds atmospheric forcing and a regularized near-impermeable target. Successful compilation is not a convergence/field-validation result. `physics/dumux_smoke.input` is a deliberately coarse smoke case. Run the built binary in its own SSD output directory with that input path. `physics/convert_dumux.py <result.pvd> --output <field.npz> --truth <private-targets.json>` converts the actual ASCII top-cell outputs to the exchange schema; it rejects binary/MPI output and 2D cross-sections rather than fabricating a surface. Generated field metadata always identifies this custom application as experimental and unaccepted until its own gates are implemented and passed.

Direct-field **numerical admission** requires actual upstream/analytical Richards and conduction checks, heterogeneous interface-flux continuity, custom DuMux mass/energy budgets, timestep/grid/domain convergence and negligible target-fluid storage/leakage. Published laboratory reproduction and independent field measurements remain informational physical-validation work; they do not block numerical admission and must not be claimed complete. A fast patch generator has its own additional direct-versus-surrogate error gate, independent of direct-field admission.

Prefer `build_campaign.py --cached --ledger --surface` and `run_dumux.py --cached --ledger --surface` for the corrected experimental application. The surface variant solves the half-cell conductive/atmospheric Robin balance, exports reconstructed boundary-face temperature, applies forcing at the adaptive implicit-step endpoint and carries evaporating water's component enthalpy. Earlier `legacy_cell_mixture_v1` outputs are retained as failed/coarse diagnostic evidence and must not be mixed into the corrected `surface_component_enthalpy_v2` dataset. Upstream geometry/state/flux caches change no equations; their equivalence was actually tested.

`conservation.csv` is written by the actual DuMux executable using its exact local residual storage, external boundary fluxes, and both sides of heterogeneous material faces. The Python assessor independently normalizes mass/energy residuals against cumulative absolute boundary transport (not large absolute stored energy). Near-zero interface flux uses explicit mixed absolute/relative roundoff tolerances. These are budget checks only; successful execution or conservation alone never auto-accepts an unconverged field.

`--graded` uses DUNE's existing orthogonal tensor-product grid manager, concentrating resolution above and around the object. `--mesh-refinement 2` independently halves every cell interval. This is a discretization optimization with unchanged PDE/materials, not a learned surrogate. Uniform centimetre-scale top cells demonstrably fail the configured surface-temperature convergence gate. Graded meshes remain experimental until their own full-day refinements pass.

`physics/admission.py` accepts explicit base/time/grid/domain/material-refinement fields and computes numerical admission from source/scene matching, full-day time coverage, actual budgets and direct temperature comparisons. It never admits a short smoke run as a day-long campaign. `--publish` updates field acceptance metadata only if **all** computed checks pass. The absent-object counterpart needs its own applicable numerical evidence. A surrogate's separate error certificate cannot substitute for a direct numerical certificate.

Refinement checks require both maximum surface-temperature error ≤0.02 K and relative error ≤2%. The relative temperature norm is trapezoid-time-weighted L2 normalized by the reference field's temporal excursion from its initial state; it is not normalized by the arbitrary absolute kelvin offset. Moisture has an independent ≤2% time-weighted relative norm. These requirements supplement each trajectory's mass/energy and interface budgets.

`build_campaign.py --cached --ledger --surface --graded --cached-parameters --compact-output --amgx` builds an optional host-side adapter to the pinned official NVIDIA AMGX C API. It loads an explicitly supplied shared library at runtime; GPU execution requires `run_dumux.py --amgx --set AMGX.Enabled=true --set AMGX.LibraryPath=/absolute/path/libamgxsh.so`. The original DuMux residual/Jacobian and Newton loop remain in C++; only the linear solve runs on the GPU in double precision, using coupled 2×2 blocks and an invertible block-row scaling. Each correction must pass independent scaled, original, water and energy residual checks. Failure aborts explicitly. GPU execution does not imply numerical admission or faster end-to-end runtime; compare against the native ISTL CPU path on the same scene.

`--set AMGX.ExportMatrix=true` exports the first genuine Newton matrix/RHS in original and scaled form. On the CPU path it also exports the native ISTL correction and timing. `benchmark_gpu_jacobian.py` is an isolated CUDA solver experiment on those exact matrices, not a replacement physics model. Its SciPy direct reference is a correctness reference; production speed comparisons use the native ISTL timing and full-case wall times. AMGX headers and BSD license are preserved under `physics/vendor/`; the adapter itself retains the DuMux application's GPL license.

`cloud_suite.py` runs a bounded CPU-only suite on an existing VM, with independent worker/thread limits; it never provisions resources. Its pilot uses short trajectories solely to measure numerical behavior and runtime. `upload_artifacts.py` sends only physics outputs/checkpoints to the designated private GCS prefix using VM application-default credentials without logging credentials. Full-day reference admission is still required after a pilot.

`live_progress.py` publishes actual ledger progress, conditional measured ETAs, native surface previews and partial grid comparisons. Partial passing results cannot admit a field. `watch_admission.py` waits for every full required reference, including halving both the initial and maximum timesteps, computes both direct certificates and publishes `registry.json` only when the pair passes. The pair certificate carries explicit `present_sha256`/`absent_sha256`; independent adaptive time arrays remain intact. Run-directory locks and source/parameter/input matching let parallel workers reuse identical completed cases without running them twice.

The measured November 14, 2026 AMGX v3 adapter passed the actual 600 s CPU/GPU field/budget comparisons with zero exported float32 differences. It was slower on this VM: base 9.87 s CPU versus 31.70 s GPU, nested fine 79.80 s CPU versus 131.35 s GPU. Production therefore remains on native ISTL CPU; GPU execution is available and measured, not assumed to accelerate this workload. The isolated SciPy direct solve is solely a correctness reference, not the production speed baseline.

Upstream sources: <https://github.com/dumux/dumux/tree/3.10.0/test/porousmediumflow/richards/nonisothermal/evaporation>, <https://github.com/dumux/dumux/blob/3.10.0/dumux/porousmediumflow/richardsextended/model.hh>. Vendored source retains upstream GPL-3.0-or-later licensing; the build script does not copy those headers into this package.
