# Thermal investigation under a flight budget

An agent receives a drone, paired thermal/RGB cameras and a limited flight
budget. It must investigate two regions, distinguish supported ground
signatures from clutter, choose useful revisits, return home, and submit an
evidence-linked report. OpenCode controls the drone through the installed
`drone` CLI; no embedded Gemini controller makes decisions on its behalf.

The hard part is deciding what to look at next. Lower flight gives more detail
but less coverage. Another image may resolve an uncertain sighting, but costs
flight time. The aircraft continues moving during inference, so an agent must
use capture-time pose rather than assume it is already at its destination.
Repeated images do not establish separate objects or independent views.

## Run

Use Harbor 0.26.0, OpenCode 1.19.5, Docker and Docker Compose (tested plugin v2.39.4). The task
uses schema 1.4. Install model credentials in the host environment, never in the
task files. Export `GEMINI_API_KEY` or `GOOGLE_GENERATIVE_AI_API_KEY` in your
shell. The runner explicitly maps the key to OpenCode's Google provider; a
plain Harbor invocation without that mapping failed authentication in testing.

```
python cloud/run_harbor_trial.py --task ./harbor/thermal-investigation --job-name gemini-warming-01 --seed 414331
harbor run -p ./harbor/thermal-investigation -a oracle
```

In the portable bundle, run from its root with
`python run_harbor_trial.py --task ./thermal-investigation --job-name gemini-warming-01 --seed 414331`.
The helper calls the actual `harbor run` command with the OpenCode harness,
pins OpenCode 1.19.5 and configures text/image input. It also checks the sealed
ledger's actual seed against the requested seed. `--model`, `--phase`,
`--budget`, `--jobs-dir` and `--harbor` are configurable. Default limits are
180 simulated flight seconds, 64 actions and 1800 wall-clock agent seconds.

The native scene assets and runtime sources must first be staged into
`environment/backend/assets/` and `environment/backend/runtime/`. The deployment
staging script is `cloud/stage_harbor_native.py`. Then run
`cloud/build_harbor_scene.py --assets TASK/environment/backend/assets --case QUALIFIED_CASE`
to freeze new insertions and their private reference together, using the
admitted full-day axisymmetric present/absent pair. The source hashes must match
the admission certificate. It preserves the existing photographic/recorded
ground appearance and generates all four thermal-condition fields without scaling
the physical contrast. Camera fields and private reference share frozen hashes.
`native-assets-manifest.json` records the build. **An unstaged source checkout is not yet a
self-contained distributable**; the final submission bundle must include these
assets or a pinned, publicly downloadable asset archive. The task is under
active integration; only actual completed Harbor logs count as run evidence.

## Score

The sidecar holds target identities, flight state, captures and the report.
The agent container receives only the public CLI and evidence it collects.
The verifier writes exactly one scalar in `[0,1]` to
`/logs/verifier/reward.txt`:

```
0.65 * F2 + 0.20 * surveyed_region_coverage + 0.15 * multi_view_target_fraction
```

A report must be submitted, at least one airborne observation collected, and
home return/landing completed within 180 simulated seconds and 64 actions.
Otherwise reward is zero. The flight clock stops after verified landing;
report-writing time on the ground does not consume flight budget.

Candidate centers are projected from their recorded image boxes using actual
capture pose and camera calibration, then associated by their median ground
position. One candidate has at most one distinct box in any capture; repeating
the identical box adds no evidence. One-to-one matching uses a 0.65m radius
around the frozen inserted synthetic targets. Duplicate/unmatched candidates
are false positives in F2. Coverage is the fraction of 1m grid cells in the
requested regions inside actual captured camera footprints. Evidence grouped
into one candidate must project to consistent ground locations (maximum pair
separation 1.3m); copying unrelated guesses into every record is rejected. Multi-view credit
requires matched evidence from distinct captures at least 1m apart. Image
identities and hashes stay in the sidecar. Full-frame boxes, invented capture
IDs, repeated frames and local reward-file edits cannot create retrieval
credit. Submission and verifier sealing are irreversible; reward queries cannot
be used to optimize a still-running trial.

The verifier also saves score details, the authoritative ledger and
checksum-verified camera images. The reference policy uses only public images
and telemetry. Version 0.3.1 first surveys at 10m, fits broad thermal appearance
from geometrically registered RGB, proposes compact thermal residuals of the
reported polarity, and spends remaining budget revisiting candidates at 3m.
It merges consistent locations across views and has no access to target
coordinates or the number of inserted objects. This revised reference still
needs native evaluation; its unit/offline checks are not mission scores.

At container startup, the scene factory redraws the insertions using a private
random seed. The seed is not exposed to the agent. Set `AEROROZUM_TASK_SEED` on
the Harbor host to reproduce an instance or pair model comparisons; the sealed
ledger records the actual instance seed and field identities. The physical
response kernels and terrain background arrays are checksummed and unchanged
when positions are redrawn. Freezing the background arrays avoids differences
between host/container image-resizing and reduction libraries. A native startup
check reproduced the staged field hashes for seed 414331 and confirmed that
another seed changes the fields.
`AEROROZUM_TASK_PHASE` selects `warming` (7.5 elapsed solver hours), `cooling`
(17h), `dawn` (1h), or `late_warming` (11.75h), and
`AEROROZUM_TASK_BUDGET` selects 60–600 simulated seconds (180 by default).
All budgets/conditions are exposed through public status before takeoff. The
runtime factory and private reference stay in the simulator container.

## Scope and limitations

Flight uses native Gazebo multirotor dynamics. The local axisymmetric thermal
response pair has passed numerical admission for one vertical cylinder in
homogeneous flat soil with spatially uniform forcing. Thermal frames are perspective
crops of composed fields; RGB uses a separate
photographic material. This is an inserted-object investigation benchmark,
not evidence of real mine detection or accepted full-domain thermal physics.
Each trial freezes its freshly generated layout; independent layout replication and
shortcut/exploit controls remain necessary before claiming a robust frontier
performance gap. GPU post-training experiments are separate until a checkpoint
has actually been evaluated through the same Harbor task and harness.

Version 0.2 adds the per-capture uniqueness and median-position rules. Version
0.3 adds two weaker-contrast snapshots from the same admitted full-day pair;
it does not scale physical contrast or change the sensor noise/resolution.
These new conditions still require native reference/model validation. Earlier
development scores from version 0.1 cannot be substituted for a rerun of this
verifier. Completed run evidence and exact task checksums belong in the
submission bundle. A successfully built image or passing unit tests alone do
not establish submission readiness or RL superiority.
