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

Use Harbor 0.26.0, Docker and Docker Compose (tested plugin v2.39.4). The task
uses schema 1.4. Install model credentials in the host environment, never in the
task files. For the hackathon Google key use `GEMINI_API_KEY`; the Harbor
OpenCode integration resolves the Google provider credentials.

```
harbor run -p ./harbor/thermal-investigation -a opencode -m gemini/gemini-4-argon
harbor run -p ./harbor/thermal-investigation -a oracle
```

The native scene assets and runtime sources must first be staged into
`environment/backend/assets/` and `environment/backend/runtime/`. The deployment
staging script is `cloud/stage_harbor_native.py`. It copies the existing native
scene losslessly, checks field hashes, rewrites container paths and records
`native-assets-manifest.json`. **An unstaged source checkout is not yet a
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
capture pose and camera calibration. One-to-one matching uses a 0.65m radius
around the frozen inserted synthetic targets. Duplicate/unmatched candidates
are false positives in F2. Coverage is the fraction of 1m grid cells in the
requested regions inside actual captured camera footprints. Multi-view credit
requires matched evidence from distinct captures at least 1m apart. Image
identities and hashes stay in the sidecar. Full-frame boxes, invented capture
IDs, repeated frames and local reward-file edits cannot create retrieval
credit. Submission and verifier sealing are irreversible; reward queries cannot
be used to optimize a still-running trial.

The verifier also saves score details, the authoritative ledger and
checksum-verified camera images. The reference policy uses only public images
and telemetry. It follows a survey grid and proposes strong compact thermal
residuals; it has no access to target coordinates.

## Scope and limitations

Flight uses native Gazebo multirotor dynamics. Thermal frames are perspective
crops of experimental DuMux-derived composed fields; RGB uses a separate
photographic material. This is an inserted-object investigation benchmark,
not evidence of real mine detection or accepted full-domain thermal physics.
The first task uses a frozen layout; independent layout replication and
shortcut/exploit controls remain necessary before claiming a robust frontier
performance gap. GPU post-training experiments are separate until a checkpoint
has actually been evaluated through the same Harbor task and harness.

The reference/agent comparison and run-log paths will be documented after
actual execution. A successfully built image or passing unit tests alone do
not establish submission readiness or RL superiority.
