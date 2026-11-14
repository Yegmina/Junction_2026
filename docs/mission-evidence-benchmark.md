# Mission evidence and benchmark contract

`aerorozum.mission_evidence` binds each proposed detection to the image bytes,
original capture context and a stable evidence ID. It never constructs an API
client, controls a drone, reads target truth, or fabricates a camera image.

## Working local integration

```python
from aerorozum.mission_evidence import (
    local_thermal_proposals, record_observation_evidence, build_mission_report,
)

context = {
    "mission_id": "mission_001",
    "observation_id": "obs_001",
    "source_kind": "gazebo_camera",
    "sensor_modality": "simulated_thermal_intensity",
    "sim_time_s": 32.1,
    "pose": {"x_m": 2.0, "y_m": 4.0, "z_m": 6.0, "yaw_rad": 0.2},
    "capture": {
        "camera_topic": "/world/survey/model/drone/thermal",
        "event_id": "capture_001",
    },
}
proposals = local_thermal_proposals(actual_camera_image)
evidence = record_observation_evidence(actual_camera_image, context, proposals)
report = build_mission_report("mission_001", [evidence], candidate_tracks=[])
```

For an actual native bridge receipt, use the strict adapter instead of the
manual context above:

```python
from aerorozum.mission_evidence import context_from_capture
ctx = context_from_capture(
    actual_camera_image, receipt,
    mission_id=mission_id, observation_id=observation_id,
)
evidence = record_observation_evidence(
    actual_camera_image, ctx, local_thermal_proposals(actual_camera_image),
)
```

The receipt must include `atomic_capture=true`, `image_sha256`,
`pose={x_m,y_m,z_m,...}`, `pose_frame`, `sim_time_s` and `camera_topic`.
Optional `camera`, `display_mapping`, `event_id`, `bridge` and `sensor_frame`
are preserved through a public allowlist. A body pose must be labeled as such;
unknown camera extrinsics do not become measured optical calibration. The
adapter rejects missing/mismatched hashes and non-atomic metadata, and never
substitutes the desired navigation target or an earlier pose. Separate live
`/frame` and `/frame-info` requests do not establish an atomic capture.

The caller obtains `pose`, sensor time and image from one authoritative capture
transaction; this module cannot independently authenticate a Gazebo bridge.
`image_sha256` may be supplied in the context and is checked against the actual
file. Original source files are immutable. Saved `raw_image` and `overlay_image`
references are relative to the artifact root when located under it; an explicit
external `output_dir` uses paths relative to that directory.

`local_thermal_proposals(image_path, max_proposals=12)` consumes **only pixels**.
It computes adaptive background residuals, finds both warm and cold connected
components, and rejects broad regions, image edges and implausible component
sizes/aspect ratios. Its `confidence` is an **uncalibrated contrast score**, not
a mine probability. All of its overlay boxes are amber. An empty proposal set
has `zero_candidates=null` and never establishes absence. The deterministic
baseline and frontier policy receive the same proposal tool and original
camera pixels. Both may inspect the image; neither receives privileged labels.

`record_observation_evidence(image_path, context, detection, output_dir=...)`
also accepts actual frontier-agent proposals with explicit `model` and
`provenance`. `boxes` and `uncertain_boxes` use named normalized image boxes
`x_min,y_min,x_max,y_max` in 0..1000 and scores in 0..1. Invalid geometry is
rejected rather than clipped or silently reordered. The helper does not verify
a caller's provider identity; the orchestrator retains that actual API receipt.
It records that limitation explicitly and issues no requests of its own.

`build_mission_report(mission_id, evidence, candidate_tracks=...)` accepts tracks
of this form:

```json
{"candidate_id":"candidate_001","evidence":[
  {"evidence_id":"opaque_image_bound_id",
   "bbox":{"x_min":200,"y_min":300,"x_max":260,"y_max":390}}
],"notes":"Provisional image signature; revisit requested"}
```

Reports distinguish tracked candidates from box measurements and set confirmed
physical mines to zero. They reject missing, conflicting or cross-mission
evidence. They never infer geographic mine coordinates. Repeated sightings do
not by themselves establish distinct objects or reduce positional uncertainty.

## Pixel source and fidelity

| `source_kind` | Required source binding | Valid interpretation |
|---|---|---|
| `gazebo_camera` | Actual camera topic, capture pose, original sensor image; caller supplies sensor time/event | Actual simulator camera pixels; thermal appearance remains unvalidated unless independently established |
| `dumux_field_crop` | Actual crop/camera pose and `capture.field_sha256` | Display of that actual physical field; numerical admission is distinct from field validation |
| `domain_randomized_thermal_world` | Actual game-render capture pose/image/event | Explicit simulation demonstration, not validated thermal physics |
| `recorded_reference` | Exact original recorded-image hash; **no simulated moved pose** | Static recorded-image/review benchmark only |

Recorded flight/ZIP pixels may texture the rendered terrain when declared as
`appearance={"kind":"recorded_texture","source_sha256":"..."}`. The actual
camera must render that terrain at the actual pose. A replayed ZIP image cannot
stand in for a moved Gazebo camera. Recorded texture identity stays
`not_verified_target_free`, including teacher-screened negative crops.

The public context is a deep allowlist. Target arrays, private object positions,
segmentation labels, evaluator state, hidden eligibility and arbitrary task
fields do not reach detector inputs or reports. Detector requests send only the
fixed registered prompt and original image. Camera pose is evidence metadata,
not an input for locating targets through privileged scene data.

High-bit temperature arrays require an explicit upstream display mapping.
The detector rejects a 16-bit image rather than silently truncating it into a
misleading 8-bit image. The bound field/camera metadata should state that mapping,
PSF, contrast, emissivity assumptions and calibration state. Merely rendering a
thermal-looking grayscale frame does not establish radiometric validity.

## Optional exact Robotics ER2 interface: currently no live mission allowance

```python
result = await analyze_observation(
    actual_camera_image, context,
    provider=mission_dedicated_provider,
    request_authorizer=coordinator_authorize,
    output_dir=mission_evidence_dir,
)
```

Without a provider this is cache-only. Without an explicit coordinated lease it
returns `awaiting_request_lease`, empty display lists, `model=null` and
`zero_candidates=null`. This is unavailable analysis, not a negative detection.
It uses exactly `gemini-robotics-er-2-preview`; no alternate model/client is
constructed. The configured registered thermal prompt is recorded by hash and
version; RGB frames explicitly adapt its wording and record a distinct prompt
hash rather than calling an RGB camera a physical thermal sensor.

The authorization callback receives a request identity and returns:

```json
{"allowed":true,"lease_id":"mission-reserved-slice",
 "allowance_rpm":10,"expires_at":"2026-11-14T18:00:00+00:00",
 "global_quota_coordinated":true}
```

The mission provider's actual `limiter.rpm` must be at most that allowance,
including retries. The caller is responsible for reconciling the remote bulk
worker's allowance before granting it: **bulk120 + mission10 is forbidden**.
A lease may be granted only after a coordinated bulk110/mission10 split or a
quiet exclusive handoff is actually installed. Until then use the local pixel
reference or existing cached ER2 output. The separate frontier agent may use its
explicitly authorized different-model quota under the root orchestrator.

An immutable request reservation precedes dispatch. The paid output and raw
provider record precede schema validation. Malformed/refused output is retained
and never becomes a negative. Unresolved reservations prevent automatic
repayment. Concurrent identical observations share one inference cache but
retain distinct context-bound evidence IDs. Process file locks prevent a second
process from dispatching the same image/prompt simultaneously. Caller-managed
shutdown must drain a real provider before terminating its process.

## Evaluator-only reference and fair scoring

`score_mission_report(report, evidence, private_reference,
qualified_coverage=..., completed=True)` is a **privileged evaluator function**,
not an agent/MCP/detector tool. A reference has:

```json
{"reference_id":"frozen_seed_17","reference_kind":"simulator_private",
 "reference_scope":"complete","frozen_before_run":true,
 "eligible_target_ids":["private_target_A","private_unvisited_B"],
 "observability_verified":false,
 "observations":{"obs_001":{"image_sha256":"actual_camera_sha256",
   "targets":[{"target_id":"private_target_A","observable":true,
     "bbox":{"x_min":200,"y_min":300,"x_max":260,"y_max":390}}]}}}
```

- Freeze object layouts, eligibility/observability conditions, seeds, routes'
  allowed area, sensor quality, action/flight budget and completion rules before
  a run. The eligible target denominator must not shrink to what the agent
  happens to visit. Unvisited eligible targets remain false negatives.
- Derive synthetic per-frame boxes from the **actual** render/camera projection
  and visibility, or reviewed rendered-image annotations. Do not match world
  coordinates against unrelated recorded images. Keep all target/object IDs
  and masks outside the agent-visible interface.
- Match submitted candidate tracks one-to-one at the fixed IoU0.5 threshold.
  Duplicate tracks are false positives. Report TP/FP/FN, recall, precision, F2,
  qualified coverage, actions, simulated flight time, inference calls, tokens
  and wall time; retain per-seed results and failures, not only a best run.
- Use the same action/time/capture budgets, frames, proposal tools, execution
  limits and public state for lawnmower/reference and frontier policies. The
  same public candidate association rules should apply to both. Teacher output
  is not independent ground truth or a training-policy improvement claim.
- Proposed mission splits should be frozen before tuning. Development seeds
  can tune controllers; validation seeds choose prompts; sealed test seeds are
  opened only after that selection. This mission benchmark supplements the
  existing three-seed training protocol; it does not silently replace it.
- Include both complete synthetic empty scenes and target scenes, distractors,
  warm/cold contrast, adverse times/altitudes and sensor artifacts. A policy
  cannot win simply by returning nothing: recall and coverage remain visible.
- For reviewed recorded frames use `reference_kind="reviewed_recorded"` and
  `human_reviewed=true`, exact original image hashes, and preserve the archive's
  frozen zones. Positive-only zone7 supports localization agreement, not a
  negative-scene false-alarm rate. Machine/user-accepted training labels are
  not independent benchmark annotations.
- If real background identities are unknown, use
  `reference_scope="inserted_objects_only"`: score inserted-object recall and
  unmatched proposals, but closed-world FP/precision/F2/reward and negative-scene
  false-alarm rates are unavailable. Explicitly inserted targets do not prove
  every other recorded texture feature is target-free.

No score is labeled real mine-detection accuracy. Numerical physics acceptance,
render consistency, sensor calibration and real-field validation are separate
receipts. Missing receipts are reported as missing.

## Existing-repository review and verification

Read-only reference review covered:

- `../gemini-robotics-live/survey/engine.py`: immutable original frames,
  observation/candidate IDs, evidence crops and linked reports; narrative
  summaries must not invent coordinates/counts.
- `../gemini-robotics-live/survey/association.py`: candidate identities require
  independent ground-background registration, mutual box matches and same-frame
  exclusion; GPS proximity alone is insufficient.
- `../DroneMovingSimulation-mine-report-v1/gemini-robotics-live/survey/README.md`:
  explicit camera/geolocation assumptions, honest replay limitations, independent
  raw/annotated evidence and safe live capture.

These patterns informed the new module. The old survey runtime, live Jetson
controls, API clients, archived detections and paid calls were not imported or
started. Unit checks cover hot/cold/flat/broad imagery, immutable pixels, truth
redaction, replay-pose rejection, malformed image/boxes, quota gating/cache
reuse, concurrent duplicate suppression, invalid paid-output retention,
one-to-one scoring, unvisited targets and incomplete references. No live
Robotics ER2 request was made while implementing or testing this module.


## Actual native-camera smoke receipt

A read-only smoke check captured a genuine native Gazebo thermal PNG at
simulation time193.628s and measured vehicle height approximately3m. The
atomic bridge image SHA was
`0a123a38333657e2e0144e95544f7c570bfb08dffe52451a52aea48c19e3ec4a`.
The640×512 image was uniform at the home pose; the local contrast reference
returned zero proposals and made no absence claim. No control command or paid
model call was issued by this check. Original pixels, the bridge receipt,
overlay and linked JSON report are saved under
`artifacts/missions/evidence-smoke/`. Evidence ID:
`0174c2937a92008c97410d7b1671d475`. This validates the capture/provenance/report
plumbing, not detector accuracy. Fourteen evidence checks plus five mission
checks passed at integration.

A second genuine native thermal frame from the west travel proof was checked at
simulation time205.876s, pose(−8.219,−0.0086,3.006)m. Native odometry was0.008s
later; that offset is retained. The image SHA is
`1aab3f7b378128a4d6a6c36f3c6fda33724e7086b34bc91b10b6348ff0bae307`.
Two visible contrast regions are cut by the image edges. The unchanged local
reference rejected them under its edge rule and returned zero proposals;
this is a documented reference limitation, not evidence of target absence.
The frontier receives the same original image and may inspect or revisit it.
The bound overlay/report are in `artifacts/missions/evidence-smoke-west/`,
evidence ID`c490ff2361c3d8c103e27d343dbf6c34`. No paid model call or control
command was issued by either evidence smoke check.
