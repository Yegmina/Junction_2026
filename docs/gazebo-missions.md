# Prompt-driven Gazebo investigations

The Missions tab at http://127.0.0.1:18900/?mission= connects the mission
controller to an isolated, local Gazebo Harmonic world. It does not connect to
the Jetson or a physical aircraft.

## Reused work

The existing `/home/yehormsi/src/PX4-Autopilot-v1.17.0` checkout supplies the
BSD-licensed x500 geometry, inertias and rotor motor plugins. Existing
`DroneMovingSimulation` and its mine-report worktree informed the native
odometry/camera setup and evidence/report conventions. Their hardware-launch
scripts are not executed by this application.

Gazebo integrates forces from native motors and its multicopter velocity
controller. A position loop issues velocity commands; dashboard positions come
from native odometry. Camera images come from Ogre2 sensors. The scene uses
assigned material temperatures: this demonstrates an investigation workflow,
and does not establish realistic soil heating, sensor calibration or mine
detection accuracy. DuMux validation and real-flight labeling remain separate.

## Mission behavior

- An instruction chooses allowed areas, altitude and a simulated flight budget.
- The multimodal agent receives public state, local contrast proposals and the
  latest camera image. It chooses bounded inspect/revisit/return actions.
- Camera observations and normalized image boxes are retained with timestamps,
  actual capture pose, immutable evidence identifiers and raw provider records.
- A scripted reference uses the same available areas and local proposal tool.
- The simulator pauses during model inference, so API latency does not consume
  flight time. Motion and airborne waiting consume simulated time.
- Return/land and an evidence-linked report complete the mission. Errors are
  retained; an invalid action does not become a scripted successful run.

The controller uses the configured Gemini bulk model on the separate hackathon
API key. Its actual model ID is retained in evidence/provider logs. The existing
Robotics ER2 bulk labeler has its own quota; this mission controller does not
issue additional ER2 requests. The reference makes no API requests.

Example prompts:

1. `Survey east at 5m and return home within 2 minutes.`
2. `Survey north and south, revisit uncertain candidates, return within 3 minutes.`
3. `Inspect west carefully at 3m, avoid east, and return within 2 minutes.`

Coordinates are ENU meters around launch, within ±24m. Regions lie on four
nonoverlapping corridors from home. The current scene is fixed;
`world_seed_applied=false` explicitly prevents interpreting mission seeds as
randomized physics layouts.

## API and storage

`GET /api/missions` reads readiness and saved runs. `POST /api/missions` accepts
`{prompt,controller:"agent"|"reference",seed}` and begins a mission.
`GET /api/missions/{id}` reads public state;
`POST /api/missions/{id}/stop` requests return and preserves observations.
`GET /api/missions/camera/{overview|thermal|rgb}` returns native camera PNGs.

Artifacts live on the external SSD under
`artifacts/missions/{id}/`: state, original camera observations, overlays,
immutable evidence, prompts, raw provider responses and final report. Public
APIs do not expose private simulator target layouts. Image boxes remain image
boxes; camera pose is not fabricated geographic target coordinates.

## Challenge scope

This adds a visible, tool-bounded investigation task to the existing AfterQuery
benchmark work. A working demo alone is not proof of a frontier-agent failure
gap or a policy improvement. Independent frozen scene references, paired seeds,
equal budgets, held-out episodes and the required submission harness still
govern any scored claim. See `mission-evidence-benchmark.md` for the scoring
contract and the current incomplete-reference restrictions.

## Measured development receipts

- Native force-driven flight: body altitude 0.227→3.31m, horizontal travel
  0→−8.24m, home error 0.053m, landing body height 0.227m. Native paused clock
  checks measured zero advancement. Full receipts are on SSD under
  `simulation/provenance/`.
- Reference run `2464606f40fc4c95` completed with 71 measured route samples,
  one east-area thermal observation, two local contrast proposals and verified
  return/landing. Two proposals do not establish two physical objects.
  Dashboard video: `/artifacts/presentation/mission-2464606f40fc4c95-dashboard.mp4`.
- Runs `69abe3e9336b47c3` and `684e8e555ef3402e` exposed environment pause/control
  defects before the first model decision. They are retained as failed
  integration runs and must not be scored as frontier-agent planning failures.
- Agent run `86bf53f1954745c9` completed: north and south at 3m, then south and
  north revisits at 2m selected by the model, followed by return and landing.
  Four actual camera observations were collected in 101.13/180 simulated flight
  seconds. Prompts, five paid raw API responses and immutable evidence are
  retained. Camera compression now runs in a latest-frame worker outside the
  telemetry/control lock; pause requests use a separate CLI process. Stable
  hover with three pause/resume cycles and zero paused-clock advancement passed.
  Dashboard video: `/artifacts/presentation/mission-86bf53f1954745c9-dashboard.mp4`.
  Both completed demonstrations are archived under
  `gs://aerorozum-thermal-rl-avid-relic-509408-e4/mission-demos/20261114/`.
- Arrival now requires position tolerance **and** measured speed below 0.4m/s
  over three samples; passing through a waypoint is insufficient.
- `simulation/mosaic_ground.py` constructs a 4096² material from 289 distinct
  training-only, teacher-screened ZIP crops with feathered overlap and recorded
  transformations. This replaces a 212×256 crop stretched across 70m. Full source
  hashes, normalized display processing, seed and the assumed 293–297K mapping
  are retained in `simulation/provenance/recorded-ground-mosaic.json` on SSD.
  The assigned 1.7cm material sampling is not measured ground calibration.
