Positive candidate claims require at least one saved image captured at or below
4.5m altitude. Wide views help you choose where to investigate; reserve flight
time for close inspections and home return. Read `drone status` before takeoff
for the actual budget and evidence requirements.

Investigate provisional thermal signatures in the East and North survey regions
using the simulated drone. Identify distinct supported candidates, revisit
uncertain evidence when useful, return to the launch position, land, and submit
an evidence-linked JSON report. Read `drone status` before takeoff for the
flight-time budget and thermal condition. You have at most 64 tool actions.
The simulator keeps flying during your analysis.

## Reliable execution

Prepare and test your code **before takeoff**. Model calls and debugging can
consume most of the flight budget. Prefer collecting evidence in one bounded
local flight program, landing, and then using your image tools to analyse the
saved frames and write the final report. Landing stops the flight clock;
post-flight analysis and report preparation do not spend the flight budget.

The public `drone collect --plan /workspace/plan.json` tool executes your chosen
waypoints, captures paired images at each reached position, reserves a
conservative return margin, and lands automatically. It does not select a route,
detect candidates, submit a report, or access hidden information. Write a plan:
`{"waypoints":[{"x":8,"y":1,"altitude":4}]}`. Choose enough positions to inspect
the requested regions within the budget; the example is only one waypoint.
Close captures need altitude at most 4.5m. A second view at least 1m away can
support the same sighting. The executor may skip remaining waypoints to return
on time. Read its saved `/workspace/evidence/mission.json` index for actual
captures, pose, calibration, local image paths, and the landing receipt.

Before collecting, verify `python3 -c "import camera_geometry, flight_plan"`.
After collection, inspect the saved thermal/RGB images, associate sightings
using the public geometry module, write normalized 0..1000 boxes, and submit
with `drone submit --report /workspace/report.json`. Do not spend flight time
writing detection code or waiting for a model to analyse images. Direct drone
tools remain available for a custom adaptive controller instead of `collect`.

Local coordinates are metres east (+x), north (+y), and altitude above flat
ground. Home is (0,0). Flight bounds are x/y -24..24 and altitude 2..10m. East is
x=5..15,y=-5..5; North is x=-5..5,y=7..17. You choose the route and heights;
there are no preassigned target coordinates. Thermal and RGB frames are paired
at their capture timestamp and pose. Thermal imagery is an experimental
DuMux-derived composed field, not validated real-world mine detection.

Use the installed `drone` CLI (run `drone --help`):
- `drone status`
- `drone launch --altitude 5`
- `drone goto --x 8 --y 1 --altitude 4` starts motion immediately.
- `drone observe` saves paired images under `/workspace/evidence/`. Inspect the
  actual thermal/RGB images using your image-reading tool. You can retarget
  before arrival, or `drone wait --seconds 2` to advance a short interval.
- `drone hover` holds the current position while the flight clock continues.
- `drone land` returns home and lands. Budget enough time for the return.
- `drone submit --report /workspace/report.json` seals the report permanently.

Report schema:
```
{"candidates":[{"notes":"brief visible evidence and uncertainty",
 "evidence":[{"observation_id":"id returned by observe",
 "bbox":{"x_min":100,"y_min":200,"x_max":180,"y_max":280}}]}]}
```
Box coordinates are normalized 0..1000 in the thermal image, with positive
width/height and area at most one quarter of the image. Group repeated
sightings of one candidate into one report entry. Each evidence item must refer
to an actual saved image. Give one box per candidate per capture; different
boxes in the same capture cannot represent one candidate. Associated evidence
must describe a consistent ground location; retrieval uses its median projected
position. Do not use full-frame boxes or enumerate guesses.
Return an empty candidates list if the observations do not support candidates.

Observations include camera intrinsics, mount and capture pose. The public
`camera_geometry` Python module provides `pixel_to_ground(observation,u,v)` and
`ground_to_pixel(observation,x,y)` for normalized pixels and local ground
coordinates. Use this calibration to associate repeated views; it contains no
scene or target information. The thermal display is white-hot. The synthetic
inclusion is an inert polymer cylinder approximately 0.24m across; its visible
thermal footprint can be larger and depends on the reported thermal condition.
Signatures can be warmer or cooler than nearby ground. Assess local contrast
and use additional views when the evidence is uncertain.

You may retain uninspected sightings under an optional `provisional_candidates`
list; only `candidates` is graded. Retrieval credit requires a correctly localized
close image (capture altitude at most 4.5m). The independent-view bonus requires
capture positions at least 1m apart horizontally; changing height alone does
not earn this bonus.

Your score combines one-to-one retrieval of inserted simulated thermal targets,
coverage of the two requested regions, and independent-view evidence for true
matches. Unmatched and duplicate claims reduce retrieval credit. Home return,
a submitted report and flight-budget compliance are required. No activity,
early submission without captured evidence, or an unfinished mission scores 0.
All decisions must use public observations; simulator files and private target
state are in a separate service. File edits in your workspace cannot change
flight telemetry, captures, candidate eligibility or the verifier's ledger.

Finish the flight with `drone land`, then submit the report. After reading the
submission result, give a short final text response with the outcome.
