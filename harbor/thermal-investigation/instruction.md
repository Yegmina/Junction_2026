Investigate provisional thermal signatures in the East and North survey regions
using the simulated drone. Identify distinct supported candidates, revisit
uncertain evidence when useful, return to the launch position, land, and submit
an evidence-linked JSON report. You have 180 simulated seconds of flight time
and at most 64 tool actions. The simulator keeps flying during your analysis.

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
to an actual saved image. Do not use full-frame boxes or enumerate guesses.
Return an empty candidates list if the observations do not support candidates.

Your score combines one-to-one retrieval of inserted simulated thermal targets,
coverage of the two requested regions, and independent-view evidence for true
matches. Unmatched and duplicate claims reduce retrieval credit. Home return,
a submitted report and flight-budget compliance are required. No activity,
early submission without captured evidence, or an unfinished mission scores 0.
All decisions must use public observations; simulator files and private target
state are in a separate service. File edits in your workspace cannot change
flight telemetry, captures, candidate eligibility or the verifier's ledger.
