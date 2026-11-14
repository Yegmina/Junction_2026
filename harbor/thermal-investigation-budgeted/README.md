# Thermal investigation with close inspection

A drone investigator has 75 simulated flight seconds to survey two regions,
choose which thermal sightings deserve a close inspection, return home, land,
and submit an evidence-linked report. Wide views cover ground quickly. Positive
claims require a saved thermal image captured at or below 4.5m, so the agent
must reserve time to inspect and return. The aircraft keeps flying during
inference. Uninspected sightings can remain in `provisional_candidates`.

The default scene is dawn, one elapsed hour into the admitted physical source
trajectory. Surface contrast is weaker than the warm control. The environment
uses native Gazebo dynamics and paired 640×512 thermal/RGB observations. It has
no real-aircraft connection.

The verifier writes one scalar to `/logs/verifier/reward.txt`:

```
0.65 * F2 + 0.20 * captured_region_coverage + 0.15 * independent_view_target_fraction
```

A candidate can match an insertion only when its median projected location is
within 0.65m and a correctly localized close image supports it. Evidence must
be consistent within 1.3m; there is one box per candidate per capture. Unmatched
and duplicate claims count as false positives. Independent-view credit requires
capture positions at least 1m apart horizontally. Changing height alone earns
no such bonus. A report, airborne evidence, home return and compliance with the
flight/action budgets are required; incomplete missions score zero. The action
limit is 64 and the agent wall-clock limit is 1800 seconds. The flight clock
stops after landing, allowing report preparation from saved evidence.

Use Harbor 0.26.0, OpenCode 1.19.5, Docker and Compose. With prepared assets:

```
python cloud/run_harbor_trial.py --task ./harbor/thermal-investigation-budgeted --job-name budgeted-01 --phase dawn --budget 75 --seed 730171
harbor run -p ./harbor/thermal-investigation-budgeted -a oracle
```

The prepared submission archive includes assets and runtime sources. A Git
checkout requires staging them first. Supply your model API credentials on the
host; none belong in the task. `drone status` exposes actual limits, regions,
thermal condition and evidence requirements before takeoff. The public
`camera_geometry` module projects captured pixels using actual pose/calibration.
The private simulator holds insertion identities and immutable camera evidence.

The reference uses public images and telemetry: wide scans, RGB background
fitting, radial-shape checks, close inspections within each region, and an
explicit reserve for the remaining survey and home return. It receives no
insertion locations or counts. Its performance must be established by actual
native Harbor runs.

This task changes the operational evidence requirements and flight budget.
Scores from the earlier thermal-investigation task are not results on this
one. Fresh reference, model and shortcut controls are required.

The local DuMux response pair passed numerical admission for one cylinder in
homogeneous soil under uniform forcing. The 70m scene composes that response
with assigned photographic albedo and recorded thermal microtexture. Physical
contrast, sensor noise, resolution and native dynamics are unchanged. Full
heterogeneous-field acceptance and real mine-detection validation are not
claimed. All findings refer to inserted synthetic thermal signatures.
