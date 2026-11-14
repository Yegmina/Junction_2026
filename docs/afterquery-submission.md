# AeroRozum: find the evidence, make it home

**One-line description:** A Harbor environment where an AI coding agent pilots a
Gazebo drone to investigate thermal anomalies and deliver an evidence-linked
report before its flight budget runs out.

**Demo and downloads:** https://136.115.179.240/presentation/submission/index.html

**Source:** https://github.com/Yegmina/Junction_2026/tree/codex/harbor-investigation

## What the agent does

The agent receives a survey request for two regions and chooses arbitrary local
`goto(x, y, altitude)` commands. It inspects paired thermal/RGB captures, decides
which sightings deserve a closer look, associates repeated views, returns home,
lands and submits a report linked to saved image IDs and boxes. The drone keeps
flying while the agent thinks. The agent can prepare a controller before
launching and analyze saved evidence after landing.

The new task's default budget is 75 simulated flight seconds. Wide scans cover
more ground, but positive claims need a correctly localized capture at ≤4.5m.
Independent-view credit requires at least 1m horizontal separation. Thermal
contrast changes with the physical day/night state. Uncertainty, revisit costs,
coverage and the return trip compete for the same budget.

## Why it is an agent challenge

Recognizing an anomaly is insufficient. The agent must choose what to inspect
next, manage motion during inference, merge evidence across changing viewpoints,
and finish the mission. The public-image reference scored **0.733/1 on each of
two fresh instances**, returning within budget. In a clean larger-allowance
Argon run on the same first instance, the model retrieved two targets with no
false claims, covered both regions and returned home—but landed at
**81.032 seconds**, beyond the 75-second budget, and earned **zero**.

Other run outcomes are published in the complete result snapshot. The earlier
baseline task is easier and has clean frontier scores up to 1.0. First short-cap
attempts that stopped before takeoff are labelled separately from genuine
flight outcomes; larger-allowance retries keep the same task and instances.

## Scoring and integrity

The verifier writes one scalar in `[0,1]` to `/logs/verifier/reward.txt`:

`0.65 × one-to-one retrieval F2 + 0.20 × captured-region coverage + 0.15 × independent-view fraction`.

A report, airborne evidence, home return and budget compliance are required.
Duplicate and unsupported claims reduce retrieval credit. A separate simulator
owns telemetry, capture identities, image hashes and target state. Five actual
native Harbor controls—no-op, early report, reward-file spoof, forged capture
with local image mutation, and private-state access—each earned zero. Unit tests
also exercise inconsistent association, duplicates, invalid boxes and timing.

## Supporting post-training experiment

Gemma 4 12B uses language and visual-projection LoRA with SFT and TRL GRPO on
602 real thermal examples with preserved label provenance. The initial visual
GRPO model scored **0.601 development F2**; a direct continuation with **96
additional GRPO updates and no extra SFT** reached **0.650**, versus Carbon's
0.642 on the same 120 development images. The student made fewer false claims
and missed more objects. This is single-seed development selection with
different inference allowances, not independent superiority evidence.

The checkpoint was frozen before a separate 182-image zone-held-out comparison
against its direct parent, untouched base and three Gemini controls. Completed
results are listed separately on the demo page. Test scores never choose a
checkpoint or change training. Perception F2 and native mission reward are
different measurements.

## Reproduction and scope

The prepared judging ZIP includes `instruction.md`, `task.toml`,
`environment/Dockerfile`, `tests/`, `solution/`, scene assets, runtime sources,
actual Harbor model logs, rewards and checksums. Its independent extraction and
fresh native smoke test preserve the original task checksum. Tested stack:
Harbor 0.26.0, OpenCode 1.19.5, Docker and Compose. Evaluators supply model
credentials; the environment itself needs CPU rendering, not a GPU.

```bash
harbor run -p ./thermal-investigation-budgeted -a oracle
harbor run -p ./thermal-investigation-budgeted -a opencode -m <provider/model>
```

The thermal response comes from a numerically admitted full-day DuMux
present/absent pair for one inert cylinder in homogeneous soil, composed with
photographic albedo and recorded thermal microtexture. Local physical admission
does not establish full heterogeneous-field acceptance or real mine-detection
accuracy. The original larger three-seed research protocol remains separate.

Presentation videos use genuine recorded camera pairs or tool-state replays,
with compression clearly labelled. No unobserved drone viewpoints or private
reference overlays are invented.
