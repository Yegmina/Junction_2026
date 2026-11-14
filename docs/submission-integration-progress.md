# AeroRozum submission evidence — 14 November 2026

The Harbor environment, scalar verifier, native OpenCode model playthroughs,
reference controller and shortcut controls run on the four-A100 VM. The full
repository check passed **320 tests**. The portable baseline archive passed an
independent extraction, file-hash check and fresh native Harbor smoke run.
The newer close-inspection task is being evaluated on fresh instances.
Frontier superiority is not established.

## Runnable submission

- [Verified baseline ZIP](https://136.115.179.240/presentation/submission/aerorozum-harbor-candidate-v3.zip)
- [Independent portability receipt](https://136.115.179.240/presentation/submission/candidate-v3-portability.json)
- [Presentation and result browser](https://136.115.179.240/presentation/submission/index.html)
- [Branch](https://github.com/Yegmina/Junction_2026/tree/codex/harbor-investigation)

The baseline ZIP is 407,309,079 bytes. SHA-256:
`46790a32e7d3fdc3e5bd645bb4c5782a13ee81c033c1e99c36a721d1c910522d`.
All 254 bundled file hashes passed independent verification. Extracting with
Unix modes preserved produced the same full task checksum and evaluation
contract as frozen source-v8. A fresh-seed native no-op run exited cleanly at
zero. Existing Docker layer caches were used; no private source or bucket mount
was required. All scene assets and runtime sources are included.

Tested stack: Harbor 0.26.0, OpenCode 1.19.5, Docker 29.1.3, Compose 2.39.4.
Evaluators supply their own model credentials. Base weights, original flight
recordings, credentials and agent caches do not enter the archive. Public event
logs preserve tools, feedback and final text; private reasoning events are
explicitly omitted. Inline image attachments are saved as byte-identical,
SHA-256-named images. Exact private-key scanning covers binary files too.

## Investigation environment

The agent sends arbitrary local `goto(x, y, altitude)` commands, inspects paired
640×512 thermal/RGB captures, associates sightings, reserves time to return,
lands and submits an evidence-linked report. Native Gazebo dynamics run in a
separate simulator container, which owns the private reference and immutable
capture ledger. The root verifier writes `/logs/verifier/reward.txt`.

The scalar is `0.65 F2 + 0.20 coverage + 0.15 independent-view fraction`, gated by
a report, airborne evidence, home return, flight-time and action compliance.
One-to-one retrieval and duplicate penalties discourage unsupported claims;
median ground association rejects inconsistent evidence. Local workspace edits
cannot change authoritative simulator state.

The baseline task is [thermal-investigation](../harbor/thermal-investigation/README.md).
The separate [close-inspection task](../harbor/thermal-investigation-budgeted/README.md)
uses a **75-second dawn flight budget**. Each positive claim needs a correctly
localized saved image captured at **≤4.5m**. Independent-view credit requires
**≥1m horizontal separation**; changing height alone earns none. Wide imagery
helps select inspections. Optional provisional sightings are retained without
being counted as positive claims. Limits are public before takeoff. Fresh
reference/frontier/shortcut runs are required because the rules differ.

## Measured native mission results

| Task / condition | Agent | Reward | Execution |
|---|---|---:|---|
| Baseline dawn / 180s | Old public-image reference v7 | 0.20000 | Clean; no matches |
| Baseline dawn / 180s | Improved public-image reference v8 | 0.74000 | Clean; 3 matched, 10 false claims |
| Baseline cooling / 90s | Reference v8 | 0.79167 | Clean; 3 matched, 3 false claims |
| Baseline warming / 180s | Reference v8 | 0.78750 | Clean |
| Baseline dawn / 180s | Gemini Argon | 0.959375 | Clean |
| Baseline cooling / 90s | Gemini Carbon | 1.00000 | Clean; 3 matched, no false claims |
| Baseline v8 | No-op / early airborne report / reward-file spoof | 0 / 0 / 0 | All clean |
| Baseline native-tools v2 | Untouched Gemma 12B | 0 | Clean; incomplete mission |
| Baseline dawn / 180s | Refined Gemma, thinking disabled | 0 | Clean; incomplete mission |
| Close inspection, seed 730171 | Public-image reference | 0.733333 | Clean; 2 matched, 1 false, 1 missed; 71.484/75s |
| Close inspection | No-op / early report / reward-file spoof | 0 / 0 / 0 | Completed controls; see machine-readable execution status |

Frozen baseline source-v8 full task checksum:
`603755a222d1e2c988fe46cdfe75fc13483640edf1facdf5625e53e566a32fe6`.
The baseline v7/v8 cooling comparison has an explicit audit of identical
instructions, environment/assets, verifier, resources, native packages and
sealed instance. Only the private reference implementation, README and version
changed. Its shared evaluation-contract hash is
`aa4490f4c0367447abeb342ee36644e68292d54f789b5331e5df77a1e038ca0f`.
This audit does **not** make baseline and close-inspection rules comparable.

Earlier warm Gemini runs earned 1.0 but ended with OpenCode Git-snapshot EPIPE;
retain their process warnings. Later clean frontier runs supersede them as the
submission's clean model evidence. Native refined Gemma v2 and the initial
visual-native trial hit the old 240-second HTTP timeout; exclude these errors
from model-skill comparisons. Transport timeout is now 900 seconds. Canonical
Gemma native tool formatting preserves roles, tool results and images. A new
frozen visual-checkpoint close-inspection trial uses thinking disabled, the
same task and fresh seed, without changing any in-flight checkpoint.

## Perception post-training

All reported image scores use the same **120-image development set**: 88
positive reference objects plus 32 screened negative images. Training has 602
images: 162 original-manual, 320 reviewed-AI and 120 user-accepted-model records.
Provenance is retained. Test zones 8/9 remain sealed. One seed, BF16 Gemma 4 12B;
these are development results, not field validation or a multi-seed causal study.

| Development image control | TP / FP / FN | Micro-F2 | Negative-image false alarms |
|---|---|---:|---:|
| Untouched Gemma | 4 / 5 / 84 | 0.05540 | See recorded summary |
| Language-only selected SFT | See source | 0.26385 | See recorded summary |
| Language-only selected GRPO | 29 / 9 / 59 | 0.37179 | 3/32 |
| Continued language-only GRPO | See source | 0.45113 | See recorded summary |
| Visual-projection SFT | 41 / 10 / 47 | 0.50868 | See recorded summary |
| Visual-projection GRPO | 50 / 14 / 38 | 0.60096 | 3/32 |
| Gemini Carbon | 60 / 55 / 28 | 0.64240 | 14/32 |
| Gemini Argon | 60 / 88 / 28 | 0.60000 | 20/32 |
| Robotics teacher | See latest snapshot | 0.61060 | See latest snapshot |

The visual model has fewer false alarms but misses more objects than the
strongest frontier. Frontier controls share visible prompt/images with Gemma,
but use native JSON and 8192 total output tokens including default reasoning;
Gemma uses greedy 192-token output. This is not matched inference compute.
Image-level F2 and native mission reward are separate measurements.

The visual projection's gradients are verified in the actual
`embed_vision.multimodal_embedder.embedding_projection` module. The selected
adapter hash is `f9bd1c6e2913f6f8446d995624d1e9f953ddce281bc03d1f1e99d7c6eac215e3`.
The extra +240 SFT continuation regressed to 0.08219 SFT / 0.09615 GRPO and was
not promoted. The +480 SFT continuation and a separate up-to-96-step **GRPO-only**
continuation are running. The latter keeps completed SFT+GRPO lineage, verifies
parent weight hashes, starts a fresh optimizer and adds no SFT. Four consecutive
zero-variance task-reward groups stop training; nonzero KL gradients alone do
not establish learning. The geometric-augmentation experiment was cancelled
before starting and has no result. Its tested implementation remains available.

Actual completed results, source-result hashes, protocol differences and
process errors are exported to [the measurement snapshot](evidence/submission-measurements-latest.json).
Complete paid responses, optimizer metrics, adapters and original jobs are
retained privately under `/opt/aerorozum/artifacts` and `/opt/aerorozum/harbor/jobs`,
with terminal evidence archived to the existing private GCS bucket.

## Physics and presentation scope

The qualified full-day DuMux present/absent pair models one inert polymer
cylinder in homogeneous flat soil under uniform forcing. Certificate checks
passed within that local scope. The runtime factory draws private insertion
positions and writes fields and truth together; active camera-field hashes must
match the private ledger. Physical contrast is not amplified. Warming, cooling,
dawn and late-warming snapshots come from the same admitted trajectory.

The composed 70m field uses photographic albedo and recorded thermal
microtexture, with explicit sensor noise and PSF. It retains
`research_accepted: false`: local-pair admission does not qualify a full
heterogeneous-domain solve. Original full 184/30k/three-seed research gates are
not complete and are not waived by this pilot benchmark. No real-mine accuracy
claim is made.

Presentation replay frames are actual saved camera pairs checked against the
sealed image hashes. Blue annotations show submitted claims, not private truth.
Videos cut between captures, two seconds per pair; they are not continuous
flight recordings. The separate coordinate-agent video demonstrates actual
retargeting during flight. Real thermal label excerpts preserve source times,
prompt versions, frame identities and lower-confidence candidates.
