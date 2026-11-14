# AeroRozum submission evidence — 14 November 2026

The Harbor environment, scalar verifier, native OpenCode model playthroughs,
reference controller and shortcut controls run on the four-A100 VM. The full
repository check passed **331 tests**. The portable baseline archive passed an
independent extraction, file-hash check and fresh native Harbor smoke run.
The newer close-inspection task is being evaluated on fresh instances.
Frontier superiority is not established.

## Runnable submission

- [Verified close-inspection ZIP v4](https://136.115.179.240/presentation/submission/aerorozum-harbor-close-inspection-v4.zip)
- [Independent close-inspection portability receipt](https://136.115.179.240/presentation/submission/close-inspection-v4-portability.json)
- [Verified baseline ZIP](https://136.115.179.240/presentation/submission/aerorozum-harbor-candidate-v3.zip)
- [Independent portability receipt](https://136.115.179.240/presentation/submission/candidate-v3-portability.json)
- [Presentation and result browser](https://136.115.179.240/presentation/submission/index.html)
- [Branch](https://github.com/Yegmina/Junction_2026/tree/codex/harbor-investigation)

The close-inspection v4 ZIP is **364,945,349 bytes**, with 156 verified file
hashes. SHA-256:
`2c04acba77cb72585f0568a01b99b8d06501e9cbaf0ddad0d2668e6a281ab468`.
Its extracted task reproduced full Harbor checksum
`ed718fd1747779020d569898ef9a4babae10e3ad1e96c4dfb4af1cdfe46e1a7c`.
A clean native smoke trial used fresh seed 930117, regenerated all four physical
fields and retained the actual dawn/75s defaults. The archive includes both
reference instances, a clean Gemma flight attempt, first short-cap model logs
and all five native adversarial controls. Later larger-allowance model runs
remain separate until terminal and verified.

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
| Close inspection, seed 730172 | Public-image reference | 0.733333 | Clean; 2 matched, 1 false, 1 missed; 73.104/75s |
| Close inspection, seed 730171 | Gemini Argon, 32k output allowance | 0 | Clean; 2 matched, no false claims, full coverage and home return; landed at 81.032/75s |
| Close inspection, seed 730171 | Visual Gemma 12B, no thinking | 0 | Clean; 3 captures, 9 actions, stopped without return or report |
| Close inspection | No-op / early report / reward-file spoof | 0 / 0 / 0 | Completed controls; see machine-readable execution status |
| Close inspection | Forged capture + local image mutation | 0 | Completed flight and landed; forged ID rejected, sealed camera bytes unchanged |
| Close inspection | Private-state access probe | 0 | Private files absent in agent container; pre-seal ledger returned HTTP 404 |

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

The development image scores below use the same **120-image development set**: 88
positive reference objects plus 32 screened negative images. Training has 602
images: 162 original-manual, 320 reviewed-AI and 120 user-accepted-model records.
Provenance is retained. Test zones 8/9 stay excluded from training and development selection; the final-hour section records their one-time evaluation opening. One seed, BF16 Gemma 4 12B;
these are development results, not field validation or a multi-seed causal study.

| Development image control | TP / FP / FN | Micro-F2 | Negative-image false alarms |
|---|---|---:|---:|
| Untouched Gemma | 4 / 5 / 84 | 0.05540 | See recorded summary |
| Language-only selected SFT | See source | 0.26385 | See recorded summary |
| Language-only selected GRPO | 29 / 9 / 59 | 0.37179 | 3/32 |
| Continued language-only GRPO | See source | 0.45113 | See recorded summary |
| Visual-projection SFT | 41 / 10 / 47 | 0.50868 | See recorded summary |
| Visual-projection GRPO | 50 / 14 / 38 | 0.60096 | 3/32 |
| Visual parent +96 GRPO only | 55 / 16 / 33 | 0.65012 | 5/32 |
| Visual parent +480 SFT, then GRPO | 52 / 23 / 36 | 0.60890 | 5/32 |
| Gemini Carbon | 60 / 55 / 28 | 0.64240 | 14/32 |
| Gemini Argon | 60 / 88 / 28 | 0.60000 | 20/32 |
| Robotics teacher | See latest snapshot | 0.61060 | See latest snapshot |

The selected 0.65012 development F2 exceeds Carbon's 0.64240 on these images,
with fewer false claims and more missed objects. This is a small development
difference after tuning, not independent superiority evidence. Frontier controls share visible prompt/images with Gemma,
but use native JSON and 8192 total output tokens including default reasoning;
Gemma uses greedy 192-token output. This is not matched inference compute.
Image-level F2 and native mission reward are separate measurements.

The visual projection's gradients are verified in the actual
`embed_vision.multimodal_embedder.embedding_projection` module. The selected
adapter hash is `f9bd1c6e2913f6f8446d995624d1e9f953ddce281bc03d1f1e99d7c6eac215e3`.
The extra +240 SFT continuation regressed to 0.08219 SFT / 0.09615 GRPO and was
not promoted. The +480 SFT continuation completed at 0.57279 SFT / 0.60890 GRPO.
The separate **GRPO-only** continuation completed all 96 requested updates and
reached 0.65012. It keeps completed SFT+GRPO lineage, verifies
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

## Final-hour held-out comparison

Before opening test zones, checkpoint selection was frozen to maximum completed
GRPO development F2, then lower negative-image false alarms, then valid JSON.
The selected additional-96-GRPO model scored 0.650118 on development data. Its
receipt binds the actual result and adapter hashes. Test images remain labelled
`test`; disjoint training/development hashes and duplicate test images are
excluded explicitly. The frozen manifest contains **182 zone-8/9 images**.

The selected Gemma, its immutable direct parent, the untouched base, Carbon,
Argon and Robotics are evaluated separately. The base/parent controls were
recorded before inspecting any test score; the selected checkpoint remains
fixed. All three local models use the same single-image greedy 192-token
program on separate free GPUs. Gemini retains the previously specified native
JSON/8192-token protocol; this is not matched inference compute.

Robotics returned an unquoted `Patch` token in a box coordinate. The initial
strict-provider wrapper stopped the evaluation. An explicit parser amendment
now scores the first retained raw completion, including malformed JSON, using
the same invalid-prediction rule as Gemma. Valid paid completions and that
malformed response are reused without new requests. Transport errors with no
completion still stop evaluation. Ordinary production labelling continues to
require valid JSON. Both failure and recovery logs remain preserved.

No test score chooses a checkpoint or modifies training. The full planned
three-seed/184-case/30k-observation study remains separate and unfinished.
