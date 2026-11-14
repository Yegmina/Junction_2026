# Submission integration — 14 November 2026

The native Harbor task, scalar verifier, clean OpenCode model playthroughs and
reference/shortcut controls run. A prepared portable archive exists; final
task-version packaging and harder-scenario evaluation remain. Frontier
superiority remains unproven. The latest full repository check passed 302 tests.

## Latest measured state

The strongest completed visual-projection arm reached **0.60096 development
F2** after GRPO, compared with its corresponding SFT **0.50868**. This is a
different adapter scope from the six language-only arms below. Continued
language-only training reached 0.45113. The untouched base scored 0.05540.
All use the same 120 development images; no frozen test zones were accessed.

| Development image control | TP / FP / FN | Micro-F2 | Negative-image false-alarm rate |
|---|---|---:|---:|
| Gemma, visual projection SFT + GRPO | 50 / 14 / 38 | 0.60096 | 3/32 |
| Gemini Carbon | 60 / 55 / 28 | 0.64240 | 14/32 |
| Gemini Argon | 60 / 88 / 28 | 0.60000 | 20/32 |

The student has fewer false alarms, but the strongest frontier has higher F2.
Frontier controls used the same visible prompt/images and native JSON mode,
with 8192 output tokens including default reasoning. Gemma used greedy 192-token
output. These decoding differences are recorded; this is not a sealed-test or
matched-compute superiority claim. [Latest measurements](evidence/submission-measurements-latest.json)
include source hashes and individual run details.

Two bounded continuations of the visual adapter are now scheduled on GPUs 1
and 3: +240 SFT at LR 0.00001 and +480 at LR 0.00003, each followed by up to 96
GRPO updates with the existing zero-variance stop. Parent adapter SHA-256 is
`f9bd1c6e2913f6f8446d995624d1e9f953ddce281bc03d1f1e99d7c6eac215e3`.
These remain development selection; starting jobs does not establish results.

An augmentation continuation is queued after the base native-tool trial. It
expands each validated training image into identity/horizontal/vertical/both
flips with the corresponding normalized-box transforms. Original labels and
provenance stay unchanged. The materialization receipt distinguishes derivative
examples from independent observations; evaluation images are never augmented.
Tests reject held-out zones and tampered source/derivative checksums. The
experiment shares its parent, LR, update count and effective batch with the
unaugmented +480 continuation. It has no measured result yet.

## Measured post-training

Six bounded experiments completed on the existing four-A100 VM. The first
sweep used 482 reviewed positive training frames and 120 screened explicit
negatives, with 120 SFT updates followed by dense-localization GRPO. The second
sweep warm-started the first sweep's selected GRPO adapter and varied additional
SFT updates. Validation/test zones never entered training. Development selection
is hyperparameter tuning, not a sealed test result.

All scores below use the same 120-image development set: 88 positive reference
objects and 32 screened negative images. The earlier 88-positive-only table has
a different denominator. Seed is 17; BF16 Gemma 4 12B and language-attention LoRA
are unchanged across these six experiments.

| Arm | SFT F2 | GRPO F2 | GRPO TP / FP / FN | Informative GRPO steps |
|---|---:|---:|---|---:|
| Initial LR 0.000002 | 0.05540 | 0.05540 | 4 / 5 / 84 | 1 of 5 |
| Initial LR 0.000010 | 0.09589 | 0.09589 | 7 / 6 / 81 | 1 of 5 |
| Initial LR 0.000030 | 0.11019 | 0.12363 | 9 / 3 / 79 | 1 of 5 |
| Warm start +120 SFT | 0.12329 | 0.12363 | 9 / 3 / 79 | 1 of 5 |
| Warm start +240 SFT | 0.12195 | 0.19841 | 15 / 11 / 73 | 5 of 15 |
| Warm start +480 SFT | 0.26385 | 0.37179 | 29 / 9 / 59 | 18 of 37 |

The +480 arm's GRPO negative-image false-alarm rate is 3/32 (9.375%); valid
JSON is 118/120 (98.33%). GRPO stops after four consecutive zero task-reward
variance groups; nonzero gradients alone are not treated as task-learning
evidence. These are single-seed perception results. They do not demonstrate
native mission-policy improvement, generalization or a frontier win.

[Machine-readable measurements](evidence/measured-refinements.json) preserve
source-result hashes, dataset identity, optimizer counts, reward diagnostics,
decoding protocol and adapter paths. Raw generations and adapters remain under
`/opt/aerorozum/artifacts/submission-refinement-v{1,2}/` and
`training/submission-refinement-v{1,2}/`; a separate archive job copies both
results and training namespaces to the private GCS bucket.

The +480 GRPO adapter was promoted to private GPU inference with SHA-256
`ef19fdcda7ed07979d3aca38a197af66dfe2d68702f3b40467dbfc9b9b68506a`.
Health and generated responses now expose the loaded adapter hash. GPU 0 serves
model-chosen OpenCode tool calls. The reward-supported curriculum and
visual-projection ablations completed; latest measurements appear above.
Both visual arms have verified nonzero gradients in
the actual `embed_vision.multimodal_embedder.embedding_projection` module.
Earlier failed module-selection attempts performed no optimizer updates and
are retained as failed jobs.

## Harbor execution and reward integrity

The task is [thermal-investigation](../harbor/thermal-investigation/README.md).
Tested software: Harbor 0.26.0, task schema 1.4, OpenCode 1.19.5, Compose
v2.39.4. Native Gazebo multirotor dynamics and paired 640×512 cameras run in a
separate simulator container. The agent receives public tools and captured
evidence; the private reference, evidence store and grading state stay in the
sidecar. The verifier writes `/logs/verifier/reward.txt`.

The task checks real flight time, return/landing, action budget, evidence
identity, one-to-one retrieval, duplicate claims, coverage and independent
views. Candidate evidence must describe a consistent ground location; copying
unrelated region-wide guesses into every candidate is rejected. Agent edits to
local reward files are cleared by a root-owned verifier run.

Completed development attempts, all preserved:

| Attempt | Result | Interpretation |
|---|---|---|
| Reference v3 | 0.20, clean completion | Full coverage and home return; its residual detector submitted 16 false candidates and missed all 3 insertions. |
| OpenCode Gemini v2 | 0, nonzero agent exit | 11 observations, 99.5% coverage, returned home; no accepted final report. This is an errored playthrough, not a clean frontier baseline. |
| OpenCode Gemma v2 | 0, protocol error | Flew and read a thermal frame. The proxy rejected a valid nested OpenAI tool envelope; translation was fixed and tested. Exclude from model-skill comparisons. |
| Reference v4 | 0.89167, clean | Three true and three false candidates; full coverage and independent views. |
| OpenCode Gemini Argon v3 | 1.0, mission completed, process warning | Three true candidates, no false candidates, full coverage and independent views. OpenCode exited nonzero with Git snapshot EPIPE. |
| OpenCode Gemini Carbon v1 | 1.0, mission completed, process warning | Same successful mission outcome; same EPIPE process error. |
| No-op / airborne early report v4 | 0 / 0, clean | Shortcut controls fail the completion gate. |
| Reference v5 | 0.77391, clean | Stricter association: three true and eight false candidates. |
| Reference v5, cooling / 90 s | 0.57111, clean | Two true, four false, one missed; 80% coverage. |
| OpenCode Gemma refined v4 / base v1 | 0 / 0, clean | Old JSON bridge; one observation each, incomplete missions. |

UID/account conflicts, absent legacy private reference and initial credential
mapping failures were setup failures, not agent failures. Initializing a real
agent-owned Git workspace removed the EPIPE error; both later Gemma runs exited
cleanly, with zero mission scores. Task v0.2 also rejects inconsistent ground
associations and multiple distinct boxes from a single capture per candidate.
[Sealed-evidence reassociation audit](evidence/association-audit.json) retained
the frontier 1.0 and reference v4 0.89167 scores; this was not a new flight.

The original Gemma proxy flattened role/image history into a JSON request.
`cloud/gemma_native_chat.py` now uses the installed Gemma processor's canonical
tool template and response parser, preserving tool responses and images.
Thinking is retained privately and excluded from public chat responses. Both
base and refined checkpoints passed actual native tool-call probes. Both first
OpenCode native trials against frozen task v0.2 / seed 414331 flew and read a
thermal image, then failed with upstream HTTP 422 after another capture. Their
scores are zero with `NonZeroAgentExitCodeError`; exclude them from skill
comparisons. The first proxy discarded the upstream error detail, so the exact
failure cause is not established by those logs.

Native protocol v2 retains failed requests/error bodies privately, records
inference starts and generation failures, and preserves machine-readable
context-overflow errors for the harness. When a conversation approaches the
24576-token inference window it first omits optional cached private thinking,
then limits output to the available window; visible messages and images are
preserved. It rejects an actually oversized public conversation with an
OpenCode-recognized context error. Unit tests cover these cases and a real
installed-processor/parser fixture is run before deployment. New paired native
trials keep the same checkpoints and frozen task; their results are pending.
The refined service initially could not write its private inference log under
systemd's read-only filesystem; a narrow log-directory write exception fixed
the probe. No task or checkpoint was changed to repair this service permission.
Actual job state remains `/opt/aerorozum/harbor/jobs/`.

## Physical source and reproducibility

`physics/cases/axisymmetric-qualification-v2` contains a numerically admitted
full-day DuMux present/absent pair. Source hashes match its certificate; all
certificate checks passed within its homogeneous flat-soil/symmetric-cylinder
scope. Field validation is not performed. The generator inserts the response
and writes its private reference together; it never invents labels for a
legacy camera field. Physical contrast is not amplified.

The composed 70m photographic terrain retains `research_accepted: false`.
Admission of a local response does not qualify a heterogeneous full-domain
solve or establish real mine-detection accuracy. Existing object/3D
qualification jobs remain separate and continue without relaxed gates.

Each native task startup draws a private seed, frozen for that trial. Paired
comparisons supply a seed on the host; it is revealed only in the sealed
ledger. Checksummed thermal kernels and background arrays are fixed assets.
Freezing backgrounds removed a host/container image-processing reproducibility
failure. The native startup check reproduced seed 414331 exactly:

- Warming: `98ddb75b3785ab92e8eb6bb36545e91888cbf1a769a89088154d2a1cffed1aaa`
- Cooling: `ab8fe13bc377e39669b2c3f1e8957173f764f47763ff004b1104722eec4896ac`
- A different seed produced different fields.

Task v0.3 adds dawn (one elapsed source hour) and late-warming (11.75 hours)
snapshots from the same admitted full-day trajectory. Their weaker contrast is
physical, without arbitrary attenuation. Assets were built in an isolated v6
fork; native reference/frontier trials for these new conditions are pending.
The active camera field must belong to the private reference's checksummed
fields; the constructor rejects stale or unrelated references.

The v7 native task (v0.3) is now being evaluated on dawn/180s and cooling/90s.
A separate immutable v8 fork (v0.3.1) changes the public-image reference, not
the task's grading rules: wide 10m coverage, registered RGB background fitting,
compact thermal-residual proposals and 3m revisits. Its development-frame
diagnostic recovered the three signatures with very few proposals; that
offline check is not a native mission score. The v8 reference matrix is queued
behind the existing reference process to bound the renderer fleet.

The second portable ZIP's archive identity and all 137 bundled file hashes
passed independent verification. It is still a v0.1 development candidate,
with explicitly warned errored frontier playthroughs; do not present it as the
final current-version submission. Completed experiment/result archives are
copied to the private GCS bucket. A first archive attempt used an unsupported
CLI flag and stopped; the corrected completed-evidence archive succeeded.

The portable packager includes assets/runtime, checksums, licenses and actual
Harbor evidence. It refuses leaked credentials or disagreement between saved
Harbor rewards and reward files. Public event logs explicitly omit reasoning;
tool actions, feedback and scores remain. Clean model playthroughs and errored
development attempts are distinguished in bundle metadata. Source checkout
alone still requires staging; a prepared bundle must be tested independently
before it is described as ready to submit.
