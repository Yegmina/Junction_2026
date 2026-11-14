# Submission integration — 14 November 2026

The native Harbor task and scalar verifier run. Submission readiness still
requires a clean model playthrough, portable assets, reference/shortcut controls,
and recorded failure analysis. Frontier superiority remains unproven.

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
model-chosen OpenCode tool calls. GPUs 1–3 run a reward-supported curriculum and
explicit visual-projection ablations. Their results must be measured before
any improvement claim. Both visual arms have verified nonzero gradients in
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

UID/account conflicts, absent legacy private reference and initial credential
mapping failures were setup failures, not agent failures. A revised public-only
reference and matched Gemini/Gemma trials are running against the same private
seed, 180-second budget and revised task. Starting a trial does not establish
its completion. Actual state is `/opt/aerorozum/harbor/jobs/`.

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

The portable packager includes assets/runtime, checksums, licenses and actual
Harbor evidence. It refuses leaked credentials or disagreement between saved
Harbor rewards and reward files. Public event logs explicitly omit reasoning;
tool actions, feedback and scores remain. Clean model playthroughs and errored
development attempts are distinguished in bundle metadata. Source checkout
alone still requires staging; a prepared bundle must be tested independently
before it is described as ready to submit.
