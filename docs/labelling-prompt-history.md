# Thermal label prompt history

The original worker was safely drained at **2026-11-14 13:53 UTC**. Its paid transports settled without cancellation before it stopped. The unfiltered sequential follow-ups are blocked.

- **thermal-v1-conservative**: exact original prompt, retained in `aerorozum/label_prompts.py`. The historical paid labels remain associated with this prompt.
- **thermal-v2-recall-calibrated**: proposed replacement. Removes the two requested prohibitive passages, includes small/faint/obscured/hot/cold/non-circular signatures, and explicitly assigns lower confidence to weak or ambiguous evidence. Acceptance stays at0.60; lower-scored proposals remain uncertain.

V2 was selected for the requested lower-confidence candidate-review behavior after a paired88-image development comparison. V1 marginally wins the automatic-label F2 comparison at0.60; V2 finds more reviewed objects among uncertain proposals. The selected V2 worker started on the VM at2026-11-14 14:06:57 UTC. Actual paid-result and canonical cloud readback verified its exact prompt hash; no accepted-recall improvement is claimed.

Every new paid label, checkpoint and frame audit includes `label_prompt_version` and the exact SHA256 of the prompt. Provider records also retain the actual complete prompt. Old cache reuse preserves its original key and v1 provenance; changing the prompt alone never schedules a paid relabel.

The complete old-label catalog is `labelling/prompt-history/index.json` under the artifact root; `registry.json` records both exact texts and hashes. The indexed records and registry are also stored immutably under the source bucket's `robotics-er2-labels/5fps-v1/progress/` prefix. The `5fps-v1` path denotes the sampling/cache archive, not a claim that all future prompts are v1.

Production requires a frozen airborne eligibility manifest derived from aligned receipt timestamps and actual flight telemetry. Known grounded frames and uncertain/unmatched timing are excluded from new model requests. Grounded-frame labels already paid for are preserved as historical evidence.

The input-token limiter now reserves in-flight tokens and reconciles against actual measured usage, maintaining the existing120RPM,500k inputTPM and16-concurrent caps. Those are local limits, not a claim about the provider account's maximum quota.
