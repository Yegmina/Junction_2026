# Submission integration — 14 November 2026

Target: a self-contained Harbor task, a real OpenCode playthrough with scalar
reward, reference and shortcut controls, reproducible evidence, and measured
post-training comparison. The objective of outperforming frontier models is
unproven until the trained checkpoint and frontier run the same held-out tasks.

Three initial GPU workers completed 120 SFT updates each, then dense-IoU GRPO
with early stopping for zero task-reward variation. Each used 482 reviewed
positive training images plus 120 explicit screened negatives. Validation and
sealed test zones never entered training. All three share initialization,
data and seed; learning rate is varied. Development selection is therefore
explicit hyperparameter tuning, not an independent test result.

| SFT learning rate | SFT F2 | After GRPO F2 | GRPO true matches | GRPO false matches |
|---|---:|---:|---:|---:|
| 0.000002 | 0.05540 | 0.05540 | 4 | 5 |
| 0.000010 | 0.09589 | 0.09589 | 7 | 6 |
| 0.000030 | 0.11019 | 0.12363 | 9 | 3 |

These numbers use the frozen 120-image development set (88 positives, 32
negatives), not the earlier positive-only table. Each GRPO run logged five
steps with task-reward variation at one measured step. A one-match change on
one seed is not a robust causal RL benefit. All raw generations and checkpoints
remain under `/opt/aerorozum/artifacts/submission-refinement-v1/` and
`training/submission-refinement-v1/` on the existing VM.

A second bounded sweep warm-starts the best GRPO checkpoint and compares 120,
240 and 480 additional SFT updates at learning rate 0.00001, followed by GRPO
and the same development scoring. GPUs 1–3 run these arms independently; GPU 0
retains the private demo inference server. The experiment ledger is mirrored
to the deployed dashboard. The first sweep's complete results remain intact.

Harbor 0.26.0 and Docker Compose v2.39.4 were installed on the VM. The native
Gazebo image was built; lossless scene assets total about 178MB. The task is
`harbor/thermal-investigation/`. Eight focused adversarial checks pass. A
reference trial was launched under the actual Harbor CLI; its authoritative
state is `/opt/aerorozum/harbor/jobs/reference-v1/`. A launched trial is not a
completed playthrough. The full local test environment is being recreated
because the external SSD holding its previous virtualenv is disconnected.
