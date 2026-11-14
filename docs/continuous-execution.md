# Continuous execution

Open <http://127.0.0.1:18900/>. Pipeline is the default page. It shows measured
CPU/GPU history, physical reference progress, actual field previews, candidate
examples, immutable batch queues, learning signals and development results.

## Data flow

1. Videos are sampled at five source frames per second. Exact Robotics ER 2
   responses are cached and archived. Confidence >=0.60 retains machine-label
   provenance and does not imply human review.
2. The ingester verifies hashes, recording/zone holdouts and uncertainty. It
   publishes eligible records, reviewed ZIP anchors and train-only backgrounds.
3. Parallel DuMux cases must pass actual numerical admission. Completion alone
   never accepts a field. The first accepted pair can release an auxiliary
   synthetic shard without waiting for all 184 planned trajectories.
4. Native sensor GSD, crop geometry and certificate hashes are retained. A metre
   patch is never stretched into a full scene. Auxiliary crops are capped at 25%
   of a training batch; realistic background hybrids require separate validation.
5. New eligible training arrivals freeze a batch with anchors and replay.
   Held-out arrivals alone cannot trigger training. Seeds 17/43/91 queue
   SFT -> GRPO -> candidate development comparison -> shared parent comparison;
   SFT checkpoints also receive a separate utility comparison.
6. Promotion requires the configured checks. Predictions, checkpoints, failed
   results and reward-variance limitations remain inspectable.

The full 184-trajectory / 30,000-image / three-seed protocol remains separate
and incomplete. Cropped auxiliary evidence does not establish a validated
64 m campaign generator or calibrated realistic full-flight images.

## Resources

The existing Spot VM has 48 vCPUs and four A100 40 GB GPUs. Its runtime
`system.slice` CPU quota is 4300% (43 cores, approximately 89.6% of the host),
with a 300 GiB memory high-water mark. Telemetry reports actual usage separately
from allocations. High utilization is a scheduling target, not busywork.

The full-BF16 single-GPU forward/backward/GRPO probe passed. Current training
uses four exclusive single-GPU groups, microbatch 2 and accumulation 2
(effective batch 4), with measured peak reservation around 25.3 GB per GPU.
Active optimizer work is preserved during controller restarts.

Local dashboard, observer tunnel/mirror and verified input bridge now have
persistent user-service definitions. After a host reboot, confirm the external
SSD is mounted before treating local artifacts as available. A connected observer
does not imply an active solver or optimizer: inspect actual queue and telemetry.

Scheduled termination is **2026-11-15 06:30:50 UTC / 08:30:50 Helsinki**, subject
to earlier Spot preemption. Checkpoints are archived outside the VM.

## Inspect execution

VM services include `aerorozum-stream-pipeline.service`, physical workers and
uploaders, and `aerorozum-research-observer.service`. The observer binds only to
VM loopback port 18920. A private SSH forward and local mirror carry measured
state and published previews to the dashboard; no public telemetry port is open.

Under the artifact root:

- `pipeline/worker.json`: mirrored physical/controller progress.
- `pipeline/telemetry.json`: utilization, RAM and disk measurements.
- `pipeline/inputs/real.json`: complete verified input index; browser gets counts.
- `pipeline/inputs/backgrounds.json`: screened train-only recorded textures.
- `pipeline/media/`: published physical and training previews.
- `cloud/health.json`: unresolved issues, including boot-2 PCR7 baseline mismatch.
- `cloud/20h-extension-verification.json`: actual scheduling readback.

Local services can be inspected with:

```sh
systemctl --user status aerorozum-research-api.service
systemctl --user status aerorozum-research-observer-tunnel.service
systemctl --user status aerorozum-research-observer-mirror.service
systemctl --user status aerorozum-real-data-ingester.service
systemctl --user status aerorozum-label-cache-recovery.service
```

The API has bounded graceful shutdown so event streams cannot indefinitely
stall updates. Opening Pipeline does not start paid calls or provision machines.
See [paid-label-archive.md](paid-label-archive.md), [gpu-physics.md](gpu-physics.md)
and [presentation/journal.md](presentation/journal.md) for receipts and screenshots.
