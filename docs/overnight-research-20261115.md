# Overnight research · 15 November 2026

The user authorized at least four additional hours of cloud experiments, use of the separate hackathon Gemini key, further high-thinking Robotics tests, and safe laptop cleanup. The resumed supervision window runs until at least **04:30 UTC / 06:30 Helsinki**. New Robotics requests and native-trial dispatch stop by 05:45 UTC; already accepted native trials may issue calls through their 45-minute runtime (until 06:30 UTC). Accepted requests are allowed to settle. The existing VM's approved 48-hour STOP policy remains unchanged.

The submitted V5 package, frozen test results and deployed model checkpoint remain unchanged. Development work is on `codex/research-replication` in Yegmina/Junction_2026.

## What is running

- **Robotics development comparison:** two frozen prompts × native MINIMAL/MEDIUM/HIGH thinking × three repetitions × 120 established development images = 2,160 requests. Every arm has the same 65,536-token thinking-plus-answer allowance. First paid invalid completions count as detection failures; transport failures are reported separately. Raw output, usage, prompts and settings are archived. This is not a new independent test set.
- **HIGH-thinking flight relabeling:** starts after the paired study completes and its rate window clears. Prompt selection uses only the 88 archived zone-7 development references (mixed original-manual/reviewed-AI provenance), averaging F2 across three HIGH repetitions; v2 wins an exact tie. The original model labels remain intact under their original prefix. The new prefix is `robotics-er2-labels/high-thinking-airborne-v1` in the private flight bucket. The input set contains **14,657 unique eligible pixel/source-generation identities**, after avoiding 156 duplicates. All 354 initially missing source images were recovered directly from the read-only SSD and verified by SHA256. Fresh, unambiguous airborne telemetry is required. Ground and unknown flight intervals do not enter this pass.
- **Gemma visual replications:** independent seeds 43 and 91 replicate the fixed 480-SFT/48-GRPO recipe, followed by a declared 96-GRPO continuation only where relative task-reward variation is verified. The seed-17 matched-update SFT control finished at development F2 0.530; the earlier parent was 0.601 and the selected GRPO continuation 0.650. These are development results with differing false-alarm rates, not a universal RL gain.
- **Expanded Gemma curriculum:** all 1,200 eligible positive frames plus 1,200 source-balanced explicit zero-candidate frames. All original train-zone anchors and every eligible positive are retained. The run uses 1,920 SFT updates (3.2 effective epochs), then up to 48 GRPO updates. Gold output lengths reach 441 tokens, so this separately versioned recipe permits 512 completion tokens. It does not truncate or drop complex labels to fit the earlier 384-token allowance. The original 120-image development comparison remains unchanged. Seed 17 is on GPU 1; seeds 43/91 are queued behind their fixed-recipe replications on GPUs 2/3. No automatic model promotion occurs.
- **Native Harbor investigations:** ten fresh seeds × dawn/warming, with fixed 75-second simulated flight budgets, for Carbon, Argon and the reference agent. The currently deployed Gemma language adapter also plays the same instances through the private Docker-bridge endpoint; it has a 24,576-token context and 4,096-token output allowance, compared with 131,072/32,768 for the frontier models. Budgets differ, so comparisons are descriptive. A completed run requires Harbor's final timestamp and zero pending/running trials; the existence of `result.json` alone is insufficient. Zero rewards and infrastructure failures are preserved. No outcome-based retries.
- **Physics:** existing full-day 3D object/domain runs continue without restarting their partial work. They use the measured faster CPU backend. The full heterogeneous-scene/184-trajectory study remains incomplete. The accepted local axisymmetric reference does not validate outdoor thermal backgrounds or a 64-m terrain scene.

## Monitoring optimization

A cloud-only A100 microbenchmark verified a read-only gradient monitor that aggregates checks on the GPU instead of synchronizing once per LoRA tensor. All gradient elements are checked; NaN and both infinities are detected. Gradient tensors are unchanged, and ten paired AdamW steps produced bit-identical parameters. For 128 BF16 tensors of shape 16 × 6,144, monitor-only overhead fell from 0.1095 s to 0.01995 s (5.49×). **This is not an end-to-end training speedup claim.** The optimized source is isolated from already-running frozen experiments and will be used for the queued expanded seeds.

## Storage and evidence

Laptop free space increased from 45 MiB to about 15 GiB by removing pip/thumbnail caches and inactive Docker build cache. Code, recordings, original datasets, Docker images/containers/volumes, credentials and Codex history were preserved. RAM had approximately 48 GiB available; the immediate resource problem was storage. Large data, models, checkpoints and image generation remain on the existing VM and private buckets. The dirty SSD stays read-only; no forced write mount or repair was attempted.

The live page is **https://136.115.179.240/presentation/research-overnight/index.html**. It reads measured CPU/GPU use, actual progress and terminal Harbor rewards. It includes original thermal images with recorded model boxes at every reasoning level. Browser checks found no page errors and no horizontal overflow at 390 px. Screenshots contain actual page values; no metric was injected.

- [Desktop capture](presentation/assets/overnight-research-desktop.png)
- [Mobile capture](presentation/assets/overnight-research-mobile.png)
- [Browser verification](evidence/overnight-research-browser.json)
- [Laptop cleanup receipt](evidence/laptop-storage-recovery-20261115.json)

The page's model-label agreement is not confirmed physical mine accuracy. Machine-screened zero-candidate frames are not human-confirmed mine-free backgrounds. The opened 182-image test is not used for new prompt selection or training.

## Corrections retained

The first expanded launch was rejected before allocating a model because the capacity check found targets exceeding 384 tokens. The new 512-token source preserves all targets. An initial native-queue observation treated Harbor's nonterminal result file as completed; that record is marked superseded and the queue now checks terminal timestamps and counters. The controller was restarted without stopping the independently managed mission trials. Public status-file permissions were corrected, then the HTTPS page and browser rendering were verified.

## Validation

The expanded local suite passed **377 tests** at 02:23 UTC. The optimization additionally passed actual A100 finite-gradient, unchanged-gradient, norm-equivalence and optimizer-trajectory checks. Further measurements and terminal run results will be appended as they arrive.

## Measured replication results, 02:23 UTC

All three fixed-recipe visual runs completed, including their 96-update GRPO continuations.

| Seed | 480 SFT F2 | +48 GRPO F2 | +144 total GRPO F2 | Final TP / FP / FN | Model-zero reference disagreements |
|---|---:|---:|---:|---|---|
| 17 | 0.509 | 0.601 | 0.650 | 55 / 16 / 33 | 5 / 32 |
| 43 | 0.513 | 0.615 | 0.670 | 58 / 23 / 30 | 4 / 32 |
| 91 | 0.467 | 0.605 | 0.706 | 61 / 19 / 27 | 4 / 32 |

Mean F2 increased from 0.496 to 0.607 to 0.675. All values use the same development references, including 30 original-manual, 58 reviewed-AI and 32 model-zero images. Every seed is shown. This is repeated development agreement, not independent mine-detection accuracy. The additional 96-SFT seed-17 control reached 0.530; update counts match, compute does not. No newer visual adapter has replaced the deployed language adapter.

- [Measured three-seed figure](evidence/visual-replications-v1.png) · [PDF](evidence/visual-replications-v1.pdf)
- [Metrics and exact source-file hashes](evidence/visual-replications-summary.json)

## Preserved data recovery and next queues

Original integer receipts and fresh autopilot IN_AIR telemetry recovered 1,206 eligible frame bindings from previously unaligned sources. Seven pixel/source identities already had cached coverage, leaving **1,199 new inputs**. Original flight audits remain unchanged, with no overrides of known ground decisions. Those exact JPEGs are now separately uploaded and server-MD5 verified in the private research bucket.

A separate conservative clock check excluded every timestamp-rollback interval plus the full two-second telemetry freshness margin. Three one-to-one, contiguous-counter segments yielded 828 safely aligned sampled frames; all 828 were on the ground. No model requests were spent on these frames. Five native-frame/receipt count mismatches remain quarantined. No timestamps or frame identities were guessed.

After the current 2,400-image runs, each GPU has paired +96-SFT and +96-GRPO controls queued from the same parent, subject to the existing task-variation guard. A subsequent three-seed fresh-base generation waits for both HIGH-label batches to settle, then uses all admissible HIGH positives, equal source-balanced explicit zeros and the same archived training anchors. Whole-session splits, nontraining pixel hashes, target lengths and frozen development evaluation are preserved. Changed labels and data composition are confounded; this is not a causal label-source ablation.

After first-pass labeling, bounded HIGH temporal-context and full-frame-plus-lossless-quadrant reviews are queued. These are correlated same-provider diagnostics, are explicitly training-ineligible, and do not overwrite first-pass labels or certify mine-free backgrounds. Raw responses, prompts, usage and derived input pixels are archived.

## Infrastructure recovery

The laptop lost new IPv4 connections while its established SSH tunnel remained usable. An encrypted IPv6 transport restored cloud management with the existing verified SSH host key; VM jobs continued throughout. No VM restart or network firewall change was needed.

The local Gemma benchmark proxy separately expired at its original six-hour runtime limit (01:56:30 UTC). The same proxy code and checkpoint were restored with a runtime that covers the VM's remaining approved life. New local benchmark trials now wait for proxy/model availability. Existing errored trials are preserved, with no outcome-based reruns. Some earlier long-context runs also encountered inference memory errors; their errors must remain separate from clean task failures.

The native full-day 3D local-object solve finished, but remains numerically unaccepted while correspondence gates are evaluated. The doubled-domain solve continues. Full heterogeneous-scene qualification remains incomplete.
