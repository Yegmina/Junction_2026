# Overnight research · 15 November 2026

The user authorized at least four additional hours of cloud experiments, use of the separate hackathon Gemini key, further high-thinking Robotics tests, and safe laptop cleanup. The resumed supervision window runs until at least **04:30 UTC / 06:30 Helsinki**. New paid calls stop by 05:45 UTC and accepted calls are allowed to settle. The existing VM's approved 48-hour STOP policy remains unchanged.

The submitted V5 package, frozen test results and deployed model checkpoint remain unchanged. Development work is on `codex/research-replication` in Yegmina/Junction_2026.

## What is running

- **Robotics development comparison:** two frozen prompts × native MINIMAL/MEDIUM/HIGH thinking × three repetitions × 120 established development images = 2,160 requests. Every arm has the same 65,536-token thinking-plus-answer allowance. First paid invalid completions count as detection failures; transport failures are reported separately. Raw output, usage, prompts and settings are archived. This is not a new independent test set.
- **HIGH-thinking flight relabeling:** starts after the paired study completes and its rate window clears. Prompt selection uses only the 88 manual-zone-7 development references, averaging F2 across three HIGH repetitions; v2 wins an exact tie. The original model labels remain intact under their original prefix. The new prefix is `robotics-er2-labels/high-thinking-airborne-v1` in the private flight bucket. The input set contains **14,657 unique eligible pixel/source-generation identities**, after avoiding 156 duplicates. All 354 initially missing source images were recovered directly from the read-only SSD and verified by SHA256. Fresh, unambiguous airborne telemetry is required. Ground and unknown flight intervals do not enter this pass.
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

The local suite passed **369 tests** before the isolated gradient-monitor optimization. The optimization additionally passed actual A100 finite-gradient, unchanged-gradient, norm-equivalence and optimizer-trajectory checks. Further measurements and terminal run results will be appended as they arrive.
