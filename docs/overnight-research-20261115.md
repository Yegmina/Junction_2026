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

After the current 2,400-image runs, each GPU has paired controls with a maximum of 96 SFT or GRPO updates from the same parent, subject to the existing task-variation guard. Actual update counts may differ when the GRPO guard stops early. A subsequent three-seed fresh-base generation waits for both HIGH-label batches to settle, then uses all admissible HIGH positives, equal source-balanced explicit zeros and the same archived training anchors. Whole-session splits, nontraining pixel hashes, target lengths and frozen development evaluation are preserved. Changed labels and data composition are confounded; this is not a causal label-source ablation.

After first-pass labeling, bounded HIGH temporal-context and full-frame-plus-lossless-quadrant reviews are queued. These are correlated same-provider diagnostics, are explicitly training-ineligible, and do not overwrite first-pass labels or certify mine-free backgrounds. Raw responses, prompts, usage and derived input pixels are archived.

## Infrastructure recovery

The laptop lost new IPv4 connections while its established SSH tunnel remained usable. An encrypted IPv6 transport restored cloud management with the existing verified SSH host key; VM jobs continued throughout. No VM restart or network firewall change was needed.

The local Gemma benchmark proxy separately expired at its original six-hour runtime limit (01:56:30 UTC). The same proxy code and checkpoint were restored with a runtime that covers the VM's remaining approved life. New local benchmark trials now wait for proxy/model availability. Existing errored trials are preserved, with no outcome-based reruns. Some earlier long-context runs also encountered inference memory errors; their errors must remain separate from clean task failures.

The native full-day 3D local-object solve finished, but remains numerically unaccepted while correspondence gates are evaluated. The doubled-domain solve continues. Full heterogeneous-scene qualification remains incomplete.

## Presentation diagnostics, 02:34 UTC

The completed local 3D trajectory now has a [scientific figure](evidence/physics-full-day-diagnostic-v1.png), [PDF](evidence/physics-full-day-diagnostic-v1.pdf), and a 12-second animation on the live page. The figure uses actual solved surface fields at four times plus all 2,887 adaptive samples for center/ring curves. The animation interpolates linearly between solved physical times. The displayed center-minus-ring difference spans −1.381 to +2.895 K. This spatial statistic is not a matched present/absent counterfactual. Its source hashes and methods are in the [render receipt](evidence/physics-full-day-diagnostic-receipt.json). No physics source, input, acceptance certificate or training admission was changed.

The [native outcome grid](evidence/native-outcomes-v1.png) shows every terminal result, with infrastructure errors, flight-budget failures and unstarted cases explicitly distinct. It is a partial snapshot; the live page refreshes the figure as jobs finish and archives each snapshot. For example, one Carbon episode retrieved two of three inserted targets but spent 83.56 of its 75 flight seconds, earning zero. An Argon episode retrieved two of three, covered both regions, returned home in 72.516 seconds, and scored 0.764. This illustrates investigation management separately from detector agreement. The vision reference completed 20 cases; one exceeded its budget by 0.236 seconds and also scored zero.

A separately predeclared canonical-tool Gemma control waits for the original 20-case Gemma cohort to settle. It retains the same deployed language adapter and task instances, but uses native tool-message formatting with a 16,384-token context to investigate the earlier long-context memory/protocol failures. Tool formatting and context change together, so any difference cannot be attributed to either alone. There are no outcome-based retries, no checkpoint promotion and no changes to the submitted environment or verifier. A simple protocol probe precedes the first actual control case.

The plotting dependencies live in an isolated cloud plotting environment. The training environment was left unchanged. Browser verification again passed desktop and 390 px mobile checks with actual live values and no injected metrics.

## Primary HIGH pass settled, 02:53 UTC

The primary pass issued 14,657 requests and retained **14,626 schema-valid results**, with 31 failed cases. Every valid result was uploaded, with zero pending uploads. The separately aligned 1,199-frame pass then started after the rate-window handover. Some failures are service-unavailable responses; others are malformed paid outputs. All originals and failures remain archived.

A single bounded recovery pass follows both batches. Its separate prompt, `thermal-v2-format-recovery-v1`, adds only a one-JSON-object / finite ordered box-format clarification to V2, while preserving the same images, HIGH setting, candidate recall instructions and 0.60 threshold. Each failed frame receives at most one additional attempt. Still-invalid recoveries remain quarantined. New full-teacher catalogs and supplemental context/crop reviews wait for this pass; they retain its distinct prompt provenance.

## Additional checkpoint evidence, declared before HIGH-teacher training

The next three-seed full-teacher generation retains its full 3.2 effective epochs and unchanged optimizer settings. Fixed adapter snapshots at approximately 0.5, 1.0 and 2.0 epochs are retained in addition to its final checkpoints, then evaluated separately using the same development protocol. These added comparisons are repeatedly used development evidence, with no automatic deployment or independent accuracy claim.

The actual snapshot callback passed a paired CPU qualification on a real tiny PEFT model with dropout: ten optimizer steps, three real adapter serializations, bit-identical losses and all final parameters, and unchanged RNG streams / parameter and gradient versions. Runtime guards also check those invariants at actual snapshot saves. [Qualification receipt](evidence/snapshot-monitor-qualification.json). The exact queued source transformation compiled successfully in the cloud without allocating a GPU or launching training. [Preflight receipt](evidence/high-teacher-source-preflight.json).

The first 2,400-image seed's supervised development evaluation completed at F2 **0.667** (TP 57, FP 18, FN 31; valid JSON 120/120; zero disagreements on the 32 model-zero reference frames). Its initial GRPO phase stopped at 15 of at most 48 updates after four consecutive zero-variance groups. Its GRPO evaluation and the other seeds remain pending; this partial larger-data cohort is separate from the completed 602-image three-seed results.

## Qualified flight pool covered, 03:07 UTC

The aligned follow-up settled with 1,194 valid results from 1,199 attempts. The separately versioned recovery produced 36 valid results from its 36 cases. Combined coverage is therefore **15,856 / 15,856 currently qualified airborne frame identities**, with 15,892 issued requests including the 36 original failures. All 15,856 valid results were uploaded with no pending uploads. Original errors and malformed paid completions remain intact. This coverage excludes ground intervals, unknown flight state and the five frame/receipt count mismatches still quarantined; it does not mean every raw video frame is labeled.

The frozen next-teacher catalog admits 7,094 native training records, keeps 482 archival training anchors, and excludes 6,598 uncertain/mixed training frames while retaining their raw evidence. Each seed's deterministic balanced curriculum has **4,688 images: 2,344 positive and 2,344 explicit zero-candidate frames**, across 22 source groups. It includes 162 original-manual, 320 reviewed-AI and 4,206 accepted-model records. Each full run requests 3,751 SFT updates and at most 48 guarded GRPO updates; fixed snapshots occur at steps 586, 1,172 and 2,344. [Input plan and hashes](evidence/high-teacher-curriculum-plan.json).

The first 2,400-image parent finished at F2 **0.677570** after 15 GRPO updates, versus **0.667447** after SFT: TP 58 vs 57, FP 18 in both, FN 30 vs 31, and zero model-zero-reference disagreements in both. These development results do not establish independent generalization. Its paired continuation controls are now running; the other seeds remain in progress.

The native-tool probe initially reached the legacy `/generate` route, which requires the flattened prompt schema. The separate control proxy now targets the existing canonical `/v1/chat/completions` route. The corrected probe returned READY, and the native control began actual Harbor missions. The original 422 probe error is retained; no task had started before this interface correction. The production service, checkpoint, task and context/output allowances were unchanged.

## Further paired replay comparison, declared 03:23 UTC

After full HIGH-teacher runs and their fixed-snapshot evaluations, three additional paired GRPO comparisons are queued. Both arms start from each seed's exact same full SFT adapter and retain every source example. The control uses one copy of each balanced source row; the treatment repeats each positive four times and each zero once, producing an effective 80/20 entry ratio before the existing shuffled grouped sampler. All image pixels, annotations, acceptance checks, loss settings and the four-zero-variance-group stop rule remain unchanged. Each arm has a maximum of 96 updates, but actual steps and compute may differ because the stop rule remains active. No model is automatically deployed.

The hypothesis is that positive replay can reduce stretches of uniform model-zero reward groups. It may also change false-alarm behavior; all outcomes and both reference metrics will be reported. Seven unit checks verify exact control order/object identity, no input mutation, complete source retention, the 80/20 ratio, and rejection of undeclared factors. A cloud preflight validated the actual frozen SFT parent and transformed 2,400 real examples into the expected 6,000 weighted entries without loading a GPU or training a model. [Predeclared protocol](evidence/paired-class-replay-protocol.json) · [Preflight](evidence/class-replay-source-preflight.json). The expanded local suite now passes **384 tests**.

## Cloud storage headroom

Fourteen temporary optimizer-checkpoint directories from the completed original replication cohort were evicted after every file matched its existing GCS server MD5 and byte count. A generation-pinned SHA256 restore manifest was uploaded before any removal. This freed **2.381 GiB** on the VM. Final adapters, active training checkpoints, source images and original datasets were retained. The archiver never enables deletion of unmatched remote objects. [Full restore receipt](evidence/completed-optimizer-cache-eviction-v1.json).

## Standalone presentation

A seven-slide research brief is available at **https://136.115.179.240/presentation/research-overnight/brief.pdf**, with a standalone HTML version at `brief.html`. It contains actual thermal examples, measured thinking and three-seed figures, the native investigation score grid, the solved physical-day diagnostic, and the queued comparisons. It is separate from the unchanged submission package.

The PDF is about 1.1 MB, below the form's 20 MB limit. Browser export verified seven pages, all images loaded, no page errors and no slide overflow. The HTML embeds the recorded figures, so it also works offline. Rebuild with `python3 scripts/build_overnight_brief.py` followed by `node scripts/export_overnight_brief.mjs` in an environment with Playwright and Chrome. Figure provenance and the browser receipt are in `docs/evidence/`.

## First continuation control completed, 03:44 UTC

The expanded seed-17 parent scored F2 0.677570 after its initial 15 GRPO updates. From that exact parent, the extra-SFT control completed all 96 requested updates and scored 0.686047 (TP 59, FP 19, FN 29). The extra-GRPO arm stopped at 25 of 96 requested updates and scored 0.704388 (TP 61, FP 20, FN 27). Both had zero disagreements on the 32 model-zero reference frames. These arms share the maximum requested update budget; their actual updates and compute differ. The earlier 602-image seed-17 comparison did complete 96 updates in both arms.

The first 4,688-image HIGH-teacher run then started on GPU 1. At 03:47 UTC it had completed 41 of 3,751 supervised updates, with 41 finite nonzero gradient checks and a nonzero visual-projection gradient. Other seeds remain in their predeclared comparisons before entering the new cohort.

## Continuous comparison reporting, 03:58 UTC

A separate read-only publisher now renders the 2,400-image cohort and its paired continuation arms whenever a terminal result appears. It shows all three declared seeds, the actual optimizer counts, and pending arms explicitly. The live page links both the figure and its source hashes. This reporter neither launches training nor changes its inputs. Source and rendered snapshots are archived privately in GCS.

The temporal-review summarizer compares accepted-frame decisions between blind three-frame context and the same context with earlier unverified proposals. It excludes invalid/unpaired outputs from the paired table and retains their counts separately. The comparison measures same-provider proposal sensitivity, not detection accuracy; no reviewed output overwrites the first-pass labels or becomes training data.

Removing only unused Docker build cache older than six hours reported 1.381 GB reclaimed. Concurrent work means measured filesystem headroom rose by 0.882 GiB during the command, rather than exactly that amount. Images, containers, volumes and original datasets were preserved. [Cleanup receipt](evidence/cloud-build-cache-cleanup.json).

## Native build recovery, 04:07 UTC

Cache cleanup exposed a dependency on cached Docker build layers: new Harbor trials attempted package installation, but container DNS could not reach the cloud resolver behind the existing link-local block. These setup failures remain infrastructure errors in the outcome grid. The cloud model, training, labeling and active Carbon mission were unaffected.

Remaining trials now use a separate operational task copy that names the **exact original retained image IDs**. Only `task.toml` and the Compose image-routing fields differ; instructions, tests, solution, budgets and runtime image contents are unchanged. All 52 packaged runtime/scene/SDK files matched their original image contents by SHA256. No rebuilt image or upgraded package is used for the resumed comparisons. A diagnostic legacy main-image build was retained as an unused infrastructure probe, not a new benchmark image.

The unchanged metadata firewall and daemon remain in place. Old dispatchers were held and replaced; accepted independent workers continued. Previously attempted worlds are never retried. A separate unscored-for-comparison smoke mission at seed 941999 reached ready state and executed actual flight/evidence actions; its terminal verifier result remains pending at this timestamp. The new dispatchers preserve their own source hashes and explicit package path. [Recovery receipt](evidence/native-prebuilt-recovery-v1.json).

The recovery smoke finished at 04:08:57 UTC with no infrastructure error and a valid verifier score of **0.000**. Its reference recovered all three inserted targets, covered both regions and returned home, but spent **75.572 / 75.000 flight seconds**. This is a real task failure despite successful execution; it is not added to the declared model cohort and will not be retried to obtain a favorable score.

## Completed temporal diagnostic

All 6,000 temporal requests settled: 5,966 schema-valid outputs and 34 invalid outputs, all retained. There are **2,967 valid pairs / 3,000 original frame identities**; one frame has invalid responses in both arms. Of those valid pairs, 1,867 had no accepted candidate in either arm, 586 had accepted candidates in both, 363 changed from no accepted candidate to accepted with proposals, and 151 changed in the reverse direction. Acceptance disagreement is therefore **17.324%**. Accepted candidate records total 859 for blind context and 1,191 with earlier proposals; uncertain records total 1,800 and 2,771 respectively.

This is a descriptive same-provider, single-completion diagnostic on correlated frames. Neither arm is a reference; it does not measure precision, recall or confirmed absence. Every review is training-ineligible. [Paired counts and frozen protocol](evidence/temporal-review-diagnostic-v1.json). The figure and PDF are linked on the live research page.

## First fixed checkpoint, 04:18 UTC

The HIGH-teacher seed-17 run saved its first fixed adapter at **step 586 / 0.5 epochs**, at 04:18:35 UTC. Actual CUDA/CPU/Python/NumPy RNG and parameter/gradient-version guards passed. Training continued beyond the snapshot (step 597 recorded at 04:19:12 with a finite nonzero gradient norm). Its development evaluation remains queued until the parent releases its GPU. [Snapshot, continued metric and archive receipt](evidence/first-fixed-snapshot-evidence.json).

A separate read-only archive worker now uploads each completed fixed snapshot as it appears, checks every remote file's byte count and server MD5 against local SHA256/MD5 records, and writes a generation-pinned manifest. The first backup verified at 04:18:48 UTC. It never deletes or modifies an active training checkpoint. The worker uses the existing cloud-auth dependencies in the ML virtual environment; an initial standard-library interpreter launch failed before any archival work and was replaced without installing packages or changing training.

The doubled-domain physical solve reached 61.2% of its full physical day at 04:16 UTC, with about 7.2 hours estimated remaining at that snapshot. The completed local 3D solve is still separate from full numerical admission. Full heterogeneous scenes and the planned 30,000 synthetic observations remain incomplete.

## Exact runtime preserved

The two original native benchmark images were streamed directly from `docker image save` through gzip to private GCS. The **1,044,707,190-byte** archive passed server MD5/size verification and has a recorded SHA256 and immutable object generation. No local tarball was created, and no original image or running container was removed. A fresh image-restore test has not been performed. [Archive identity and original image IDs](evidence/native-runtime-image-archive.json).

After image routing was repaired, the first resumed native Gemma episode completed without an infrastructure error at reward 0.000; the resumed Argon warming episode completed without an infrastructure error at reward 0.500. These are retained outcomes from the predeclared worlds, with the original model/checkpoint, inference allowances and verifier.
