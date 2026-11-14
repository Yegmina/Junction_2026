# Four-A100 mission decision checks

Existing VM: `aerorozum-gemma-pilot-20261114a`, four A100 40GB GPUs. No new VM or
quota expansion is used. Four isolated workers each load the same Gemma 4-12B
base weights, with Base, real-SFT, real-GRPO and stream-GRPO variants. Existing
training lineages are checked before launch. Bfloat16 is preserved; the
allocator is capped at 90%, CPU use at two cores per worker, runtime at 20 minutes.

`scripts/freeze_mission_probes.py` freezes 128 decision-contract probes from
actual paired camera observations. Cases cover initial inspection, limited
budget, explicit lower revisits, forbidden regions, a return-cost tradeoff,
missing visual evidence and report-ready return. Only the prompt and images
reach the model; private expected actions are used after immutable raw output
is saved. Zones 8/9 are untouched and no model/API credentials are copied.

| Checkpoint | JSON valid | Routing contract success |
|---|---:|---:|
| Base |53.125%|45.3125%|
| Real SFT |100%|75%|
| Real GRPO |100%|68.75%|
| Stream GRPO |78.125%|69.53125%|

These are **offline contract probes**, not full missions or mine-detection
accuracy. The token budget is 320; invalid or truncated completions count as
failures. The existing checkpoints have different training histories, so this
is descriptive checkpoint comparison, not a causal estimate of RL benefit.
The first result does not establish that GRPO outperforms SFT.

Fixed shortcut controls: do-nothing 0%; always-return 31.25%; always inspect the
first allowed region 43.75%. These expose easy routing shortcuts. Many cases
state their constraints explicitly, so a vision negative control is required
before attributing gains to image understanding.

The image-blind study keeps all 96 image-bearing cases' prompts, references,
image count/size and decoding unchanged, replacing image pixels with mid-gray.
It compares the same 96-case subset from the paired run. The 32 cases without
images are not rerun. Four new bounded workers perform this control.

The control also completed, for 896 total GPU generations across both studies.
Matched 96-case routing results:

| Checkpoint | Paired images | Blank images |
|---|---:|---:|
| Base |41.67%|58.33%|
| Real SFT |66.67%|58.33%|
| Real GRPO |58.33%|58.33%|
| Stream GRPO |59.375%|66.67%|

This small probe suite uses only two source image pairs. The negative control
confirms that these routing instructions are substantially text-solvable.
It does not demonstrate general visual investigation, a robust frontier gap,
or a benefit of RL. The next full benchmark must make image evidence matter
through closed-loop actions, independent references and held-out missions.

Results and raw generations are archived in the private research bucket under
`mission-eval/paired-decisions-v1/` and `mission-eval/paired-decisions-blank-v1/`.
The read-only mirror feeds `/api/mission-eval` and the Missions GPU panel.
This supplements the AfterQuery work; it does not replace the required
closed-loop frontier/reference runs or verify Harbor/OpenCode integration.
