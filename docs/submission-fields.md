# Junction submission fields

## Project Name

AeroRozum

## Punchline

A drone investigation that challenges AI agents to find thermal evidence and make it home on time.

## Description

AeroRozum turns a drone survey into an investigation for an AI coding agent: choose where to fly, revisit uncertain thermal sightings, and submit a report supported by captured evidence. It is inspired by humanitarian mine-survey research.

The environment uses native Gazebo drone dynamics and paired thermal/RGB cameras. The agent chooses arbitrary local goto coordinates and altitude. It has 75 simulated flight seconds to cover two regions, inspect candidates, return home and land. Positive claims require a correctly localized capture at or below 4.5 m; independent views require at least 1 m horizontal separation. The aircraft keeps flying during inference. The agent can prepare code before takeoff and analyze saved evidence after landing.

This is a complete Harbor task: instruction.md, task.toml, environment/Dockerfile, tests/ and solution/. OpenCode plays through the real tool interface. The verifier writes one score to /logs/verifier/reward.txt: 65% one-to-one retrieval F2, 20% captured-region coverage and 15% independent-view evidence. Report submission, airborne evidence, home return and budget compliance gate the score. A separate simulator owns the authoritative flight and capture ledger.

On two frozen fresh instances, the public-image reference scored 0.733/1 each. Gemini Argon with a 32k output allowance returned home at 81.032 and 81.872 seconds, beyond the 75-second limit, and scored zero. Five actual native adversarial controls, including reward-file spoofing, forged evidence and private-state access, also scored zero. Additional multi-seed benchmarks retain model successes, task failures and infrastructure errors separately. The difficulty is investigation planning and evidence collection under a continuous flight budget.

Gemini Robotics ER 2 HIGH produced valid annotations for 15,856 telemetry-qualified airborne frame identities sampled at 5 FPS. Prompts, source generations, raw responses and usage are versioned and archived in Google Cloud Storage. A 2,160-completion prompt/thinking experiment measured F2 0.604 for V2 HIGH versus 0.298 for MINIMAL on reused development references. Another 12,000 temporal-context and lossless-quadrant review cases examine proposal sensitivity and missed candidates; these diagnostics remain outside training.

Gemma 4 12B runs on our cloud GPU and supports open-model post-training research with language/visual-projection LoRA, SFT and TRL GRPO. In the completed 602-image three-seed cohort, mean development F2 rose from 0.496 after SFT to 0.675 after 144 total GRPO updates. A separate completed 2,400-image cohort averaged 0.675 after SFT and 0.694 after guarded GRPO. The same 120 reused references include 30 original-manual, 58 reviewed-AI and 32 model-zero examples. These measure perception-label agreement, separately from native mission reward. Further training can regress; all outcomes are preserved and no new checkpoint is automatically deployed. A subsequent larger cohort is checkpointed for resumption, with unfinished stages explicitly marked.

Thermal observations compose recorded ground appearance with a local DuMux present/absent thermal response. Numerical admission applies to one local homogeneous-soil pair; larger 3D physics qualification continues in the background. Full heterogeneous-field and real mine-detection accuracy remain unvalidated.

The judging ZIP includes scene assets, runtime sources, actual Harbor/OpenCode logs, scalar rewards and file checksums. The environment uses CPU rendering; judges do not need a GPU. A one-command wizard checks dependencies, downloads the verified package and asks privately for a Gemini API key:

bash <(curl -fsSL https://136.115.179.240/start.sh)

## Main Challenge

AfterQuery — Break the Agent

## Tech Challenges

- Google DeepMind — Best Use of Gemini API
- Google DeepMind — Best Use of Gemma

## Project Demo

https://136.115.179.240/presentation/submission/index.html

## Source Code

https://github.com/Yegmina/Junction_2026/tree/codex/research-replication

## Video

https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/aerorozum-demo.mp4

## Harbor task download

https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/aerorozum-harbor-close-inspection-v5.zip

## Presentation PDF

https://136.115.179.240/presentation/submission/AeroRozum-Junction-2026-final.pdf


## Aiven addendum

Select **Best Use of Aiven**, and include the following in the description:


**Aiven evidence ledger.** Aiven PostgreSQL stores 50 versioned mission/control records and six held-out model evaluations. SQL queries expose budget failures, retain unsuccessful runs, and keep mission rewards separate from image scores. Content hashes prevent duplicate imports; the backend verifies TLS. Explore the queried results: https://136.115.179.240/presentation/aiven/index.html


## Verified V6 OpenCode flight


**Verified OpenCode flight (V6).** Our flight-operator profile scored 0.628: two of three inserted targets matched, zero false positives, six close captures, and home landing in 68.5 of 75 simulated seconds. A public waypoint collector reserves return time; the agent chooses its plan and analyses saved thermal/RGB evidence after landing. Replay: https://136.115.179.240/presentation/submission/playthrough/sdk-v6/index.html

Runnable V6 bundle: https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/aerorozum-harbor-close-inspection-v6.zip
