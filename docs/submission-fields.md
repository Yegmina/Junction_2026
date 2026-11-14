# Junction submission fields

## Project Name

AeroRozum

## Punchline

A drone investigation that challenges AI agents to find thermal evidence and make it home on time.

## Description

AeroRozum is a Harbor environment for AI drone investigation, inspired by humanitarian mine-survey research. An agent receives two survey regions and chooses where to fly, which thermal sightings to inspect, and which evidence belongs in its final report.

The drone uses native Gazebo dynamics and paired thermal/RGB cameras. It keeps flying during inference. A wide scan covers ground quickly, but positive claims require a correctly localized capture at or below 4.5 m. The agent has 75 simulated flight seconds to inspect, revisit uncertain evidence, return home and land. It can prepare code before takeoff and analyze saved evidence after landing.

The verifier writes one reward in [0,1]: 65% one-to-one retrieval F2, 20% captured-region coverage and 15% independent-view evidence. A report, airborne evidence, home return and budget compliance are required. Duplicate and unsupported claims reduce credit. A separate simulator owns the authoritative flight and evidence ledger.

The public-image reference completed two fresh instances at 0.733/1 each. In two clean runs with a larger output allowance, Gemini Argon found two supported targets with no false claims and full coverage, but landed at 81.032 and 81.872 seconds—over the 75-second limit—and scored zero. Other model outcomes and execution errors are preserved in the complete result table. Five native adversarial controls, including reward-file spoofing and forged evidence, each scored zero.

Supporting research uses Gemma 4 12B visual/language LoRA, SFT and TRL GRPO on 602 thermal training examples with retained label provenance. A direct 96-update GRPO continuation improved development F2 from 0.601 to 0.650, versus Carbon's 0.642 on the same 120 development images. The student made fewer false claims and missed more reference objects. These are single-seed development results. On 182 previously held-out images, additional GRPO improved Gemma F2 from 0.611 to 0.632; the untouched base scored 0.036, Robotics 0.587, Carbon 0.649 and Argon 0.700. Test scores never select a checkpoint. These label-agreement results use different inference allowances.

The prepared ZIP includes the complete Harbor task, scene assets, runtime sources, reference policy, real OpenCode model logs, scores and file hashes. A fresh run from an independently extracted archive reproduced the frozen task checksum. The thermal response comes from a numerically admitted local DuMux present/absent simulation composed with recorded ground appearance; full-field and real mine-detection accuracy are not established.

## Main Challenge

AfterQuery — Break the Agent

## Tech Challenges

- Google DeepMind — Best Use of Gemini API
- Google DeepMind — Best Use of Gemma

## Project Demo

https://136.115.179.240/presentation/submission/index.html

## Source Code

https://github.com/Yegmina/Junction_2026/tree/codex/harbor-investigation

## Video

https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/aerorozum-demo.mp4

## Other Link

https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/aerorozum-harbor-close-inspection-v5.zip

## Presentation

AeroRozum-Junction-2026.pdf

## Presentation Download

https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/AeroRozum-Junction-2026.pdf

