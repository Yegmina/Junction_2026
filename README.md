# AeroRozum RE:VISION

### The next Renaissance begins when AI learns not just to see, but to judge.

**A Harbor-compatible, physics-based drone investigation environment for evaluating frontier AI agents under uncertainty, evidence requirements, and strict flight constraints.**

Inspired by humanitarian mine-survey research, RE:VISION asks a question beyond image recognition: **can an AI agent conduct a complete aerial investigation, justify its findings, and finish the mission safely within its constraints?**

> The Renaissance transformed the way humanity understood the visible world. But can we trust what artificial intelligence believes it sees?

## The challenge

Recognizing a suspicious object in an image is not the same as completing an investigation. An agent must decide what to observe, where to fly, when to revisit uncertain evidence, what to report, and when to return. An agent may identify the right objects and still fail the mission.

RE:VISION presents an agent with **two survey regions** and a simulated drone equipped with paired **thermal and RGB cameras**. The drone follows native **Gazebo flight dynamics** and **continues flying while the agent reasons**, so slow or inefficient decisions have consequences.

The agent must:

- Choose where to fly and which thermal observations merit closer inspection.
- Revisit uncertain evidence when necessary.
- Support positive claims with correctly localized captures taken at **4.5 m altitude or lower**.
- Investigate, collect evidence, return home, and land within **75 simulated flight seconds**.
- Submit a final report containing only adequately supported findings.

The agent can prepare code before takeoff and analyze saved evidence after landing.

## How evaluation works

An **independent verifier** produces a reward in **[0, 1]**:

| Component | Weight | What it measures |
| --- | ---: | --- |
| One-to-one retrieval F2 | **65%** | Supported findings, missed targets, and false claims |
| Captured-region coverage | **20%** | Meaningful coverage of the assigned survey regions |
| Independent-view evidence | **15%** | Corroboration from independent observations |

A valid report, airborne evidence, return home, and compliance with the flight-time budget are **mandatory**. Duplicate and unsupported claims reduce credit.

The simulator maintains the **authoritative flight and evidence ledger** separately from the agent. Editing a report or reward file cannot legitimately establish mission success.

## Breaking frontier agents

The environment exposes failures that image-level accuracy alone can miss.

| Run | Observed outcome | Final reward |
| --- | --- | ---: |
| Public-image reference, fresh instance 1 | Completed reference run | **0.733 / 1** |
| Public-image reference, fresh instance 2 | Completed reference run | **0.733 / 1** |
| Gemini Argon, clean run 1 | Two supported targets, no false claims, full coverage; landed at **81.032 s** | **0** |
| Gemini Argon, clean run 2 | Two supported targets, no false claims, full coverage; landed at **81.872 s** | **0** |

The Gemini Argon runs used a larger output allowance. Both exceeded the **75-second** flight limit. The agent gathered strong evidence but **failed the overall mission**.

We also ran **five native adversarial controls**, including forged evidence and reward-file spoofing. **All five scored zero.** This demonstrates resistance to those tested attacks, not a guarantee against every possible exploit.

Other model outcomes and execution errors are retained in the complete submitted results.

## Supporting research: thermal-image interpretation

Alongside the Harbor environment, we explored reinforcement-learning-based fine-tuning for visual/language interpretation of thermal observations.

- **Model and methods:** Gemma 4 12B, LoRA, supervised fine-tuning (SFT), and TRL GRPO.
- **Training data:** 602 thermal examples with preserved label provenance.
- **Development:** A direct 96-update GRPO continuation improved F2 from **0.601 to 0.650** (Carbon: **0.642** on the same 120 development images).
- **Previously held-out images (182):** Additional GRPO improved F2 from **0.611 to 0.632**. Comparative results were: untouched base **0.036**, Robotics **0.587**, Carbon **0.649**, and Argon **0.700**.

The student model made fewer false claims but missed more reference objects. Development results are single-seed; the comparisons used different inference allowances. **Test scores were not used to select checkpoints.**

These metrics measure **agreement with image labels**, not operational mine-detection accuracy. This research is distinct from the Harbor mission evaluation.

## Reproducibility and integrity

The prepared submission archive includes:

- The complete Harbor task and Gazebo scene assets.
- Runtime sources and reference policy.
- Real agent execution logs and scoring results.
- File hashes and evidence for the frozen task.

A fresh run from an independently extracted archive reproduced the frozen task checksum.

**Thermal-model limitation:** Thermal responses come from a numerically admitted local **DuMux present/absent simulation**, composed with recorded ground appearance. We do **not** claim validated full-field thermal physics or real-world mine-detection accuracy.

## Why it matters

As AI agents move from answering questions to controlling real-world systems, recognizing the right object is no longer enough. They must make decisions under uncertainty, collect verifiable evidence, respect physical constraints, and complete their objectives.

RE:VISION provides a reproducible environment for exposing failures that conventional image-recognition benchmarks can miss.

**Because trustworthy autonomy begins with understanding where AI fails.**

**[View the pitch deck](RE%20VISION.pptx)**

[Video](https://youtu.be/aA5_dlPKcpE)
**The next Renaissance begins when AI learns not just to see, but to judge.**
