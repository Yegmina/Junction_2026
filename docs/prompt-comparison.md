# Reviewed-image prompt comparison

Measured development winner at confidence 0.6: **thermal-v1-conservative**.

The exact Robotics ER2 model saw the same 88 zone 7 development images under two frozen prompts. Zones 8 and 9 were not used. Reviewed boxes were available only to the evaluator.

| Prompt | Precision | Recall | F2 | Unmatched boxes/image |
|---|---:|---:|---:|---:|
| thermal-v1-conservative | 0.8846 | 0.2614 | 0.3042 | 0.0341 |
| thermal-v2-recall-calibrated | 0.8214 | 0.2614 | 0.3026 | 0.0568 |

All development frames are positive. Unmatched boxes measure disagreement with reviewed annotations, not a true background false-positive rate. One-zone development selection does not establish independent test performance.

## Lower-confidence proposals

- thermal-v1-conservative: retaining all lower-confidence proposals adds 13 matched boxes and 4 unmatched boxes. The accepted-training threshold remains 0.6.
- thermal-v2-recall-calibrated: retaining all lower-confidence proposals adds 22 matched boxes and 13 unmatched boxes. The accepted-training threshold remains 0.6.

## Production schema diagnostics

| Policy | V2 rejected frames | V2 matches at ≥0.6 | V2 matches at all confidences | V2 unmatched at all confidences |
|---|---:|---:|---:|---:|
| strict_raw_whole_image | 6 | 23 | 45 | 18 |
| lossless_aliases_then_strict_whole_image | 1 | 23 | 49 | 19 |

The original raw-output comparison is unchanged. Lossless aliases rename only explicit xmin/ymin/xmax/ymax keys; values and confidence stay identical. An ambiguous `max` key remains invalid. Whole-image rejection discards every proposal in an invalid frame. All six observed malformed frames contained only low-confidence proposals, so the ≥0.6 accepted comparison is unchanged.

V2 is the operational candidate-generation choice for the stated recall goal, with low-confidence proposals retained for review. V1 remains the marginal raw F2 winner at the fixed 0.6 acceptance threshold. This is a descriptive single-pass development comparison, not proof of a general improvement.

## Reproduction

Run `python -m aerorozum.prompt_comparison --output '/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/artifacts/prompt-comparisons/zone7-v1-v2-er2'`. Resume uses saved local/GCS responses before paid inference. Recompute figures with `--report-only` without invoking a model.

Frozen protocol, per-frame raw provider logs, parsed proposals, paired predictions, CSV threshold curves, plots and qualitative overlays are stored together. Green boxes are proposals ≥0.6; amber are lower-confidence proposals; cyan are reviewed annotations. Evaluation records are excluded from training.
