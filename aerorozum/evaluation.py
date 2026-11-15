"""Privileged deterministic evidence scoring. Never exposed as an agent tool."""
from __future__ import annotations

import math
from numbers import Real


BOX_KEYS = ("x_min", "y_min", "x_max", "y_max")


def valid_box(box):
    try:
        raw = [box[k] for k in BOX_KEYS]
        if not all(isinstance(x,Real) and not isinstance(x,bool) for x in raw):
            return False
        values = [float(x) for x in raw]
        return (all(math.isfinite(x) and 0 <= x <= 1000 for x in values)
                and values[0] < values[2] and values[1] < values[3])
    except (TypeError, KeyError, ValueError):
        return False


def iou(a, b):
    if not valid_box(a) or not valid_box(b):
        return 0.0
    w = max(0, min(a["x_max"],b["x_max"])-max(a["x_min"],b["x_min"]))
    h = max(0, min(a["y_max"],b["y_max"])-max(a["y_min"],b["y_min"]))
    inter = w*h
    area_a = (a["x_max"]-a["x_min"])*(a["y_max"]-a["y_min"])
    area_b = (b["x_max"]-b["x_min"])*(b["y_max"]-b["y_min"])
    return inter / max(area_a+area_b-inter, 1e-12)


def score_submission(candidates, observations, observable_target_ids, coverage,
                     *, completed=True, threshold=.5, observability_verified=False):
    """Maximum-cardinality one-to-one matching; repeats count as false positives.

    observations is the privileged mapping of observation IDs to visible truth.
    The caller fixes the eligible target set independently of the agent route.
    """
    truth = sorted(str(x) for x in observable_target_ids)
    edges = {}
    for i, candidate in enumerate(candidates):
        for evidence in candidate.get("evidence", []):
            record = observations.get(evidence.get("observation_id"), {})
            for target in record.get("truth", []):
                key = str(target["target_id"])
                if target.get("observable") and key in truth:
                    overlap = iou(evidence.get("box", {}), target["box"])
                    if overlap >= threshold:
                        edges.setdefault(i, set()).add(key)
    # Augmenting paths maximize cardinality, unlike greedy overlap sorting.
    assigned = {}
    def match(i, seen):
        for target in sorted(edges.get(i, [])):
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or match(assigned[target], seen):
                assigned[target] = i
                return True
        return False
    for i in range(len(candidates)):
        match(i, set())
    tp = len(assigned); fp = len(candidates)-tp; fn = len(truth)-tp
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else None
    f2 = 5*tp/(5*tp+4*fn+fp) if (5*tp+4*fn+fp) else 0.0
    coverage = max(0.0, min(1.0, float(coverage)))
    if not completed:
        reward = 0.0
    elif not truth:
        reward = coverage if fp == 0 else 0.0
    else:
        reward = .7*f2 + .3*coverage
    return {"reward":float(reward), "true_positives":tp, "false_positives":fp,
            "false_negatives":fn, "precision":float(precision), "recall":recall,
            "f2":float(f2), "qualified_coverage":coverage,
            "evaluable_targets":len(truth), "matched_target_ids":sorted(assigned),
            "observability_verified":bool(observability_verified),
            "interpretation":"Simulation evidence score; not real mine confirmation"}
