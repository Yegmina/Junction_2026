"""Explicit training-only class weighting without removing any source example."""
from .common import digest


def repeat_positives(rows,repeats=1):
    if type(repeats) is not int or repeats not in (1,4):raise ValueError('Predeclared positive repeat factors are 1 and 4')
    positive=[r for r in rows if r.get('gold_boxes')];negative=[r for r in rows if not r.get('gold_boxes')]
    if not positive or not negative:raise ValueError('Replay comparison requires both classes')
    expanded=[row for row in rows for _ in range(repeats if row.get('gold_boxes') else 1)]
    protocol=dict(version='class-weighted-replay-v1',positive_repeats=repeats,negative_repeats=1,
        source_rows=len(rows),source_positive_rows=len(positive),source_zero_rows=len(negative),
        effective_entries=len(expanded),positive_entry_fraction=len(positive)*repeats/len(expanded),
        every_source_example_retained=True,source_order_unchanged_when_factor_one=True,
        metadata_only_weights=False,labels_and_source_pixels_unchanged=True,
        interpretation='Repeat eligible positive examples before the existing shuffled GRPO sampler. The four-group zero-variance stop rule is unchanged.')
    protocol['sha256']=digest(protocol)
    return expanded,protocol
