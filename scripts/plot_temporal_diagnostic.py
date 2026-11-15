"""Plot paired model decisions; neither arm is an independent reference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('docs/evidence')
d = json.loads((root / 'temporal-review-diagnostic-v1.json').read_text())
assert d['status'] == 'completed'
table = d['accepted_frame_table_blind_then_proposal']
matrix = np.array([[table.get('none/none', 0), table.get('none/accepted', 0)],
                   [table.get('accepted/none', 0), table.get('accepted/accepted', 0)]])
assert int(matrix.sum()) == d['valid_paired_frames']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
axes[0].imshow(matrix, cmap='Blues', vmin=0, vmax=max(matrix.max(), 1))
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f'{matrix[i,j]:,}', ha='center', va='center', fontsize=21,
                     color='white' if matrix[i,j] > matrix.max() * .6 else '#143345')
labels = ['No accepted candidate', 'Accepted candidate(s)']
axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
            xlabel='Same context + earlier unverified proposals', ylabel='Blind three-frame context')
axes[0].set_title('Paired frame decisions · threshold 0.60', loc='left', fontweight='bold')
for spine in axes[0].spines.values():
    spine.set_visible(False)
arms = ['blind_context', 'proposal_context']
accepted = [d['accepted_candidate_totals'][a] for a in arms]
uncertain = [d['uncertain_candidate_totals'][a] for a in arms]
x = np.arange(2)
bars1 = axes[1].bar(x - .18, accepted, .35, color='#087e8b', label='Accepted ≥0.60')
bars2 = axes[1].bar(x + .18, uncertain, .35, color='#a64b35', label='Uncertain <0.60')
axes[1].bar_label(bars1, padding=4)
axes[1].bar_label(bars2, padding=4)
axes[1].set(xticks=x, xticklabels=['Blind context', 'Context + proposals'],
            ylim=(0, max(uncertain + accepted) * 1.3), ylabel='Candidate records across valid pairs')
axes[1].set_title('Candidate totals · same paired-frame subset', loc='left', fontweight='bold')
axes[1].legend(frameon=False)
axes[1].spines[['top', 'right']].set_visible(False)
fig.suptitle(f"Review arms disagreed on {d['acceptance_disagreement_fraction']:.1%} of paired frames.",
             x=.02, ha='left', fontsize=18, fontweight='bold')
fig.supxlabel(f"{d['valid_paired_frames']:,} valid pairs / {d['expected_pairs']:,} frozen frame identities; "
              f"{sum(d['invalid_outputs_by_arm'].values())} invalid outputs retained separately. "
              'HIGH thinking · same three original thermal frames in both arms.\n'
              'Same-provider, single-completion diagnostic: neither arm establishes truth or physical absence.\n'
              'Source frames are correlated. No first-pass labels overwritten; these reviews are training-ineligible.', fontsize=9)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(root / ('temporal-review-diagnostic-v1.' + ext), dpi=180)
svg = root / 'temporal-review-diagnostic-v1.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
print(json.dumps({'paired_frames': d['valid_paired_frames'], 'disagreement_fraction': d['acceptance_disagreement_fraction']}))
