"""Present the expanded cohort with incomplete arms explicitly left pending."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('docs/evidence')
data = json.loads((root / 'expanded-comparison-v1.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout='constrained')
colors = ['#087e8b', '#a64b35', '#654b98']
for row, color in zip(data['seeds'], colors):
    stages = row['stages']
    for index, ax in enumerate(axes):
        values = stages[:2] if index == 0 else stages[2:]
        xs = [i for i, s in enumerate(values) if s['state'] == 'completed']
        ys = [s['micro_f2'] for s in values if s['state'] == 'completed']
        if xs:
            ax.plot(xs, ys, marker='o', color=color, lw=2, label=f"Seed {row['seed']}")
        for i, stage in enumerate(values):
            if stage['state'] == 'completed':
                ax.annotate(f"{stage['micro_f2']:.3f} · {stage['actual_updates']} updates",
                            (i, stage['micro_f2']), xytext=(5, 9 if row['seed'] != 43 else -16),
                            textcoords='offset points', fontsize=9, color=color)
        if len(xs) < 2:
            ax.text(.02, .07 + .06 * (17, 43, 91).index(row['seed']),
                    f"Seed {row['seed']}: " + ', '.join(s['stage'] + ' pending' for s in values
                                                     if s['state'] != 'completed'),
                    transform=ax.transAxes, color=color, fontsize=9)
axes[0].set_xticks([0, 1], ['1,920 SFT updates', 'Then guarded GRPO'])
axes[1].set_xticks([0, 1], ['Extra SFT', 'Extra GRPO'])
axes[0].set_title('2,400-image cohort · initial training', loc='left', fontweight='bold')
axes[1].set_title('Same-parent continuations · maximum 96 updates', loc='left', fontweight='bold')
for ax in axes:
    ax.set(xlim=(-.12, 1.65), ylim=(0, 1), ylabel='Development micro F2')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    if ax.lines:
        ax.legend(loc='upper left', frameon=False)
fig.suptitle('Report every seed and the actual update counts.', x=.02, ha='left', fontsize=18, fontweight='bold')
fig.supxlabel('Same 120 reused development images: 30 original-manual, 58 reviewed-AI and 32 model-zero references.\n'
              'Continuation arms share their seed’s initial GRPO parent; guard stops can make actual updates and compute differ.\n'
              'Pending cells are not zero scores. Label agreement does not establish independent field accuracy. '
              + data['at'][:16] + ' UTC.', fontsize=9)
for suffix in ['png', 'pdf', 'svg']:
    fig.savefig(root / ('expanded-comparison-v1.' + suffix), dpi=180)
svg = root / 'expanded-comparison-v1.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
print(json.dumps({'all_stages_completed': data['all_stages_completed'], 'seeds': len(data['seeds'])}))
