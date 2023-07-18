import numpy as np
from matplotlib import pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
COL_WIDTH = (8.5 - 1.5 - 0.25) / 2
TEXT_WIDTH = 8.5 - 1.5

pow_1530 = np.array([275, 269, 275])
pow_825 = np.array([172, 140, 140])

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(COL_WIDTH, 2))
bar_width = 0.35

bar_positions = np.arange(3)
ax.bar(bar_positions - bar_width/2, pow_1530, bar_width, color=colors[0], label='1530 MHz')
ax.bar(bar_positions + bar_width/2, pow_825, bar_width, color=colors[2], label='825 MHz')

ax.set_xticks(bar_positions)
ax.set_xticklabels([
    'Bulk-Sync',
    'Pipelined',
    'Pipelined-NS'
], fontsize=8)

ax.set_ylabel('Power (W)', fontsize=8)
ax.tick_params(labelsize=8)

ax.legend(
    fontsize=8,
    ncol=2,
    loc='upper center',
    framealpha=1.0,
    facecolor='#FFFFFF',
    edgecolor='#FFFFFF')

ax.set_ylim([0, 375])

plt.title('Power Masurements of mgn_linears', fontsize=8)
plt.tight_layout()
plt.savefig('power.pdf')
