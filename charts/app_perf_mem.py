import numpy as np
from matplotlib import pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
COL_WIDTH = (8.5 - 1.5 - 0.25) / 2
TEXT_WIDTH = 8.5 - 1.5

app_names = [
    'MGN',
    'DLRM',
    'BERT',
    'NERF',
    'GRC'
]

speedup = np.array([
    1.115,
    1.139,
    0.934,
    1.212,
    0.871,
])

traffic_reduction = np.array([
    1.59,
    1.95,
    1.87,
    117,
    1.37,
])

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))


bar_width = 0.35


ax.plot([-bar_width, 4+bar_width], [1, 1], color='black', linestyle='--', linewidth=1.0)

bar_positions = np.arange(len(app_names))
ax.bar(bar_positions - bar_width/2, speedup, bar_width, color=colors[0], label='Speedup')
ax.bar(bar_positions + bar_width/2, traffic_reduction, bar_width, color=colors[2], label='Traffic Reduction')
# ax.set_xlabel('Application', fontsize=8)
# ax.set_ylabel('Performance Metrics', fontsize=8)
ax.set_ylim([0, 2])
ax.set_xticks(bar_positions)
ax.set_xticklabels(app_names)

ax.tick_params(labelsize=8)

ax.set_xlim([-bar_width, 4+bar_width])

# Add text annotations for NERF bar
nerf_index = app_names.index('NERF')
nerf_speedup = speedup[nerf_index]
nerf_traffic_reduction = traffic_reduction[nerf_index]
text_offset = 0.1

ax.text(
    bar_positions[nerf_index] + bar_width/2,
    2 - text_offset,
    f'{int(nerf_traffic_reduction)}X',
    ha='center',
    fontsize=6,
    zorder=999)

# Set the legend, title, and display the plot
ax.legend(fontsize=8, loc='lower right')
plt.title('Application Speedup and Memory Traffic Reduction', fontsize=8)
plt.tight_layout()
plt.savefig('app_perf_mem.pdf')