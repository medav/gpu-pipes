from .common import *

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
    1.024,
    1.212,
    0.960
])

traffic_reduction = np.array([
    1.59,
    1.95,
    1.87,
    116,
    1.37,
])

# Add geomean
app_names.append('GM')
speedup = np.append(speedup, gmean(speedup))
traffic_reduction = np.append(traffic_reduction, gmean(traffic_reduction))


with figure(COL_WIDTH, 2, 1, 2) as (fig, (ax1, ax2)):

    bars(ax1, app_names, speedup, ylim=2, labeloverflow=True, color=colors[0], baseline=True)
    bars(ax2, app_names, traffic_reduction, ylim=4, labeloverflow=True, color=colors[2], baseline=True)

    ax1.set_title('Speedup', fontsize=8)
    ax2.set_title('Traffic Reduction', fontsize=8)

    plt.tight_layout()

