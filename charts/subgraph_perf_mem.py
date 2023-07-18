from .common import *

app_names = [
    'mgn_mlp',
    'dlrm_bot',
    'dlrm_top',
    'bert_ffn',
    'nerf_all',
    'gc_nodes',
    'gc_edges'
]

speedup = np.array([1.30, 3.31, 0.99, 1.10, 1.21, 0.99, 0.91])
traffic_reduction = np.array([3.5, 15.3, 9.6, 7.6, 116.1, 3.0, 2.5])

# Add geomean
app_names.append('GM')
speedup = np.append(speedup, gmean(speedup))
traffic_reduction = np.append(traffic_reduction, gmean(traffic_reduction))


with figure(COL_WIDTH, 2, 1, 2) as (fig, (ax1, ax2)):
    bars(ax1, app_names, speedup, ylim=2, labeloverflow=True, color=colors[0], baseline=True)
    bars(ax2, app_names, traffic_reduction, ylim=15, labeloverflow=True, color=colors[2], baseline=True)

    ax1.set_title('Speedup', fontsize=8)
    ax2.set_title('Traffic Reduction', fontsize=8)

    plt.tight_layout()

