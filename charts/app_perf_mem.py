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


fastq_speedup = np.array([
    1.58,
    1.74,
    1.26,
    4,
    1.26
])

traffic_reduction = np.array([
    1.44,
    1.65,
    1.29,
    39.1,
    1.54,
])

# Add geomean
app_names.append('GM')
speedup = np.append(speedup, gmean(speedup))
fastq_speedup = np.append(fastq_speedup, gmean(fastq_speedup))
traffic_reduction = np.append(traffic_reduction, gmean(traffic_reduction))


with figure(COL_WIDTH, 1.75, 1, 2) as (fig, (ax1, ax2)):
    bars(ax1, app_names, fastq_speedup, ylim=2, labeloverflow=True, color=colors[1], barlabel='Kitsune', zorder=-10)


    bars(ax1, app_names, speedup, ylim=2, labeloverflow=True, color=colors[0], baseline=True, barlabel='KitsuneSW', hatch='///')

    bars2please_forgiveme(ax2, app_names, traffic_reduction, ylim=4, labeloverflow=True, color=colors[2], baseline=True, hatch='///')

    ax1.set_title('Speedup', fontsize=8)
    ax1.legend(
        loc='lower center',
        ncol=1,
        fontsize=6,
        frameon=True,
        framealpha=1.0,
        facecolor='#FFFFFF',
        edgecolor='#FFFFFF')
    ax2.set_title('Traffic Reduction', fontsize=8)

    plt.tight_layout()

