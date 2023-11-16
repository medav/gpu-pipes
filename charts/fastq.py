from .common import *

gpu_names = [
    'RTX 3090',
    'RTX A6000',
    'A100',
    'RTX 4090',
    'RTX 6000 Ada',
    'H100'
]

app_names = [
    'mgn_mlp',
    'dlrm_bot',
    'dlrm_top',
    'bert_ffn',
    'nerf_all',
    'gc_nodes',
    'gc_edges'
]

a100_baseline_gflops = \
    np.array([33983, 4315, 33301, 34723,  48440,  97943, 109042])

a100_dataflow_gflops = \
    np.array([44122, 14389, 32857, 38416, 58741, 97299, 99267])


fp16_tflops = {
    'RTX 3090': 142,
    'RTX A6000': 309,
    'A100': 312,
    'RTX 4090': 330,
    'RTX 6000 Ada': 720,
    'H100': 756
}

mem_bw = {
    'RTX 3090': 936,
    'RTX A6000': 768,
    'A100': 1555,
    'RTX 4090': 1008,
    'RTX 6000 Ada': 960,
    'H100': 2039
}

def dataflow_speedup(gpu : str):
    gpu_baseline_gflops = a100_baseline_gflops * (mem_bw[gpu] / mem_bw['A100'])
    gpu_dataflow_gflops = a100_dataflow_gflops * (fp16_tflops[gpu] / fp16_tflops['A100'])
    return gpu_dataflow_gflops / gpu_baseline_gflops

def fastq_speedup(gpu : str):
    gpu_baseline_gflops = a100_baseline_gflops * (mem_bw[gpu] / mem_bw['A100'])
    return 0.62 * fp16_tflops[gpu] * 1000 / gpu_baseline_gflops

np.set_printoptions(linewidth=np.inf, precision=2)
for gpu in ['A100']:
    dataflow = dataflow_speedup(gpu)
    dataflow = np.append(dataflow, gmean(dataflow))
    fastqs = fastq_speedup(gpu)
    fastqs = np.append(fastqs, gmean(fastqs))
    print(f'{gpu}:')
    for i, app in enumerate(app_names + ['GM']):
        print(f'    {app.ljust(8)}: {dataflow[i]:.2f} {fastqs[i]:.2f} {fastqs[i] / dataflow[i]:.2f}')
    print()


with figure(COL_WIDTH, 2, 1, 1) as (fig, ax):

    speeds = np.array([
        dataflow_speedup('A100'),
        fastq_speedup('A100')
    ])

    # add geomean
    speeds = np.array([
        np.append(speeds[0], gmean(speeds[0])),
        np.append(speeds[1], gmean(speeds[1]))
    ])

    app_names_gm = app_names + ['GM']

    multibars(
        ax,
        app_names_gm,
        ['+Dataflow', '+Fast Qs'],
        speeds,
        baseline=True,
        ylim=8,
        labeloverflow=True,
        bar_width=0.5,
        pad_width=0.1,
        locator_bins=4,
        intylabels=True,
    )

    ax.legend(
        fontsize=6,
        ncol=2,
        loc='upper right',
        framealpha=0.0,
        bbox_to_anchor=(1.0, 1.0),
        facecolor='#FFFFFF',
        edgecolor='#FFFFFF')

    # plt.title('Impact of Fast Queues', fontsize=8)
    plt.tight_layout()
