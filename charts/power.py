from .common import *

payload_kb = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

pow_1q = np.array([72, 71, 71, 72, 72, 74, 76, 76, 78, 80, 85, 84])
pow_54q = np.array([141, 145, 147, 161, 185, 198, 217, 220, 250, 245, 248, 249])

dram_1q = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
dram_54q = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.33, 0.73, 0.74, 0.75])



with figure(COL_WIDTH, 2, 1, 2) as (fig, (ax1, ax2)):
    labels = list(map(lambda s: f'{s} KB', payload_kb))
    bars(
        ax1,
        payload_kb,
        pow_1q,
        ylim=100,
        intylabels=True,
        locator_bins=8,
        labeloverflow=True,
        color=colors[4],
        baseline=True)

    ax1.set_title('1 Queue Power (W)', fontsize=8)
    ax1.set_xlabel('Payload Size (KB)', fontsize=8)

    xs = bars(
        ax2,
        payload_kb,
        pow_54q,
        ylim=300,
        intylabels=True,
        labeloverflow=True,
        locator_bins=8,
        color=colors[9],
        baseline=True)

    ax2.plot(
        list(ax2.get_xlim()),
        [250, 250],
        color='black',
        label='TDP',
        linestyle='--',
        linewidth=1.0)

    for i in range(len(dram_54q)):
        if dram_54q[i] > 0:
            ax2.text(
                xs[i],
                pow_54q[i] - 60,
                f'{int(dram_54q[i] * 100)}%',
                ha='center',
                fontsize=6,
                rotation=90,
                zorder=999)

    ax2.set_title('54 Queues Power (W)', fontsize=8)
    ax2.set_xlabel('Payload Size (KB)', fontsize=8)


    plt.tight_layout()

