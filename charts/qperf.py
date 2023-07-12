import numpy as np
from matplotlib import pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
COL_WIDTH = (8.5 - 1.5 - 0.25) / 2
TEXT_WIDTH = 8.5 - 1.5

l2bw = np.array([
    [0.72, 39.94, 8.53, 431.03],
    [1.46, 72.09, 16.7, 659.64],
    [2.61, 143.01, 39.69, 969.44],
    [6.65, 347.10, 32.45, 912.9],
    [9.88, 527.12, 38.39, 1017.73],
    [20.56, 1011.64, 37.79, 1336.22],
    [39.27, 1553.40, 45.06, 1662.11],
    [42.67, 2005.75, 49.52, 1835.31],
    [51.87, 2060.69, 53.51, 1855.27],
    [54.09, 1254.44, 54.8, 1153.8],
    [55.89, 1248.38, 57.1, 1155.42],
    [57.15, 1247.80, 57.58, 1191.13]
])

print(l2bw[2] / l2bw[0])

num_prod_cons = np.array([1, 27])

payload_kb = np.array([
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
])

labels = np.array([
    '1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1k', '2k'
])

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(COL_WIDTH, 3), sharex=True, )

lines = []
labels = []

ax1.plot(payload_kb, l2bw[:, 0], label=f'1 Q w/ sync')
ax1.plot(payload_kb, l2bw[:, 2], label=f'1 Q NO sync')
ax1.legend(fontsize=6)
ax1.set_ylim(0, 60)
ax1.tick_params(axis='y', labelsize=8)
# ax1.set_ylabel('Bandwidth (GB/s)', fontsize=8)

ax2.plot([1, 2048], [1555, 1555], '--', color='black', label='L2 BW')
ax2.plot(payload_kb, l2bw[:, 1], label=f'54 Q w/ sync')
ax2.plot(payload_kb, l2bw[:, 3], label=f'54 Q NO sync')
ax2.legend(fontsize=6)
ax2.set_ylim(0, 2300)
ax2.tick_params(axis='y', labelsize=8)
# fig.set_ylabel('Bandwidth (GB/s)', fontsize=8)

fig.text(0.04, 0.5, 'Bandwidth (GB/s)', va='center', rotation='vertical', fontsize=8)


plt.semilogx(base=2)
plt.xlabel('Payload Size (KB)', fontsize=8)


plt.tight_layout()
plt.subplots_adjust(left=0.2)
plt.savefig('qperf.pdf')
