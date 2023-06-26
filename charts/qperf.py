import numpy as np
from matplotlib import pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
COL_WIDTH = (8.5 - 1.5 - 0.25) / 2
TEXT_WIDTH = 8.5 - 1.5

l2bw = np.array([[
    1.4,
    2.8,
    6.5,
    11.9,
    22.5,
    22.5,
    22.5,
    22.3
], [
    74.0,
    154.0,
    302.0,
    626.0,
    837.0,
    828.0,
    863.0,
    860.0
]])

num_prod_cons = np.array([1, 54])

payload_kb = np.array([
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128
])

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2))

lines = []
labels = []

for i in range(len(num_prod_cons)):
    line, = ax.plot(payload_kb, l2bw[i], color=colors[i], label=f'{num_prod_cons[i]} Queues')
    lines.append(line)

plt.loglog()
plt.legend(fontsize=8, loc='lower right')
plt.xlabel('Payload Size (KB)', fontsize=8)
plt.xticks(payload_kb, payload_kb, fontsize=8)
plt.ylabel('Bandwidth (GB/s)', fontsize=8)
plt.yticks(fontsize=8)
plt.title('Performance of Ring Queue', fontsize=10)
plt.tight_layout()
plt.savefig('qperf.pdf')
