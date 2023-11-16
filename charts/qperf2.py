from .common import *

cont_1 = {
    (1, 4, 1):       25188126,
    (2, 4, 2):       50873944,
    (4, 4, 4):      100995752,
    (8, 4, 8):      201991552,
    (16, 4, 16):    403983840,
    (32, 4, 32):    807973504,
    (64, 4, 64):   1562897536,
    (108, 4, 108): 1581495808,
}
cont_2 = {
    (2, 4, 1): 50841896,
    (4, 4, 2): 100996536,
    (8, 4, 4): 201992224,
    (16, 4, 8): 403988960,
    (32, 4, 16): 504141984,
    (64, 4, 32): 927001536,
    (108, 4, 54): 1686451840,
    (216, 4, 108): 1744399488,
}
cont_4 = {
    (4, 4, 1): 100994760,
    (8, 4, 2): 202056544,
    (16, 4, 4): 219619728,
    (32, 4, 8): 471515840,
    (64, 4, 16): 556601216,
    (108, 4, 32): 900235904,
    (216, 4, 64): 1887217920,
    (432, 4, 108): 1625763968,
}
cont_8 = {
    (8, 4, 1): 202083360,
    (16, 4, 2): 236169728,
    (32, 4, 4): 261581136,
    (64, 4, 8): 485346848,
    (108, 4, 16): 547374656,
    (216, 4, 32): 1012806400,
    (432, 4, 64): 2057894272,
    (864, 4, 108): 1621274752,
}
cont_16 = {
    (16, 4, 1): 221200816,
    (32, 4, 2): 278971680,
    (64, 4, 4): 263265456,
    (108, 4, 8): 493591488,
    (216, 4, 16): 560784512,
    (432, 4, 32): 1059412288,
    (864, 4, 64): 1980104576,
}
cont_32 = {
    (32, 4, 1): 269997760,
    (64, 4, 2): 286684704,
    (108, 4, 3): 256180080,
}
cont_64 = {
    (64, 4, 1): 269998976,
    (108, 4, 2): 281262464,
}
cont_108 = {
    (108, 4, 1): 269999264,
}

tables = {
    1: cont_1,
    2: cont_2,
    4: cont_4,
    8: cont_8,
    16: cont_16,
    32: cont_32,
    64: cont_64,
    108: cont_108,
}


l2bw = np.array([
    [0.72,  39.94,   8.53,  438.20],
    [1.46,  72.09,   16.7,  883.14],
    [2.61,  143.01,  39.69, 1269.00],
    [6.65,  347.10,  32.45, 2083.50],
    [9.88,  527.12,  38.39, 2144.97],
    [20.56, 1011.64, 37.79, 2347.11],
    [39.27, 1553.40, 45.06, 2535.68],
    [42.67, 2005.75, 49.52, 2843.55],
    [51.87, 2060.69, 53.51, 2293.20],
    [54.09, 1254.44, 54.8,  1381.43],
    [55.89, 1248.38, 57.1,  1321.99],
    [57.15, 1247.80, 57.58, 1330.54],
])

num_prod_cons = np.array([1, 27])

payload_kb = np.array([
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
])

print('1 Queues')
for i, x in enumerate(l2bw[:, 2] / l2bw[:, 0]):
    print(f'{payload_kb[i]}KB: {x:.2f}x')
print()
print('54 Queues')
for i, x in enumerate(l2bw[:, 3] / l2bw[:, 1]):
    print(f'{payload_kb[i]}KB: {x:.2f}x')


labels = np.array([
    1, 4, 16, 64, 256, 1024
])

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(COL_WIDTH, 3.0))

for cont in [1, 2, 4, 16, 64, 108]:
    keys = np.array([
        [nb, nw, nv]
        for (nb, nw, nv) in tables[cont].keys()
    ])

    idxs = np.argsort(keys[:, 2], axis=0)
    keys = keys[idxs]

    x_points = keys[:, 2]

    y_points = np.array([
        tables[cont][tuple(k)]
        for k in keys
    ]) * keys[:, 1] / keys[:, 0]

    print(x_points)
    print(y_points)

    ax1.plot(x_points, y_points / 1e6, label=f'{cont}x', marker='^')

ax1.semilogx()
ax1.set_xlabel('# Variables', fontsize=8)
ax1.set_xticks([1, 2, 4, 8, 16, 32, 64, 108], [1, 2, 4, 8, 16, 32, 64, 108])
ax1.set_ylabel('Mega-atomicAdd / sec \n per threadblock', fontsize=8)
ax1.set_ylim([0, 4*3.2e7 / 1e6])
ax1.tick_params(labelsize=8)


ax1.legend(
    fontsize=6,
    ncol=6,
    loc='upper center',
    # bbox_to_anchor=(0.5, -0.15),
    markerscale=0.8,
    columnspacing=1,
    handlelength=1.5,
    frameon=False)

ax2.plot([1, 2048], [1555, 1555], '--', color='black', label='HBM BW')
ax2.plot([379, 379], [0, 3000], '--', color='red', label='L2 Cap')
ax2.plot(payload_kb, l2bw[:, 1], label=f'54 Q w/s', marker='o', markersize=4)
ax2.plot(payload_kb, l2bw[:, 3], label=f'54 Q NS', marker='o', markersize=4)
ax2.legend(fontsize=6, ncol=2, loc='lower right')
ax2.set_ylim(0, 3000)
ax2.tick_params(axis='y', labelsize=8)
ax2.set_ylabel('Bandwidth (GB/s)', fontsize=8)
ax2.semilogx(base=2)
ax2.legend(fontsize=6, ncol=2, loc='lower right')
ax2.set_xlabel('Payload Size (KB)', fontsize=8)
ax2.set_xticks(labels, labels)
ax2.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('atomics-qperf.pdf')
