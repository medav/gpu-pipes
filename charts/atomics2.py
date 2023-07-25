

# nb, nw, nv
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

def lookup(nblocks : int, nwarps : int, nvars : int) -> float:
    return data_lookup[(nblocks, nwarps, nvars)]

import numpy as np
from matplotlib import pyplot as plt
import itertools

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
COL_WIDTH = (8.5 - 1.5 - 0.25) / 2
TEXT_WIDTH = 8.5 - 1.5

global_nvs = np.array([1, 3, 6, 13, 27, 54, 108])
global_nbs = np.array([1, 6, 27, 108])

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

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

    ax.plot(x_points, y_points, label=f'{cont}x', marker='o')

plt.semilogx()
# plt.loglog()

plt.xlabel('# Variables', fontsize=8)
plt.xticks([1, 2, 4, 8, 16, 32, 64, 108], [1, 2, 4, 8, 16, 32, 64, 108], fontsize=8)

plt.ylabel('atomicAdd / sec / Block', fontsize=8)
plt.yticks(fontsize=8)
plt.ylim([0, 4*3.2e7])

plt.title('Performance of atomicAdd', fontsize=10)

plt.legend(
    fontsize=6,
    ncol=6,
    loc='upper center',
    # bbox_to_anchor=(0.5, -0.15),
    markerscale=0.8,
    columnspacing=1,
    handlelength=1.5,
    frameon=False)
plt.tight_layout()
plt.savefig('atomics.pdf')


