
import numpy as np

combos = [
    ('mgn_linears', 'bs', 825, True),
    ('mgn_linears', 'bs', 1530, True),
    ('mgn_linears', 'pl', 825, True),
    ('mgn_linears', 'pl', 825, False),
    ('mgn_linears', 'pl', 1530, True),
    ('mgn_linears', 'pl', 1530, False),

    ('test_mlp', 'bs', 825, True),
    ('test_mlp', 'bs', 1530, True),
    ('test_mlp', 'pl', 825, True),
    ('test_mlp', 'pl', 825, False),
    ('test_mlp', 'pl', 1530, True),
    ('test_mlp', 'pl', 1530, False),
]

def summarize(app, mode, freq, sync):
    mode_str = {
        'bs': 'bulksync',
        'pl': 'pipelined'
    }[mode]

    sync_str = '' if sync else '-ns'

    filename = f'ncu_results/{app}/ncu-{mode_str}-{freq}{sync_str}.txt'

    thrpts = []
    durations = []
    fp16_pipes = []
    tensor_pipes = []
    dram_thrpts = []

    with open(filename, 'r') as f:
        for line in f:
            if 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed' in line:
                dram_thrpts.append(float(line.split()[-1]))
            elif 'gpu__time_duration.sum' in line:
                durations.append(float(line.split()[-1]))
            elif 'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed' in line:
                fp16_pipes.append(float(line.split()[-1]))
            elif 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed' in line:
                tensor_pipes.append(float(line.split()[-1]))
            elif 'sm__throughput.avg.pct_of_peak_sustained_elapsed' in line:
                thrpts.append(float(line.split()[-1]))

    thrpts = np.array(thrpts)
    durations = np.array(durations)
    fp16_pipes = np.array(fp16_pipes)
    tensor_pipes = np.array(tensor_pipes)
    dram_thrpts = np.array(dram_thrpts)

    durations = durations / np.sum(durations)
    weighted_dram = np.sum(dram_thrpts * durations)
    weighted_fp16 = np.sum(fp16_pipes * durations)
    weighted_tensor = np.sum(tensor_pipes * durations)
    weighted_thrpt = np.sum(thrpts * durations)

    print(f'{app} {mode_str} {freq} sync={sync}')
    print(f'    {weighted_dram / 100:.4f}, {weighted_thrpt / 100:.4f}, {weighted_tensor / 100:.4f}, {weighted_fp16 / 100:.4f}')
    print()


for (app, mode, freq, sync) in combos:
    summarize(app, mode, freq, sync)

