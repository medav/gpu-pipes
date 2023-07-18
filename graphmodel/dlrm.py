from dataclasses import dataclass, field
from .common import *


B = 2048


def mlp(x : Node, out_widths : list[int], activate_lnorm=False, name='mlp'):
    widths = [x.shape[1]] + out_widths
    for i, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:])):
        x = linear(x, n_out, name=f'{name}.linear[{i}]')

        if i < len(widths) - 2:
            x = relu(x, name=f'{name}.relu[{i}]')

    if activate_lnorm:  x = layernorm(x, name=f'{name}.layernorm')
    return x



with Graph(name='DLRM') as g:
    dense_x = input((B, 13), name='dense_x')
    sparse_x = input((B, 26, 128), name='sparse_x').reshape((B, 26 * 128))

    # Bottom MLP
    bot_mlp_out = mlp(dense_x, [13, 512, 256, 128], name='botmlp')

    features = concat([bot_mlp_out, sparse_x], name='features').reshape((B, 27, 128))

    interact_bmm_out = generic(
        'interact_bmm',
        [features, features],
        (B, 27, 27),
        dtype=DType.float32)

    interact_out = generic(
        'interact_index',
        [interact_bmm_out],
        (B, 351),
        dtype=DType.float32)

    interact_cat = concat([bot_mlp_out, interact_out], name='interact_cat')

    # Top MLP
    top_mlp_out = mlp(interact_cat, [479, 1024, 1024, 512, 256, 1], name='topmlp')

sg0 = g.subgraph(r'botmlp..*', 'botmlp')
sg1 = g.subgraph(r'topmlp..*', 'topmlp')

pipelined_analysis(g, [sg0, sg1])


