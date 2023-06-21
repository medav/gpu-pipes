from dataclasses import dataclass, field
from .common import *


@dataclass
class EdgeSet:
    ef : Node
    srcs : Node
    dsts : Node

@dataclass
class MultiGraph:
    nf : Node
    edges : list[EdgeSet]


def mlp(x : Node, out_widths : list[int], activate_lnorm=True, name='mlp'):
    widths = [x.shape[1]] + out_widths
    for i, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:])):
        x = linear(x, n_out, name=f'{name}.linear[{i}]')

        if i < len(widths) - 2:
            x = relu(x, name=f'{name}.relu[{i}]')

    if activate_lnorm:  x = layernorm(x, name=f'{name}.layernorm')
    return x

def update_edges(nf : Node, edge_set : EdgeSet, widths, name='edges'):
    src_nf = index(nf, edge_set.srcs, name=f'{name}.src_nf')
    dst_nf = index(nf, edge_set.dsts, name=f'{name}.dst_nf')
    concat_out = concat([src_nf, edge_set.ef, dst_nf], name=f'{name}.concat')
    return mlp(concat_out, widths, name=f'{name}.mlp')

def message_passing(i : int, x : MultiGraph):
    widths = [512, 512]

    new_edges = [
        EdgeSet(
            ef=update_edges(x.nf, edge_set, widths, name=f'mp[{i}].edges[{j}]'),
            srcs=edge_set.srcs,
            dsts=edge_set.dsts,
        )
        for j, edge_set in enumerate(x.edges)
    ]

    node_concat = [
        x.nf
    ] + [
        index(x.nf, edge_set.dsts, name=f'mp[{i}].edge_dsts[{j}]')
        for j, edge_set in enumerate(new_edges)
    ]

    node_concat_out = concat(node_concat, name=f'mp[{i}].nodes.concat')
    new_nf = mlp(node_concat_out, widths, name=f'mp[{i}].nodes.mlp')

    for ei in range(len(x.edges)):
        new_edges[ei].ef = add(
            new_edges[ei].ef, x.edges[ei].ef, name=f'mp[{i}].edge_resid[{ei}]')

    return MultiGraph(add(new_nf, x.nf, name=f'mp[{i}].node_resid'), new_edges)

B = 1
num_mp_steps = 16

with Graph(name='MLP') as g:
    x = MultiGraph(
        nf=input((B * 40962, 512), name='node_features'),
        edges=[
            EdgeSet(
                ef=input((B * 327660, 512), name='edge_features'),
                srcs=input((B * 327660,), name='src_nodes'),
                dsts=input((B * 327660,), name='dst_nodes'),
            ),
        ]
    )

    for i in range(num_mp_steps):
        x = message_passing(i, x)

g.dump_dot()

sgs = [
    g.subgraph(r'mp\[' + str(i) +  r'\].nodes.mlp..*')
    for i in range(num_mp_steps)
] + [
    g.subgraph(r'mp\[' + str(i) +  r'\].edges\[0\].mlp..*')
    for i in range(num_mp_steps)
]

pipelined_analysis(g, sgs)
