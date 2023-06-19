from dataclasses import dataclass, field
from .common import *


B = 65536


with Graph(name='NERF') as g:
    x = input((B, 60), name='x')
    d = input((B, 24), name='d')

    t = relu(linear(x, 256, 'lin1'), 'relu1')
    t = relu(linear(t, 256, 'lin2'), 'relu2')
    t = relu(linear(t, 256, 'lin3'), 'relu3')
    t = relu(linear(t, 256, 'lin4'), 'relu4')
    t = relu(linear(t, 256, 'lin5'), 'relu5')
    t = concat([t, d], 'concat0')
    t = relu(linear(t, 256, 'lin6'), 'relu6')
    t = relu(linear(t, 256, 'lin7'), 'relu7')
    t = relu(linear(t, 256, 'lin8'), 'relu8')

    # Radiance out
    r_out = relu(linear(t, 1, 'lin9'), 'relu9')

    # RGB out
    t = relu(linear(t, 256, 'lin10'), 'relu10')
    t = concat([t, d], 'concat1')
    t = relu(linear(t, 128, 'lin11'), 'relu11')
    rgb_out = relu(linear(t, 3, 'lin12'), 'relu12')

    out = concat([r_out, rgb_out], 'concat2')



sg0 = g.subgraph(r'.*')

pipelined_analysis(g, [sg0])


