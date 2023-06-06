from dataclasses import dataclass, field
from .common import *


S = 512
B = 64

with Graph(name='TransformerLayer') as g:
    x = input((B, S, 1024), name='x')

    q = linear(x, 1024, name='layer[0].attn.q').reshape((B, 16, S, 64))
    k = linear(x, 1024, name='layer[0].attn.k').reshape((B, 16, S, 64))
    v = linear(x, 1024, name='layer[0].attn.v').reshape((B, 16, S, 64))

    qk = matmul(q, k, name='layer[0].attn.qk')
    qk_scale = scale(qk, name='layer[0].attn.qk_scale')
    attn_probs = softmax(qk_scale, name='layer[0].attn.attn_probs')

    attn_out = matmul(attn_probs, v, name='layer[0].attn.attn_out').reshape((B, S, 1024))

    y0w0 = linear(attn_out, 1024, name='layer[0].mlp.linear[0]')
    y0w0_ln = layernorm(y0w0, name='layer[0].mlp.ln[0]')

    y1w1 = linear(y0w0_ln, 4096, name='layer[0].mlp.linear[1]')
    y1w1_gelu = relu(y1w1, name='layer[0].mlp.gelu[0]')

    y2w2 = linear(y1w1_gelu, 1024, name='layer[0].mlp.linear[2]')
    y2w2_ln = layernorm(add(y0w0_ln, y2w2, name='layer[0].mlp.add'), name='layer[0].mlp.ln[2]')



g.dump_dot()

sg = g.subgraph(r'layer\[0\].mlp..*')

sg.pipelined_analysis()




