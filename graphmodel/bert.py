from dataclasses import dataclass, field
from .common import *


S = 512
B = 64

def transformer_layer(x, nheads=2, nhidden=128, nmlp=512, prefix='layer'):
    q = linear(x, nhidden, name=f'{prefix}.attn.q').reshape((B, nheads, S, 64))
    k = linear(x, nhidden, name=f'{prefix}.attn.k').reshape((B, nheads, S, 64))
    v = linear(x, nhidden, name=f'{prefix}.attn.v').reshape((B, nheads, S, 64))

    qk = matmul(q, k, name=f'{prefix}.attn.qk')
    qk_scale = scale(qk, name=f'{prefix}.attn.qk_scale')
    attn_probs = softmax(qk_scale, name=f'{prefix}.attn.attn_probs')

    attn_out = matmul(attn_probs, v, name=f'{prefix}.attn.attn_out').reshape((B, S, nhidden))

    y0w0 = linear(attn_out, nhidden, name=f'{prefix}.attn.linear[0]')
    y0w0_ln = layernorm(y0w0, name=f'{prefix}.attn.ln[0]')

    y1w1 = linear(y0w0_ln, nmlp, name=f'{prefix}.mlp.linear[1]')
    y1w1_gelu = relu(y1w1, name=f'{prefix}.mlp.gelu[0]')

    y2w2 = linear(y1w1_gelu, nhidden, name=f'{prefix}.mlp.linear[2]')
    y2w2_ln = layernorm(add(y0w0_ln, y2w2, name=f'{prefix}.mlp.add'), name=f'{prefix}.mlp.ln[2]')

    return y2w2_ln

with Graph(name='TransformerLayer') as g:
    x = input((B, S, 128), name='x')
    y1 = transformer_layer(x, prefix='layer[0]')
    y2 = transformer_layer(y1, prefix='layer[1]')



sg0 = g.subgraph(r'layer\[0\].mlp..*', 'bert_ffn')
sg1 = g.subgraph(r'layer\[1\].mlp..*', 'bert_ffn')

pipelined_analysis(g, [sg0, sg1])


