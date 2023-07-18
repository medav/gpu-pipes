from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
import re

class DType(IntEnum):
    float32 = 0
    float16 = 1
    int32 = 2
    int64 = 3

    @property
    def bytewidth(self):
        return {
            DType.float32: 4,
            DType.float16: 2,
            DType.int32: 4,
            DType.int64: 8,
        }[self]

@dataclass
class Node:
    name : str
    inputs : list["Node"]
    params : list["Node"]
    shape : tuple[int]
    dtype : DType = DType.float32

    def __len__(self): return np.prod(self.shape)
    def __hash__(self): return hash(self.name)
    def __repr__(self): return self.name

    @property
    def dot_format(self): return f'label="{self.name}",shape=box'

    @property
    def in_bytes(self): return sum(x.out_bytes for x in self.inputs)

    @property
    def param_bytes(self): return sum(x.out_bytes for x in self.params)

    @property
    def out_bytes(self): return self.dtype.bytewidth * len(self)

    def reshape(self, shape):
        assert np.prod(shape) == np.prod(self.shape), f'Cannot reshape {self.shape} to {shape}'
        self.shape = shape
        return self

class Input(Node):
    def __init__(self, shape, dtype=DType.float32, name='input'):
        super().__init__(name, [], [], shape, dtype)

    @property
    def dot_format(self): return f'label="{self.name}"'

class Parameter(Node):
    def __init__(self, shape, dtype=DType.float32, name='param'):
        super().__init__(name, [], [], shape, dtype)

    @property
    def dot_format(self): return f'label="{self.name}",shape=plain'

class GenericOp(Node):
    def __init__(self, name, inputs, shape, dtype, params=None):
        super().__init__(name, inputs, params, shape, dtype)

class Matmul(Node):
    def __init__(self, x : Node, y : Node, name='matmul'):
        super().__init__(name, [x, y], [], (*x.shape[:-1], y.shape[-1]), x.dtype)

class Linear(Node):
    def __init__(self, x : Node, w : Node, b : Node, name='linear'):
        super().__init__(name, [x], [w, b], (*x.shape[:-1], w.shape[-1]), x.dtype)

class Relu(Node):
    def __init__(self, x : Node, name='relu'):
        super().__init__(name, [x], [], x.shape, x.dtype)

class Softmax(Node):
    def __init__(self, x : Node, name='softmax'):
        super().__init__(name, [x], [], x.shape, x.dtype)

class LayerNorm(Node):
    def __init__(self, x : Node, gamma : Node, beta : Node, name='layernorm'):
        super().__init__(name, [x], [gamma, beta], x.shape, x.dtype)

class Add(Node):
    def __init__(self, x : Node, y : Node, name='add'):
        super().__init__(name, [x, y], [], x.shape, x.dtype)

class Scale(Node):
    def __init__(self, x : Node, scale : Node, name='scale'):
        super().__init__(name, [x, scale], [], x.shape, x.dtype)

class UnsortedSegmentSum(Node):
    def __init__(self, x : Node, seg_ids : Node, num_seg, name='uss'):
        super().__init__(name, [x, seg_ids], [], (num_seg, x.shape[1]), x.dtype)

class Index(Node):
    def __init__(self, x : Node, indices : Node, name='index'):
        super().__init__(name, [x, indices], [], (indices.shape[0], x.shape[1]), x.dtype)

class Concat(Node):
    def __init__(self, xs : list[Node], name='concat'):
        out_shape = (xs[0].shape[0], sum(x.shape[1] for x in xs))
        super().__init__(name, xs, [], out_shape, xs[0].dtype)

_current_graph = None

class Graph:
    nodes : dict[str, Node]

    def __init__(self, nodes=None, name='graph'):
        self.name = name
        self.nodes = nodes if nodes is not None else dict()
        self.fused_blocks = []

    def __enter__(self):
        global _current_graph
        _current_graph = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _current_graph
        _current_graph = None

    def subgraph(self, pattern : str, name=None):
        name = name if name is not None else self.name
        return Graph(nodes={
            name: node
            for name, node in self.nodes.items()
            if re.match(pattern, name)
        }, name=f'fused_{name}')


    def dump_dot(self):
        print('digraph {')
        name_map = {}

        for i, node in enumerate(self.nodes.values()):
            nname = f'node_{i}'
            name_map[node.name] = nname
            print(f'    {nname} [{node.dot_format}];')

        for node in self.nodes.values():
            for t in node.inputs:
                if t.name not in name_map: continue
                print(f'    {name_map[t.name]} -> {name_map[node.name]};')

            for t in node.params:
                if t.name not in name_map: continue
                print(f'    {name_map[t.name]} -> {name_map[node.name]};')

        print('}')

    @property
    def total_weight_bytes(self):
        return sum(x.param_bytes for x in self.nodes.values())

    @property
    def total_iact_bytes(self):
        tot_input_bytes = 0

        for n in self.nodes.values():
            node_iact_bytes = 0
            if not isinstance(n, Input):
                for i in n.inputs:
                    if not isinstance(i, Parameter):
                        node_iact_bytes += i.out_bytes

            # print(f'{n.name}: {node_iact_bytes / 2**20} MB')
            tot_input_bytes += node_iact_bytes

        return tot_input_bytes

    @property
    def total_read_bytes(self):
        return self.total_weight_bytes + self.total_iact_bytes


    @property
    def pipelined_iact_bytes(self):
        tot_input_bytes = sum(
            x.out_bytes
            for x in self.nodes.values()
            if isinstance(x, Input) and not isinstance(x, Parameter)
        )

        for n in self.nodes.values():
            for i in n.inputs:
                if i.name not in self.nodes:
                    tot_input_bytes += i.out_bytes

        return tot_input_bytes

    @property
    def pipelined_read_bytes(self):
        return self.total_weight_bytes + self.pipelined_iact_bytes


    @property
    def total_write_bytes(self):
        return sum(x.out_bytes for x in self.nodes.values() if not isinstance(x, Input))

    @property
    def total_bytes(self):
        return self.total_read_bytes + self.total_write_bytes

    @property
    def output_node(self):
        all_nodes = set(self.nodes.values())
        for n in self.nodes.values():
            for i in n.inputs:
                if i in all_nodes: all_nodes.remove(i)

            for i in n.params:
                if i in all_nodes: all_nodes.remove(i)

        assert len(all_nodes) == 1, all_nodes
        return all_nodes.pop()

    @property
    def pipelined_write_bytes(self):
        return self.output_node.out_bytes

    @property
    def pipelined_bytes(self):
        return self.pipelined_read_bytes + self.pipelined_write_bytes

    @property
    def pipelined_bytes_saved(self):
        return self.total_bytes - self.pipelined_bytes

    def pipelined_analysis(self):
        print('Pipelined Analysis')

        unpipe_total_bytes = self.total_bytes

        print(f'+ Non-Pipelined Bytes In: {self.total_iact_bytes / 2**20} MB')
        print(f'+ Non-Pipelined Weight Bytes: {self.total_weight_bytes / 2**10} KB')
        print(f'+ Non-Pipelined Bytes Out: {self.total_write_bytes / 2**20} MB')
        print(f'+ Non-Pipelined Traffic: {unpipe_total_bytes} B')
        print('|')

        pipe_total_bytes = self.pipelined_bytes

        print(f'+ Pipelined Bytes In: {self.pipelined_read_bytes / 2**20} MB')
        print(f'+ Pipelined Weight Bytes: {self.total_weight_bytes / 2**10} KB')
        print(f'+ Pipelined Bytes Out: {self.output_node.out_bytes / 2**20} MB')
        print(f'+ Pipelined Traffic: {pipe_total_bytes} B')
        print('|')

        print(f'+ Pipelined Traffic Savings: {unpipe_total_bytes / pipe_total_bytes:.2f} x')




def add_node(node):
    global _current_graph
    if _current_graph is None: raise RuntimeError('No active graph')
    if node.name in _current_graph.nodes: raise RuntimeError(f'Node {node.name} already exists')
    _current_graph.nodes[node.name] = node


def record_node(node : Node):
    add_node(node)
    print(f'node: {node.name}')
    print(f'    inputs: {node.inputs}')
    if len(node.params) > 0:
        print(f'    params: {node.params}')
    print(f'    output: {node.shape}')
    print(f'    Total Bytes in: {node.in_bytes}')
    print(f'    Total Bytes out: {node.out_bytes}')
    print()
    return node

def generic(name, inputs, shape, dtype, params=[]):
    return record_node(GenericOp(name, inputs, shape, dtype, params))

def input(shape, dtype=DType.float32, name='input'): return record_node(Input(shape, dtype, name=name))
def matmul(x : Node, y : Node, name='matmul'): return record_node(Matmul(x, y, name=name))

def linear(x : Node, n_out, name='linear'):
    n_in = x.shape[-1]
    w = record_node(Parameter((n_in, n_out), name=f'{name}.weight'))
    b = record_node(Parameter((n_out,), name=f'{name}.bias'))
    return record_node(Linear(x, w, b, name=name))

def relu(x : Node, name='relu'): return record_node(Relu(x, name=name))
def softmax(x : Node, name='softmax'): return record_node(Softmax(x, name=name))

def layernorm(x : Node, name='layernorm'):
    dim = x.shape[1]
    gamma = record_node(Parameter((dim,), name=f'{name}.gamma'))
    beta = record_node(Parameter((dim,), name=f'{name}.beta'))
    return record_node(LayerNorm(x, gamma, beta, name=name))

def add(x : Node, y : Node, name='add'): return record_node(Add(x, y, name=name))
def scale(x : Node, name='scale'):
    s = record_node(Parameter((1,), name=f'{name}.scale'))
    return record_node(Scale(x, s, name=name))

def unsorted_segment_sum(x : Node, seg_ids, num_seg, name='unsorted_segment_sum'):
    return record_node(UnsortedSegmentSum(x, seg_ids, num_seg, name=name))

def index(x : Node, indices, name='index'): return record_node(Index(x, indices, name=name))
def concat(xs : list[Node], name='concat'): return record_node(Concat(xs, name=name))

if __name__ == '__main__':
    x = Input((128, 1024))
    y = Linear(x, n_out=512)
    print(y.params)
    print(y.output.fullname)

def pipelined_analysis(g : Graph, sgs : list[Graph]):

    for sg in sgs:
        print('Subgraph: ', sg.name)
        sg.pipelined_analysis()
        print()

    tot_bytes_pipe = sum(sg.total_bytes for sg in sgs)
    tot_bytes_saved = sum(sg.pipelined_bytes_saved for sg in sgs)

    print(f'Graph total iact bytes: {g.total_iact_bytes / 2**20} MB')
    print(f'Graph total weight bytes: {g.total_weight_bytes / 2**20} MB')
    print(f'Graph total write bytes: {g.total_write_bytes / 2**20} MB')

    print(f'Total DRAM Traffic: {g.total_bytes / 2**20} MB')
    print(f'Subgraph DRAM Traffic: {tot_bytes_pipe / 2**20} MB')
    print(f'Total traffic saved: {tot_bytes_saved / 2**20} MB')
    print(f'DRAM traffic with pipelined subgraphs: {(g.total_bytes - tot_bytes_saved) / 2**20} MB')
    print(f'Subgraph DRAM traffic with pipelined subgraphs: {(tot_bytes_pipe - tot_bytes_saved) / 2**20} MB')
