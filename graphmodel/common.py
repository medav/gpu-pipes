from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

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


@dataclass(frozen=True)
class NamedObject:
    name : str

@dataclass(frozen=True)
class Tensor(NamedObject):
    shape : tuple[int]
    dtype : DType

    def __len__(self): return np.prod(self.shape)

    @property
    def nbytes(self): return self.dtype.bytewidth * len(self)


    def __repr__(self):
        shape_str = ', '.join(str(x) for x in self.shape)
        return f'{self.name}[{shape_str}]'


@dataclass(frozen=True)
class Node(NamedObject):
    inputs : list[Tensor]
    output : Tensor
    params : list[Tensor] = field(default_factory=list)

    @property
    def shape(self): return self.output.shape

@dataclass(frozen=True)
class Input(Tensor):
    def __init__(self, shape, dtype=DType.float32, name='input'):
        super().__init__(name, shape, dtype)

    def __repr__(self): return super().__repr__()

@dataclass(frozen=True)
class Parameter(Tensor):
    def __init__(self, shape, dtype=DType.float32, name='weight'):
        super().__init__(name, shape, dtype)


    def __repr__(self): return super().__repr__()

@dataclass(frozen=True)
class Linear(Node):
    def __init__(
        self,
        x : Tensor,
        w : Tensor,
        b : Tensor,
        name='linear'
    ):
        super().__init__(
            name,
            [x],
            Tensor(f'{name}.out', (x.shape[0], b.shape[0]), x.dtype),
            [w, b])

@dataclass(frozen=True)
class Relu(Node):
    def __init__(self, x, name='relu'):
        super().__init__(name, [x], Tensor(f'{name}.out', x.shape, x.dtype))

@dataclass(frozen=True)
class LayerNorm(Node):
    def __init__(self, x, gamma, beta, name='layernorm'):
        super().__init__(
            name, [x], Tensor(f'{name}.out', x.shape, x.dtype), [gamma, beta])

@dataclass(frozen=True)
class Add(Node):
    def __init__(self, x, y, name='add'):
        super().__init__(name, [x, y], Tensor(f'{name}.out', x.shape, x.dtype))

@dataclass(frozen=True)
class UnsortedSegmentSum(Node):
    def __init__(self, x : Tensor, seg_ids : Tensor, num_seg, name='unsorted_segment_sum'):
        super().__init__(
            name,
            [x, seg_ids],
            Tensor(f'{name}.out', (num_seg, x.shape[1]), x.dtype))


@dataclass(frozen=True)
class Index(Node):
    def __init__(self, x : Tensor, indices : Tensor, name='index'):
        super().__init__(
            name,
            [x, indices],
            Tensor(f'{name}.out', (indices.shape[0], x.shape[1]), x.dtype))

@dataclass(frozen=True)
class Concat(Node):
    def __init__(self, xs : list[Tensor], name='concat'):
        num_rows = xs[0].shape[0]
        dtype = xs[0].dtype
        out_shape = (num_rows, sum(x.shape[1] for x in xs),)
        super().__init__(
            name,
            xs,
            Tensor(f'{name}.out', out_shape, dtype))

_current_graph = None

class Graph(NamedObject):
    nodes : list[Node]
    edges : list[Tensor]

    def __init__(self, name='graph'):
        super().__init__(name)
        self.nodes = []
        self.edges = []

    def __enter__(self):
        global _current_graph
        _current_graph = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _current_graph
        _current_graph = None

def add_node(node):
    global _current_graph
    if _current_graph is None:
        raise RuntimeError('No active graph')
    _current_graph.nodes.append(node)

def add_edge(edge):
    global _current_graph
    if _current_graph is None:
        raise RuntimeError('No active graph')
    _current_graph.edges.append(edge)

def record_node(node):
    add_node(node)
    add_edge(node.output)
    print(f'node: {node.name}')
    print(f'    inputs: {node.inputs}')
    if len(node.params) > 0:
        print(f'    params: {node.params}')
    print(f'    output: {node.output}')
    print(f'    Total Bytes in: {sum(x.nbytes for x in node.inputs)}')
    print(f'    Total Bytes out: {node.output.nbytes}')
    print()
    return node.output

def linear(x, n_out, name='linear'):
    n_in = x.shape[1]
    w = Parameter((n_in, n_out), name=f'{name}.weight')
    b = Parameter((n_out,), name=f'{name}.bias')
    return record_node(Linear(x, w, b, name=name))


def relu(x, name='relu'):
    return record_node(Relu(x, name=name))

def layernorm(x, name='layernorm'):
    dim = x.shape[1]
    gamma = Parameter((dim,), name=f'{name}.gamma')
    beta = Parameter((dim,), name=f'{name}.beta')
    return record_node(LayerNorm(x, gamma, beta, name=name))

def add(x, y, name='add'):
    return record_node(Add(x, y, name=name))

def unsorted_segment_sum(x, seg_ids, num_seg, name='unsorted_segment_sum'):
    return record_node(UnsortedSegmentSum(x, seg_ids, num_seg, name=name))

def index(x, indices, name='index'):
    return record_node(Index(x, indices, name=name))

def concat(xs, name='concat'):
    return record_node(Concat(xs, name=name))



if __name__ == '__main__':
    x = Input((128, 1024))
    y = Linear(x, n_out=512)
    print(y.params)
    print(y.output.fullname)

