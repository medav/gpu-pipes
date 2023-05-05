#pragma once
#include "mpmcq.cuh"


#include <cuda_fp16.h>

struct MemoryReader {
    half * const base;
    const size_t stride;
    size_t offset;

    __device__ MemoryReader(half * const base, size_t stride) :
        base(base), stride(stride), offset(0) {}

    __device__ half* read_acquire() { return base + offset; }
    __device__ void read_release() { offset += stride; }
    __device__ void reset() { offset = 0; }
};

struct NullReader {
    __device__ half* read_acquire() { return nullptr; }
    __device__ void read_release() {}
    __device__ void reset() {}
};

template <typename QT>
struct QueueReader {
    using Queue = QT;
    Queue& q;
    size_t seq_n;

    __device__ QueueReader(Queue& q) : q(q), seq_n(0) { }
    __device__ half* read_acquire() { return q.read_wait(seq_n).as_ptr(); }
    __device__ void read_release() { q.read_commit(seq_n); seq_n++; }
    __device__ void reset() { seq_n = 0; }
};

struct MemoryWriter {
    half * base;
    const size_t stride;
    size_t offset;

    __device__ MemoryWriter(half * base, size_t stride) :
        base(base), stride(stride), offset(0) {}

    __device__ half* write_acquire() { return base + offset; }
    __device__ void write_release() { offset += stride; }
    __device__ void reset() { offset = 0; }
};

template <typename QT>
struct QueueWriter {
    using Queue = QT;
    Queue& q;
    size_t seq_n;

    __device__ QueueWriter(Queue& q) : q(q), seq_n(0) { q.reset(); }
    __device__ half* write_acquire() { return q.write_wait(seq_n).as_ptr(); }
    __device__ void write_release() { q.write_commit(seq_n++); }
    __device__ void reset() { seq_n = 0; q.reset(); }
};
