#pragma once
#include "mpmcq.cuh"
#include "common.cuh"

#include <cuda_fp16.h>

struct MemoryReader {
    half * const base;
    const size_t stride;
    const size_t tile_stride;
    size_t offset;

    __device__ MemoryReader(half * const base, size_t stride, size_t tile_stride) :
        base(base), stride(stride), tile_stride(tile_stride), offset(0) {}

    __device__ TensorView read_acquire() { return {base + offset, (int)tile_stride}; }
    __device__ void read_release() { offset += stride; }
    __device__ void reset() { offset = 0; }
};

struct NullReader {
    __device__ TensorView read_acquire() { return {nullptr, 0}; }
    __device__ void read_release() {}
    __device__ void reset() {}
};

template <typename QT>
struct QueueReader {
    using Queue = QT;
    Queue& q;
    size_t seq_n;

    __device__ QueueReader(QT& q) : q(q), seq_n(0) { }
    __device__ TensorView read_acquire() { return q.read_wait(seq_n).as_view(); }
    __device__ void read_release() { q.read_commit(seq_n); seq_n++; }
    __device__ void reset() { seq_n = 0; }
};

template <typename QT>
struct SplitQueueReader {
    using Queue = QT;
    Queue& q;
    size_t seq_n;
    const size_t seq_off;
    const size_t seq_stride;

    __device__ SplitQueueReader(QT& q, size_t _seq_off, size_t _seq_stride) :
        q(q), seq_n(_seq_off), seq_off(_seq_off), seq_stride(_seq_stride) { }
    __device__ TensorView read_acquire() { return q.read_wait(seq_n).as_view(); }
    __device__ void read_release() { q.read_commit(seq_n); seq_n += seq_stride; }
    __device__ void reset() { seq_n = seq_off; }
};

struct MemoryWriter {
    half * base;
    const size_t stride;
    const size_t tile_stride;
    size_t offset;

    __device__ MemoryWriter(half * base, size_t stride, size_t tile_stride) :
        base(base), stride(stride), tile_stride(tile_stride), offset(0) {}

    __device__ TensorView write_acquire() { return {base + offset, (int)tile_stride}; }
    __device__ void write_release() { offset += stride; }
    __device__ void reset() { offset = 0; }
};

template <typename QT>
struct QueueWriter {
    using Queue = QT;
    Queue& q;
    size_t seq_n;

    __device__ QueueWriter(QT& q) : q(q), seq_n(0) { q.reset(); }
    __device__ TensorView write_acquire() { return q.write_wait(seq_n).as_view(); }
    __device__ void write_release() { q.write_commit(seq_n++); }
    __device__ void reset() { seq_n = 0; q.reset(); }
};

struct NullWriter {
    __device__ TensorView write_acquire() { return {nullptr, 0}; }
    __device__ void write_release() {}
    __device__ void reset() {}
};
