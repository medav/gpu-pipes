#pragma once

typedef unsigned long size_t;

#define CL 128
#define PAD_CLS 0
#define DEBUG 0
// #define NOSYNC

template<typename T, size_t N, size_t NP, size_t NC>
struct QueueSlot {
    using Item = T;
    int seq_n[CL/4];
    int read_done[CL/4];
    int write_done[CL/4];

#if PAD_CLS > 0
    char pad[CL * PAD_CLS];
#endif

    Item data;

    __device__ QueueSlot() : read_done({0}), write_done({0}), data() {}
    __device__ void reset(int seq_i) {
        seq_n[0] = seq_i;
        read_done[0] = 0;
        write_done[0] = 0;
    }

    __device__ int get_seq_n() const {
        return seq_n[0];
    }

    __device__ void commit_write() {
#ifndef NOSYNC
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            atomicAdd((int *)&write_done[0], 1);
        }
#endif
    }

    __device__ void commit_read() {
#ifndef NOSYNC
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int old = atomicAdd((int *)&read_done, (int)1);

            if (old == NC - 1) {
                read_done[0] = 0;
                write_done[0] = 0;
                atomicAdd(&seq_n[0], N);
            }
        }
#endif
    }

};

template<typename T, size_t N, size_t NP, size_t NC>
class MpmcRingQueue {
public:
    using Slot = QueueSlot<T, N, NP, NC>;

private:
    Slot slots[N];

public:

    __device__ void reset() {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (int i = 0; i < N; i++) {
                slots[i].reset(i);
            }
        }
        __syncthreads();
    }

    __device__ MpmcRingQueue() : slots{} { }

    __device__ Slot& allocate(int seq_n) {
        Slot& slot = slots[seq_n % N];
#ifndef NOSYNC
        if (threadIdx.x == 0) {
            while (atomicAdd(&slot.seq_n[0], (int)0) != seq_n) { }
        }
        __syncwarp();
#endif
        return slot;
    }

    __device__ typename Slot::Item& write_wait(int seq_n) {
        Slot& slot = allocate(seq_n);
        return slot.data;
    }

    __device__ void write_commit(int seq_n) {
        Slot& slot = slots[seq_n % N];
        slot.commit_write();
    }

    __device__ typename Slot::Item& read_wait(int seq_n) {
        Slot& slot = allocate(seq_n);
#ifndef NOSYNC
        if (threadIdx.x == 0) {
            while (atomicAdd(&slot.write_done[0], (int)0) != NP) {  }
        }
        __syncwarp();
#endif
        return slot.data;
    }

    __device__ void read_commit(int seq_n) {
        Slot& slot = slots[seq_n % N];
        slot.commit_read();
    }
};
