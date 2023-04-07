#pragma once

typedef unsigned long size_t;


__device__ bool is_master() {
    return threadIdx.x == 0 && threadIdx.y == 0;
}

#define DEBUG 0

template<typename T, size_t N, size_t NP, size_t NC>
struct QueueSlot {
    using Item = T;
    int seq_n;
    int read_done;
    int write_done;
    Item data;


    __device__ QueueSlot() : read_done(0), write_done(0), data() {}


    __device__ void reset(size_t seq_i) {
        seq_n = seq_i;
        read_done = 0;
        write_done = 0;
    }

    __device__ bool filled() const { return write_done == NC; }
    __device__ bool drained() const { return read_done == NP && write_done == NC; }

    __device__ void commit_write() {
        if (is_master()) {
            atomicAdd((int *)&write_done, 1);
        }
        __syncthreads();
    }

    __device__ void commit_read() {
        if (is_master()) {
            int old = atomicAdd((int *)&read_done, (int)1);

            if (old == NC - 1) {
                read_done = 0;
                write_done = 0;
                atomicAdd(&seq_n, N);
            }
        }
        __syncthreads();
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
        if (is_master()) {
            for (int i = 0; i < N; i++) {
                slots[i].reset(i);
            }
        }
        __syncthreads();
    }

    __device__ MpmcRingQueue() : slots{} { }

    __device__ Slot& allocate(int seq_n) {
        Slot& slot = slots[seq_n % N];
        if (is_master()) {
            while (atomicAdd(&slot.seq_n, (int)0) != seq_n) { }
        }
        __syncthreads();
        return slot;
    }

    __device__ Slot& write_wait(int seq_n) {
        if (DEBUG && is_master()) printf("WRITE ALLOC: %d, %d\n", seq_n, seq_n % N);
        Slot& slot = allocate(seq_n);
        return slot;
    }

    __device__ void write_commit(int seq_n) {
        if (DEBUG && is_master()) printf("WRITE COMMIT: %d, %d\n", seq_n, seq_n % N);
        Slot& slot = allocate(seq_n);
        slot.commit_write();
    }

    __device__ Slot& read_wait(int seq_n) {
        if (DEBUG && is_master()) printf("READ ALLOC: %d, %d\n", seq_n, seq_n % N);
        Slot& slot = allocate(seq_n);
        if (is_master()) {
            while (atomicAdd(&slot.write_done, (int)0) != NP) { }
        }

        __syncthreads();
        return slot;
    }

    __device__ void read_commit(int seq_n) {
        if (DEBUG && is_master()) printf("READ COMMIT: %d, %d\n", seq_n, seq_n % N);
        Slot& slot = allocate(seq_n);
        slot.commit_read();
    }
};
