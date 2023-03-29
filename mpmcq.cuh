#pragma once

typedef unsigned long size_t;

template<typename T, size_t N, size_t NP, size_t NC>
struct QueueSlot {
    using Item = T;
    volatile size_t seq_n;
    volatile int read_done;
    volatile int write_done;
    Item data;

    __device__ QueueSlot() : read_done(0), write_done(0), data() {}


    __device__ void reset(size_t seq_i) {
        seq_n = seq_i;
        read_done = 0;
        write_done = 0;
    }

    __device__ bool filled() const { return write_done == NC; }
    __device__ bool drained() const { return read_done == NP && write_done == NC; }

    __device__ void commit_write(int lane_id) {
        if (lane_id == 0) {
            // printf("COMMIT WRITE seq_n: %lu, slot: %lu\n", seq_n, seq_n % N);
            atomicAdd((int *)&write_done, 1);
        }
        __syncwarp();
    }

    __device__ void commit_read(int lane_id) {
        if (lane_id == 0) {
            // printf("COMMIT READ seq_n: %lu, slot: %lu\n", seq_n, seq_n % N);
            int old = atomicAdd((int *)&read_done, 1);

            if (old == NC - 1) {
                seq_n += N;
                read_done = 0;
                write_done = 0;
            }
        }
        __syncwarp();
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
        for (int i = 0; i < N; i++) {
            slots[i].reset(i);
        }
    }

    __device__ MpmcRingQueue() : slots{} {
        reset();
    }

    __device__ Slot& allocate(int seq_n, int lane_id) {
        Slot& slot = slots[seq_n % N];
        if (lane_id == 0) {
            while (slot.seq_n != seq_n) { /* spin */ }
        }
        __syncwarp();
        return slot;
    }

    __device__ Slot& write_wait(int seq_n, int lane_id) {
        // if (lane_id == 0) printf("WRITE seq_n: %d, slot: %d\n", seq_n, (int)(seq_n % N));
        Slot& slot = allocate(seq_n, lane_id);
        return slot;
    }

    __device__ Slot& read_wait(int seq_n, int lane_id) {
        // if (lane_id == 0) printf("READ seq_n: %d, slot: %d\n", seq_n, (int)(seq_n % N));
        Slot& slot = allocate(seq_n, lane_id);
        if (lane_id == 0) {
            while (!slot.filled()) { /* spin */ }
        }
        __syncwarp();
        return slot;
    }
};
