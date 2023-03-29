#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#include "mpmcq.cuh"
#include "warpgemm.cuh"
#include "utils.cuh"

#define NUM_ITERS 1
#define M 1024 * 1024
#define MBLK 32
#define D 128

#define WARPS_PER_BLOCK 8

#define SMEM (WARPS_PER_BLOCK * MBLK * D * 2 + D * D) * sizeof(half)

struct QueueEntry {
    half buf[MBLK][D];

    __device__ QueueEntry() : buf{(half)0.0f} {}
};

struct Problem {
    half in[M][D];
    half w1[D][D];
    half w2[D][D];
    half out[M][D];
};

struct Pipeline {
    using Queue1 = MpmcRingQueue<QueueEntry, 16, 1, 1>;
    Queue1 q1;
};

using GemmOp = cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<MBLK, D, D>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, 32>,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, 32>,
    cutlass::layout::RowMajor
    >;

__global__ void init_pipe(Pipeline *pipe) {
    new (&pipe->q1) Pipeline::Queue1();
}

__global__ void init_prob(Problem *prob) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < D; j++) {
            prob->in[i][j] = (half)1.0f;
            prob->out[i][j] = (half)0.0f;
        }
    }

    for (size_t i = 0; i < D; i++) {
        for (size_t j = 0; j < D; j++) {
            prob->w1[i][j] = (half)1.0f;
            prob->w2[i][j] = (half)1.0f;
        }
    }
}

__device__ void sm0(half * smem, Problem * prob, Pipeline *pipe) {
    GemmOp gemm_op;
    auto this_block = cooperative_groups::this_thread_block();

    const size_t mblk = MBLK;
    const size_t lane_id = threadIdx.x;
    const size_t warp_id = threadIdx.y;
    const ssize_t num_blocks = M / mblk;

    half * inbuf  = (half *)(smem + MBLK * D * warp_id);
    half * w1     = (half *)(smem + WARPS_PER_BLOCK * MBLK * D);
    half * outbuf = (half *)(smem + WARPS_PER_BLOCK * MBLK * D + D * D);

    cooperative_groups::memcpy_async(this_block, w1, &prob->w1[0][0], D * D * sizeof(half));
    this_block.sync();
    __syncwarp();

    size_t seq_n = warp_id;
    auto this_warp = cooperative_groups::tiled_partition<32>(this_block);
    auto& q = pipe->q1;

    cooperative_groups::memcpy_async(
        this_warp,
        (void *)inbuf,
        (void *)&prob->in[seq_n * mblk][0],
        MBLK * D * sizeof(half)
    );
    this_warp.sync();

    for (; seq_n < num_blocks; seq_n += WARPS_PER_BLOCK) {
        // if (lane_id == 0) { printf("sm0: %d\n", seq_n); } __syncwarp();

        gemm_op(
            GemmOp::TensorRefA((GemmOp::MmaWarp::ElementA *)inbuf, D),
            GemmOp::TensorRefB((GemmOp::MmaWarp::ElementB *)w1, D),
            GemmOp::TensorRefC((GemmOp::MmaWarp::ElementC *)outbuf, D),
            lane_id
        );

        // if (lane_id == 0) { printf("2\n"); } __syncwarp();
        auto& slot = q.write_wait(seq_n, lane_id);

        // if (lane_id == 0) { printf("3\n"); } __syncwarp();
        cooperative_groups::memcpy_async(
            this_warp,
            (void *)&slot.data.buf[0][0],
            (void *)outbuf,
            MBLK * D * sizeof(half)
        );

        // if (lane_id == 0) { printf("4\n"); } __syncwarp();
        if (seq_n + WARPS_PER_BLOCK < num_blocks) {
            cooperative_groups::memcpy_async(
                this_warp,
                (void *)inbuf,
                (void *)&prob->in[(seq_n + WARPS_PER_BLOCK) * mblk],
                MBLK * D * sizeof(half)
            );
        }

        // if (lane_id == 0) { printf("5\n"); } __syncwarp();
        this_warp.sync();
        slot.commit_write(lane_id);
    }

}

__device__ void sm1(half * smem, Problem * prob, Pipeline *pipe) {
    GemmOp gemm_op;
    auto this_block = cooperative_groups::this_thread_block();

    const size_t mblk = MBLK;
    const size_t lane_id = threadIdx.x;
    const size_t warp_id = threadIdx.y;
    const ssize_t num_blocks = M / mblk;

    half * inbuf  = (half *)(smem + MBLK * D * warp_id);
    half * w2     = (half *)(smem + WARPS_PER_BLOCK * MBLK * D);
    half * outbuf = (half *)(smem + WARPS_PER_BLOCK * MBLK * D + D * D);

    cooperative_groups::memcpy_async(this_block, w2, &prob->w2[0][0], D * D * sizeof(half));
    this_block.sync();
    __syncwarp();

    size_t seq_n = warp_id;
    auto this_warp = cooperative_groups::tiled_partition<32>(this_block);
    auto& q = pipe->q1;

    auto& slot = q.read_wait(seq_n, lane_id);
    cooperative_groups::memcpy_async(
        this_warp,
        (void *)inbuf,
        (void *)&slot.data.buf[0][0],
        MBLK * D * sizeof(half)
    );
    this_warp.sync();
    slot.commit_read(lane_id);

    for (; seq_n < num_blocks; seq_n += WARPS_PER_BLOCK) {
        // if (lane_id == 0) { printf("        sm1: %d\n", seq_n); } __syncwarp();

        gemm_op(
            GemmOp::TensorRefA((GemmOp::MmaWarp::ElementA *)inbuf, D),
            GemmOp::TensorRefB((GemmOp::MmaWarp::ElementB *)w2, D),
            GemmOp::TensorRefC((GemmOp::MmaWarp::ElementC *)outbuf, D),
            lane_id
        );

        // if (lane_id == 0) { printf("    2\n"); } __syncwarp();
        cooperative_groups::memcpy_async(
            this_warp,
            (void *)&prob->out[seq_n * mblk][0],
            (void *)outbuf,
            MBLK * D * sizeof(half)
        );

        // if (lane_id == 0) { printf("    3\n"); } __syncwarp();
        if (seq_n + WARPS_PER_BLOCK < num_blocks) {
            auto& slot = q.read_wait(seq_n + WARPS_PER_BLOCK,  lane_id);
            cooperative_groups::memcpy_async(
                this_warp,
                (void *)inbuf,
                (void *)&slot.data.buf[0][0],
                MBLK * D * sizeof(half)
            );

            // if (lane_id == 0) { printf("    4\n"); } __syncwarp();
            this_warp.sync();
            slot.commit_read(lane_id);
        }
        else {
            // if (lane_id == 0) { printf("    4\n"); } __syncwarp();
            this_warp.sync();
        }


    }
}

__global__ void kernel(Problem * prob, Pipeline *pipe) {
    int bid = blockIdx.x;
    extern __shared__ half smem[];

    for (int i = 0; i < NUM_ITERS; i++) {
        // if (threadIdx.x == 0 && bid == 0) {
        //     printf("iter %d\n", i);
        // }

        if (bid == 0) { sm0(smem, prob, pipe); }
        else {
            sm1(smem, prob, pipe);

        }

    }
}

int main() {
    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));


    Pipeline * pipe;
    cudaErrCheck(cudaMallocManaged(&pipe, sizeof(Pipeline)));

    Problem * prob;
    cudaErrCheck(cudaMallocManaged(&prob, sizeof(Problem)));

    printf("Init...\n");
    init_pipe<<<1, 1>>>(pipe);
    init_prob<<<1, 1>>>(prob);
    cudaErrCheck(cudaDeviceSynchronize());

    dim3 block(32, WARPS_PER_BLOCK);

    printf("SMEM: %lu\n", SMEM);

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<2, block, SMEM>>>(prob, pipe);
        }
    );

    printf("gemm took %fms\n", time_ms);

    float flops_v1 = 2.0f * M * D * D * 2 * NUM_ITERS;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    return 0;
}
