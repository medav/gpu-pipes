#include "mgn_node_pipe.cuh"
#include "mgn_node_mlp0s0.cuh"
#include "mgn_node_mlp0s1.cuh"
#include "mgn_node_mlp0s2.cuh"

using Block = cutlass::gemm::GemmShape<MgnNodeMlp::mblk, MgnNodeMlp::d, 128>;
using Mlp0Stage0 = Mlp0s0<Block, MgnNodeMlp::num_warps>;
using Mlp0Stage1 = Mlp0s1<Block, MgnNodeMlp::num_warps>;
using Mlp0Stage2 = Mlp0s2<Block, MgnNodeMlp::num_warps>;

const size_t max_smem = std::max({
    Mlp0Stage0::smem_bytes,
    Mlp0Stage1::smem_bytes,
    Mlp0Stage2::smem_bytes
});

__device__ void mlp0_sm0(half *smem, MgnNodeMlp *prob) {
    Mlp0Stage0 pipe(smem, prob->m, threadIdx.x, threadIdx.y);
    pipe.run(&prob->w1[0][0][0], &prob->in[0][0][0], prob->q1[0]);
}

__device__ void mlp0_sm1(half *smem, MgnNodeMlp *prob) {
    Mlp0Stage1 pipe(smem, prob->m, threadIdx.x, threadIdx.y);
    pipe.run(&prob->w1[1][0][0], &prob->in[1][0][0], prob->q1[0], prob->q1[1]);
}

__device__ void mlp0_sm2(half *smem, MgnNodeMlp *prob) {
    Mlp0Stage2 pipe(smem, prob->m, threadIdx.x, threadIdx.y);
    pipe.run(&prob->w1[2][0][0], &prob->in[2][0][0], prob->q1[1], prob->q12);
}

template<typename QT>
__device__ void consume_dummy(QT& q, size_t num_iters) {

    for (size_t i = 0; i < num_iters; i++) {
        q.read_wait(i);
        q.read_commit(i);
    }
}


__global__ void kernel(MgnNodeMlp * prob) {
    extern __shared__ half smem[];

    size_t pipe_col = blockIdx.x;

    switch (pipe_col) {
        case 0:
            mlp0_sm0(smem, prob);
            break;
        case 1:
            // mlp0_sm1(smem, prob);
            break;
        case 2:
            // mlp0_sm2(smem, prob);
            break;
        case 3:
            consume_dummy(prob->q1[0], MgnNodeMlp::m / MgnNodeMlp::mblk);
            break;
        case 4:

            break;

        default: return;
    }
}

int main() {

    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));

    MgnNodeMlp * prob;
    cudaErrCheck(cudaMalloc(&prob, sizeof(MgnNodeMlp) + 128));
    // Align prob
    prob = (MgnNodeMlp*)(((size_t)prob + 0x3F) & ~0x3F);

    printf("Init...\n");
    init_prob<<<1, 128>>>(prob);
    cudaErrCheck(cudaDeviceSynchronize());

    dim3 grid(5, 1);
    dim3 block(32, MgnNodeMlp::num_warps);

    printf("SMEM: %lu\n", max_smem);

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<grid, block, max_smem>>>(prob);
        }
    );

    printf("gemm took %fms\n", time_ms);

    float flops_v1 = 2.0f * MgnNodeMlp::m * (3 * MgnNodeMlp::d) * MgnNodeMlp::d;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    return 0;
}
