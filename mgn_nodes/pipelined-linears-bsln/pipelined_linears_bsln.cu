#include "mgn_node_pipe.cuh"
#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_gemm_bias_layer_norm.cuh"

#include "utils.cuh"

using ProblemShape = cutlass::gemm::GemmShape<MgnNodeMlp::mblk, 128, 128>;

const size_t max_smem = std::max({
    sizeof(typename PipeGemm<ProblemShape>::SmemBuffers),
    sizeof(typename PipeGemmBias<ProblemShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<ProblemShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasLayerNorm<ProblemShape>::SmemBuffers)
});

const size_t max_warps = std::max({
    PipeGemm<ProblemShape>::num_warps,
    PipeGemmBias<ProblemShape>::num_warps,
    PipeGemmBiasRelu<ProblemShape>::num_warps,
    PipeGemmBiasLayerNorm<ProblemShape>::num_warps
});

__device__ void mlp0_sm0(MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    if (threadIdx.y >= PipeGemm<ProblemShape>::num_warps) return;

    Input input(&prob->in[0][row * prob->mi][0], mblk * prob->d);
    Accum accum;
    Output output(prob->qs.q01[row]);

    input.reset();
    pipe_gemm<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >(&prob->w1[0][0][0], input, accum, output, num_iters);

}

__device__ void mlp0_sm1(MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = QueueReader<MgnNodeMlp::Queue>;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    if (threadIdx.y >= PipeGemm<ProblemShape>::num_warps) return;

    Input input(&prob->in[1][row * prob->mi][0], mblk * prob->d);
    Accum accum(prob->qs.q01[row]);
    Output output(prob->qs.q12[row]);

    input.reset();
    pipe_gemm<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >(&prob->w1[1][0][0], input, accum, output, num_iters);
}

__device__ void mlp0_sm2(MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = QueueReader<MgnNodeMlp::Queue>;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    if (threadIdx.y >= PipeGemmBiasRelu<ProblemShape>::num_warps) return;

    Input input(&prob->in[2][row * prob->mi][0], mblk * prob->d);
    Accum accum(prob->qs.q12[row]);
    Output output(prob->qs.q23[row]);

    input.reset();
    pipe_gemm_bias_relu<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >(&prob->w1[2][0][0], &prob->b1[0], input, accum, output, num_iters);
}

__device__ void mlp1_sm0(MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    if (threadIdx.y >= PipeGemmBias<ProblemShape>::num_warps) return;

    Input input(prob->qs.q23[row]);
    Accum accum;
    Output output(prob->qs.q34[row]);

    pipe_gemm_bias_relu<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >(&prob->w2[0][0], &prob->b2[0], input, accum, output, num_iters);
}

__device__ void mlp2_sm0(MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    using Output = MemoryWriter;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    if (threadIdx.y >= PipeGemmBiasLayerNorm<ProblemShape>::num_warps) return;

    Input input(prob->qs.q34[row]);
    Accum accum;
    Output output(&prob->out[row * prob->mo][0], mblk * prob->d);

    output.reset();
    pipe_gemm_bias<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >(&prob->w3[0][0], &prob->b3[0], input, accum, output, num_iters);
}



template<typename QT>
__device__ void consume_dummy(QT& q, size_t num_iters) {
    QueueReader<QT> r(q);

    r.reset();

    for (size_t i = 0; i < num_iters; i++) {
        r.read_acquire();
        r.read_release();
    }
}


__global__ void kernel(MgnNodeMlp * prob) {

    if (blockIdx.y < MgnNodeMlp::mo) {
        size_t pipe_col = blockIdx.x;
        size_t pipe_row = blockIdx.y;

        switch (pipe_col) {
            case 0: mlp0_sm0(prob, pipe_row); break;
            case 1: mlp0_sm1(prob, pipe_row); break;
            case 2: mlp0_sm2(prob, pipe_row); break;
            case 3: mlp1_sm0(prob, pipe_row); break;
            case 4: mlp2_sm0(prob, pipe_row); break;
            default: return;
        }
    }
    else {
        size_t ln_bid =
            blockIdx.y * gridDim.x + blockIdx.x - MgnNodeMlp::tot_mlp_blocks;

        size_t m_stride = MgnNodeMlp::tot_ln_blocks * MgnNodeMlp::ln_mblk;

        for (size_t m = ln_bid * MgnNodeMlp::ln_mblk; m < MgnNodeMlp::m; m += m_stride) {
            int mblk = min(MgnNodeMlp::ln_mblk, MgnNodeMlp::m - m);

            layer_norm<128>(
                &prob->ln_in[0][0],
                &prob->gamma[0],
                &prob->beta[0],
                &prob->ln_out[0][0],
                mblk);
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
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));

    MgnNodeMlp * prob;
    cudaErrCheck(cudaMalloc(&prob, sizeof(MgnNodeMlp) + 128));

    printf("prob: %p\n", prob);


    size_t tot_pipe_bytes = sizeof(MgnNodeMlp::Queues);

    printf("Total pipe bytes: %lu\n", tot_pipe_bytes);
    printf("Total pipe bytes: %lu KB\n", tot_pipe_bytes / 1024);

    printf("Init...\n");
    init_prob<<<1, 128>>>(prob);
    cudaErrCheck(cudaDeviceSynchronize());

    cudaStreamAttrValue attribute;
    auto& window = attribute.accessPolicyWindow;
    window.base_ptr = &prob->qs;
    window.num_bytes = sizeof(MgnNodeMlp::Queues);
    window.hitRatio = 1.0;
    window.hitProp = cudaAccessPropertyPersisting;
    window.missProp = cudaAccessPropertyStreaming;

    cudaStreamSetAttribute(
        cudaStreamDefault,
        cudaStreamAttributeAccessPolicyWindow,
        &attribute
    );


    const size_t tot_loop_iters = MgnNodeMlp::ni * MgnNodeMlp::mi / MgnNodeMlp::mblk;
    printf("Total loop iters: %lu\n", tot_loop_iters);

    printf("SMEM: %lu\n", max_smem);
    printf("# Warps: %lu\n", max_warps);

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            for (int ii = 0; ii < MgnNodeMlp::ni; ii++) {
                // dim3 grid1(5, MgnNodeMlp::mo);
                dim3 grid1(5, MgnNodeMlp::mo + MgnNodeMlp::ln_rows);
                dim3 block1(32, 4);
                kernel<<<grid1, block1, max_smem>>>(prob);

                // dim3 grid2(MgnNodeMlp::m / 512);
                // dim3 block2(32, 4);
                // device_layer_norm<MgnNodeMlp::d><<<grid2, block2>>>(
                //     &prob->out[0][0],
                //     &prob->gamma[0],
                //     &prob->beta[0],
                //     &prob->out[0][0],
                //     512);
            }
        }
    );

    printf("Total time: %f ms\n", time_ms);
    printf("Avg. loop iter time: %f ms\n", time_ms / tot_loop_iters);

    float flops_v1 =
        2.0f * MgnNodeMlp::m * (3 * MgnNodeMlp::d) * MgnNodeMlp::d +
        2.0f * MgnNodeMlp::m * MgnNodeMlp::d * MgnNodeMlp::d +
        2.0f * MgnNodeMlp::m * MgnNodeMlp::d * MgnNodeMlp::d;
    float gflops_v1 = MgnNodeMlp::ni * flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    return 0;
}
