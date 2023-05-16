#include "mgn_node_pipe.cuh"
#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"

#include "utils.cuh"

using ProblemShape = cutlass::gemm::GemmShape<MgnNodeMlp::mblk, 128, 128>;

const size_t max_smem = std::max({
    sizeof(typename PipeGemm<ProblemShape>::SmemBuffers),
    sizeof(typename PipeGemmBias<ProblemShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<ProblemShape>::SmemBuffers)
});

const size_t max_warps = std::max({
    PipeGemm<ProblemShape>::num_warps,
    PipeGemmBias<ProblemShape>::num_warps,
    PipeGemmBiasRelu<ProblemShape>::num_warps
});

__device__ void mlp0_sm0(MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(&prob->in[row * prob->mi][0], mblk * prob->d, MgnNodeMlp::d * 3);
    Accum accum;
    Output output(prob->qs.q01[row]);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        input.reset();
        pipe_gemm<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >({&prob->w1[0][0], MgnNodeMlp::d * 3}, input, accum, output, num_iters);
    }
}

__device__ void mlp0_sm1(MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = QueueReader<MgnNodeMlp::Queue>;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(&prob->in[row * prob->mi][MgnNodeMlp::d], mblk * prob->d, MgnNodeMlp::d * 3);
    Accum accum(prob->qs.q01[row]);
    Output output(prob->qs.q12[row]);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        input.reset();
        pipe_gemm<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >({&prob->w1[0][MgnNodeMlp::d], MgnNodeMlp::d * 3}, input, accum, output, num_iters);
    }
}

__device__ void mlp0_sm2(MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = QueueReader<MgnNodeMlp::Queue>;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(&prob->in[row * prob->mi][MgnNodeMlp::d * 2], mblk * prob->d, MgnNodeMlp::d * 3);
    Accum accum(prob->qs.q12[row]);
    Output output(prob->qs.q23[row]);


    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        input.reset();
        pipe_gemm_bias_relu<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >({&prob->w1[0][MgnNodeMlp::d * 2], MgnNodeMlp::d * 3}, {&prob->b1[0], 0}, input, accum, output, num_iters);
    }
}

__device__ void mlp1_sm0(MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(prob->qs.q23[row]);
    Accum accum;
    Output output(prob->qs.q34[row]);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        pipe_gemm_bias_relu<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >({&prob->w2[0][0], MgnNodeMlp::d}, {&prob->b2[0], 0}, input, accum, output, num_iters);
    }
}

__device__ void mlp2_sm0(MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::LayerNormQueue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(prob->qs.q34[row]);
    Accum accum;
    Output output(prob->qs.lnq[row]);


    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        output.reset();
        pipe_gemm_bias<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >({&prob->w3[0][0], MgnNodeMlp::d}, {&prob->b3[0], 0}, input, accum, output, num_iters);
    }
}

__device__ void ln_sm(MgnNodeMlp *prob, size_t row, int seq_off, int num_lns) {
    using Input = SplitQueueReader<MgnNodeMlp::LayerNormQueue>;
    using Output = MemoryWriter;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk / num_lns;

    Input input(prob->qs.lnq[row], seq_off, num_lns);

    Output output(
        &prob->out[mblk * seq_off][0],
        mblk * num_lns * MgnNodeMlp::d,
        MgnNodeMlp::d);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        output.reset();
        pipe_layer_norm<LayerNormShape<mblk, 128>, Input, Output>(
            {&prob->gamma[0], 0}, {&prob->beta[0], 0}, input, output, num_iters);
    }

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
    void * smem = nullptr;
    size_t pipe_col = blockIdx.x;
    size_t pipe_row = blockIdx.y;

    switch (pipe_col) {
        case 0: mlp0_sm0(prob, pipe_row); break;
        case 1: mlp0_sm1(prob, pipe_row); break;
        case 2: mlp0_sm2(prob, pipe_row); break;
        case 3: mlp1_sm0(prob, pipe_row); break;
        case 4: mlp2_sm0(prob, pipe_row); break;
        default:
            int ln_col = pipe_col - MgnNodeMlp::n_mlp_cols;

            if (ln_col < MgnNodeMlp::n_ln_cols) {
                ln_sm(prob, pipe_row, ln_col, MgnNodeMlp::n_ln_cols);
            }

            return;
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

    dim3 grid(MgnNodeMlp::n_cols, MgnNodeMlp::mo);
    dim3 block(32, 4);

    const size_t tot_loop_iters = MgnNodeMlp::ni * MgnNodeMlp::mi / MgnNodeMlp::mblk;
    printf("Total loop iters: %lu\n", tot_loop_iters);

    printf("SMEM: %lu\n", max_smem);
    printf("# Warps: %lu\n", max_warps);

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            for (size_t i = 0; i < MgnNodeMlp::no; i++) {
                kernel<<<grid, block, max_smem>>>(prob);
            }
        }
    );

    printf("Total time: %f ms\n", time_ms);
    printf("Avg. loop iter time: %f ms\n", time_ms / tot_loop_iters);

    float flops_v1 =
        2.0f * MgnNodeMlp::m * (3 * MgnNodeMlp::d) * MgnNodeMlp::d +
        2.0f * MgnNodeMlp::m * MgnNodeMlp::d * MgnNodeMlp::d +
        2.0f * MgnNodeMlp::m * MgnNodeMlp::d * MgnNodeMlp::d;
    float gflops_v1 = MgnNodeMlp::no * MgnNodeMlp::ni * flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    return 0;
}
