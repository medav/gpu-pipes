#include "mgn_node_pipe.cuh"
#include "pipes.cuh"
#include "pipegemm.cuh"
#include "pipebiasaddrelu.cuh"
#include "utils.cuh"

const size_t max_smem = sizeof(SmemBuffers<MgnNodeMlp::mblk, 128, 128>);

__device__ void mlp0_gemm0(void * smem, MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(&prob->in[0][row * prob->mi][0], mblk * prob->d);
    Accum accum;
    Output output(prob->qs.qs[row][0]);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        input.reset();
        gemmpipe<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >(smem, &prob->w1[0][0][0], input, accum, output, num_iters);
    }
}

__device__ void mlp0_gemm1(void * smem, MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = QueueReader<MgnNodeMlp::Queue>;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(&prob->in[1][row * prob->mi][0], mblk * prob->d);
    Accum accum(prob->qs.qs[row][0]);
    Output output(prob->qs.qs[row][1]);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        input.reset();
        gemmpipe<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >(smem, &prob->w1[1][0][0], input, accum, output, num_iters);
    }
}

__device__ void mlp0_gemm2(void * smem, MgnNodeMlp *prob, size_t row) {
    using Input = MemoryReader;
    using Accum = QueueReader<MgnNodeMlp::Queue>;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(&prob->in[2][row * prob->mi][0], mblk * prob->d);
    Accum accum(prob->qs.qs[row][1]);
    Output output(prob->qs.qs[row][2]);


    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        input.reset();
        gemmpipe<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >(smem, &prob->w1[2][0][0], input, accum, output, num_iters);
    }
}

__device__ void mlp0_bias_relu(void * smem, MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(prob->qs.qs[row][2]);
    Accum accum;
    Output output(prob->qs.qs[row][3]);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        biasaddrelupipe<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >(smem, &prob->b1[0], input, accum, output, num_iters);
    }
}

__device__ void mlp1_gemm0(void * smem, MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(prob->qs.qs[row][3]);
    Accum accum;
    Output output(prob->qs.qs[row][4]);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        gemmpipe<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >(smem, &prob->w2[0][0], input, accum, output, num_iters);
    }
}

__device__ void mlp1_bias_relu(void * smem, MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(prob->qs.qs[row][4]);
    Accum accum;
    Output output(prob->qs.qs[row][5]);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        biasaddrelupipe<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >(smem, &prob->b2[0], input, accum, output, num_iters);
    }
}

__device__ void mlp2_gemm0(void * smem, MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    using Output = QueueWriter<MgnNodeMlp::Queue>;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(prob->qs.qs[row][5]);
    Accum accum;
    Output output(prob->qs.qs[row][6]);


    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        gemmpipe<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >(smem, &prob->w3[0][0], input, accum, output, num_iters);
    }
}

__device__ void mlp2_bias_relu(void * smem, MgnNodeMlp *prob, size_t row) {
    using Input = QueueReader<MgnNodeMlp::Queue>;
    using Accum = NullReader;
    // using Output = QueueWriter<MgnNodeMlp::Queue>;
    using Output = MemoryWriter;

    const size_t mblk = MgnNodeMlp::mblk;
    const size_t num_iters = prob->mi / MgnNodeMlp::mblk;

    Input input(prob->qs.qs[row][6]);
    Accum accum;
    // Output output(prob->qs.qs[row][7]);
    Output output(&prob->out[row * prob->mo][0], mblk * prob->d);

    for (size_t i = 0; i < MgnNodeMlp::ni; i++) {
        output.reset();
        biasaddrelupipe<
            cutlass::gemm::GemmShape<mblk, 128, 128>,
            Input,
            Accum,
            Output
        >(smem, &prob->b2[0], input, accum, output, num_iters);
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
        case 0: mlp0_gemm0(smem, prob, pipe_row); break;
        case 1: mlp0_gemm1(smem, prob, pipe_row); break;
        // case 2: consume_dummy(prob->qs.qs[pipe_row][1], prob->mi / MgnNodeMlp::mblk); break;
        case 2: mlp0_gemm2(smem, prob, pipe_row); break;
        case 3: mlp0_bias_relu(smem, prob, pipe_row); break;
        // case 4: consume_dummy(prob->qs.qs[pipe_row][3], prob->mi / MgnNodeMlp::mblk); break;
        case 4: mlp1_gemm0(smem, prob, pipe_row); break;
        case 5: mlp1_bias_relu(smem, prob, pipe_row); break;
        case 6: mlp2_gemm0(smem, prob, pipe_row); break;
        case 7: mlp2_bias_relu(smem, prob, pipe_row); break;
        // case 8: consume_dummy(prob->qs.qs[pipe_row][7], prob->mi / MgnNodeMlp::mblk); break;
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
    // prob = (MgnNodeMlp*)(((size_t)prob + 0x3F) & ~0x3F);

    // Print address of prob
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

    dim3 grid(9, MgnNodeMlp::mo);
    // dim3 grid(5, 1);
    dim3 block(32, num_warps);

    const size_t tot_loop_iters = MgnNodeMlp::ni * MgnNodeMlp::mi / MgnNodeMlp::mblk;
    printf("Total loop iters: %lu\n", tot_loop_iters);

    printf("SMEM: %lu\n", max_smem);
    printf("# Warps: %lu\n", num_warps);

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<grid, block, max_smem>>>(prob);
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
