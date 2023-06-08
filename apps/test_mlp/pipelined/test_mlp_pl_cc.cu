
#include <cuda_fp16.h>
#include "test_mlp_pl.cuh"
#include "utils.cuh"

inline typename TestMlp::Queues * global_queue_space() {
    static typename TestMlp::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, TestMlp::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, TestMlp::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, TestMlp::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)testmlp_device, max_smem);
    configured = true;
}


int main(int argc, char * argv[]) {
    const size_t NI = argc > 1 ? std::atoi(argv[1]) : 1000;
    printf("NI: %zu\n", NI);
    const size_t m = 1280 * 1024;

    half * x_dev = nullptr;
    half * w1_dev = nullptr;
    half * b1_dev = nullptr;
    half * w2_dev = nullptr;
    half * b2_dev = nullptr;
    half * w3_dev = nullptr;
    half * b3_dev = nullptr;
    half * out_dev = nullptr;

    cudaErrCheck(cudaMalloc(&x_dev, m * TestMlp::d * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w1_dev, TestMlp::d * TestMlp::d * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b1_dev, TestMlp::d * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w2_dev, TestMlp::d * TestMlp::d * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b2_dev, TestMlp::d * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w3_dev, TestMlp::d * TestMlp::d * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b3_dev, TestMlp::d * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&out_dev, m * TestMlp::d * sizeof(*x_dev)));


    dim3 grid(TestMlp::n_cols, TestMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            testmlp_device<<<grid, block, max_smem>>>(
                m,
                x_dev,
                w1_dev, b1_dev,
                w2_dev, b2_dev,
                w3_dev, b3_dev,
                out_dev,
                global_queue_space()
            );
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = m * (
        TestMlp::d * TestMlp::d +
        TestMlp::d * TestMlp::d +
        TestMlp::d * TestMlp::d) * 2.0f;

    printf("GFLOPS: %f\n", NI * flops / (time_ms * 1e-3f) / 1e9f);
}
