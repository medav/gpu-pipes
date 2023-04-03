#include "blockgemm.cuh"
#include "utils.cuh"

#define NI 100000
#define NW 8
#define M 32
#define N 128
#define K 64

// using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;

using Block = cutlass::gemm::GemmShape<M, N, K>;

using BlockGemmOp = ThreadBlockGemmTensorOp<NW, Block>;

__global__ void kernel() {
    BlockGemmOp gemm_op;
    __shared__ half I[M * K];
    __shared__ half W[N * K];
    __shared__ half O[M * N];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    for (int i = 0; i < NI; i++) {
        gemm_op(
            (cutlass::half_t *)I,
            (cutlass::half_t *)W,
            (cutlass::half_t *)O,
            warp_id,
            lane_id);
    }

}
using WarpShape = cutlass::gemm::GemmShape<M, N / NW, K>;

using WarpGemmOp = cutlass::gemm::warp::GemmTensorOp<
    WarpShape,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
    cutlass::layout::RowMajor,
    true
    >;

__global__ void kernel2() {
    WarpGemmOp gemm_op;
    __shared__ half I[M * K];
    __shared__ half W[N * K / NW];
    __shared__ half O[M * N];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    for (int i = 0; i < NI; i++) {
        gemm_op(
            WarpGemmOp::TensorRefA((cutlass::half_t *)I, K),
            WarpGemmOp::TensorRefB((cutlass::half_t *)W, K),
            WarpGemmOp::TensorRefC((cutlass::half_t *)O, N / NW),
            WarpGemmOp::TensorRefC((cutlass::half_t *)O, N / NW),
            lane_id);
    }

}

int main() {
    dim3 grid(1, 1);
    dim3 block(32, NW);

    // half * O;
    // cudaMalloc(&O, M * N * sizeof(half));


    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<grid, block>>>();
        }
    );

    printf("gemm1 took %fms\n", time_ms);

    float flops_v1 = 2.0f * M * N * K * NI;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);


    // printf("Running...\n");
    // float time_ms2 = cuda_time_kernel_ms(
    //     [&]() {
    //         kernel2<<<grid, block>>>();
    //     }
    // );

    // printf("gemm2 took %fms\n", time_ms);

    // float flops_v2 = 2.0f * M * N * K * NI;
    // float gflops_v2 = flops_v2 / (time_ms2 * 1e6);
    // printf("+ GFLOPS: %f\n", gflops_v2);

    return 0;
}
