#include "warpgemm.cuh"
#include "utils.cuh"


#define CLD(N, D) ((N + D - 1) / D)

#define NI 1000000
#define NW 16
#define M 64
#define N 64
#define K 64

using WarpShape = cutlass::gemm::GemmShape<M, N, K>;

using WarpGemmOp = cutlass::gemm::warp::GemmTensorOp<
    WarpShape,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
    cutlass::layout::RowMajor,
    false
    >;


__global__ void smem_gemm() {
    WarpGemmOp gemm_op;
    __shared__ half I[M * K];
    __shared__ half W[N * K];
    __shared__ half O[M * N];


    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    for (int i = 0; i < NI; i++) {
        gemm_op(
            WarpGemmOp::TensorRefA((cutlass::half_t *)I, K),
            WarpGemmOp::TensorRefB((cutlass::half_t *)W, K),
            WarpGemmOp::TensorRefC((cutlass::half_t *)O, N),
            WarpGemmOp::TensorRefC((cutlass::half_t *)O, N),
            lane_id);
    }

}

int main(int argc, char const **args) {
    dim3 grid(1);
    dim3 block(32, NW);

    float time_ms = cuda_time_kernel_ms(
        [&]() {
            smem_gemm<<<grid, block>>>();
        }
    );

    printf("gemm took %fms\n", time_ms);

    float flops_v1 = 2.0f * M * N * K * NI * NW;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    cudaErrCheck(cudaDeviceReset());
    return 0;



    return 0;
}
