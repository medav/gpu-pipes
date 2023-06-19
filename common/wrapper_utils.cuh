#pragma once

#define ALLOC_TENSOR_2D(name, m, n) \
    half * name = nullptr; \
    cudaErrCheck(cudaMalloc(&name, m * n * sizeof(*name)));

#define ALLOC_LINEAR_WEIGHTS(prefix, n_in, n_out) \
    half * prefix ## _w = nullptr; \
    cudaErrCheck(cudaMalloc(&prefix ## _w, n_in * n_out * sizeof(*prefix ## _w))); \
    half * prefix ## _b = nullptr; \
    cudaErrCheck(cudaMalloc(&prefix ## _b, n_out * sizeof(*prefix ## _b)));

#define ALLOC_LN_WEIGHTS(prefix, n) \
    half * prefix ## _ga = nullptr; \
    cudaErrCheck(cudaMalloc(&prefix ## _ga, n * sizeof(*prefix ## _ga))); \
    half * prefix ## _be = nullptr; \
    cudaErrCheck(cudaMalloc(&prefix ## _be, n * sizeof(*prefix ## _be)));
