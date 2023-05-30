#pragma once


struct Tensor {
    const size_t R;
    const size_t C;

    float * host_ptr;
    half * dev_ptr;

    Tensor(size_t R, size_t C) : R(R), C(C) {
        host_ptr = new float[R * C];
        cudaMalloc(&dev_ptr, R * C * sizeof(half));
    }

    ~Tensor() {
        delete[] host_ptr;
        cudaFree(dev_ptr);
    }

    void rand_fill() {
        for (size_t i = 0; i < R * C; i++) {
            host_ptr[i] = 2.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
            // host_ptr[i] = (float)rand() / RAND_MAX;
        }
    }

    void fill(float val) {
        for (size_t i = 0; i < R * C; i++) {
            host_ptr[i] = val;
        }
    }

    void to_dev() {
        float * tmp_dev;
        cudaMalloc(&tmp_dev, R * C * sizeof(float));
        cudaMemcpy(tmp_dev, host_ptr, R * C * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half<<<CLD(R * C, 128), 128>>>(dev_ptr, tmp_dev, R * C);
        cudaFree(tmp_dev);
    }

    void to_host() {
        float * tmp_dev;
        cudaMalloc(&tmp_dev, R * C * sizeof(float));
        half_to_float<<<CLD(R * C, 128), 128>>>(tmp_dev, dev_ptr, R * C);
        cudaMemcpy(host_ptr, tmp_dev, R * C * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tmp_dev);
    }

    void print() {
        for (size_t r = 0; r < R; r++) {
            for (size_t c = 0; c < C; c++) {
                printf("%.2f ", host_ptr[r * C + c]);
            }
            printf("\n");
        }
    }
};



template<int kM, int kN, int kK>
__global__ void ref_gemm(half * x, half * w, half * out) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= kM || n >= kN) return;

    float sum = 0.0f;
    for (int k = 0; k < kK; k++) {
        sum += (float)x[m * kK + k] * (float)w[k * kN + n];
    }
    out[m * kN + n] = (half)sum;
}

template<int kM, int kN, int kK>
__global__ void ref_gemm_bias(half * x, half * w, half * b, half * out) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= kM || n >= kN) return;

    float sum = 0.0f;
    for (int k = 0; k < kK; k++) {
        sum += (float)x[m * kK + k] * (float)w[k * kN + n];
    }
    sum += (float)b[n];
    out[m * kN + n] = (half)sum;
}

template<int kM, int kN, int kK>
__global__ void ref_gemm_bias_relu(half * x, half * w, half * b, half * out) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= kM || n >= kN) return;

    float sum = 0.0f;
    for (int k = 0; k < kK; k++) {
        sum += (float)x[m * kK + k] * (float)w[k * kN + n];
    }
    sum += (float)b[n];
    out[m * kN + n] = sum > 0.0f ? (half)sum : (half)0.0f;
}

float rel_err(float a, float b, float eps = 1e-6f) {
    return fabs(a - b) / ((a + b) / 2.0f + eps);
}

bool isclose(float a, float b, float rtol = 0.05) {
    return rel_err(a, b) < rtol;
}


void compare(Tensor& ref, Tensor& act) {
    for (size_t r = 0; r < ref.R; r++) {
        for (size_t c = 0; c < ref.C; c++) {
            float ref_val = ref.host_ptr[r * ref.C + c];
            float act_val = act.host_ptr[r * act.C + c];
            if (!isclose(ref_val, act_val)) {
                printf("Mismatch at %zu, %zu: %f != %f\n", r, c, ref_val, act_val);
            }
        }
    }
}

float l2(Tensor& a, Tensor& b) {
    float sum_sq = 0.0f;
    for (size_t r = 0; r < a.R; r++) {
        for (size_t c = 0; c < a.C; c++) {
            float ax = a.host_ptr[r * a.C + c];
            float bx = b.host_ptr[r * a.C + c];

            sum_sq += (ax - bx) * (ax - bx);
        }
    }

    return sqrt(sum_sq);
}
