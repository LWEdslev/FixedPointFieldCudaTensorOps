#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 16

__global__ void field_matmul_i64_kernel(const int64_t* A, const int64_t* B, int64_t* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // these should be moved out as constants at some point
    int64_t n = 8;
    int64_t p = (((int64_t)2) << 31) - 1;

    if (row < M && col < N) {
        int64_t sum = 0;
        for (int k = 0; k < K; ++k) {
            int64_t z = A[row * K + k];
            int64_t z_p = B[k * N + col];

            if (z >= p/2) z -= p;
            if (z_p >= p/2) z_p -= p;
            int64_t t = z * z_p;
            t >>= n;
            t %= p;
            if (t < 0) t += p;

            sum += t;
        }
        C[row * N + col] = (sum % p);
    }
}


// CUDA kernel for matrix multiplication of int64 tensors
__global__ void matmul_int64_kernel(const int64_t* A, const int64_t* B, int64_t* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int64_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_int32_kernel(const int32_t* A, const int32_t* B, int32_t* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void encode_to_field_int64_kernel(const double* input, int64_t* output, int size) {
    int64_t n = 8;
    int64_t p = (((int64_t)2) << 31) - 1;
    double p_f = (double)p;
    double two_pow_n = (double)pow(2,n);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double z = input[idx];
        z = fmod(z * two_pow_n, p_f);
        if (z < 0) z += p;
        int64_t z_i = (int64_t)z;
        output[idx] = z_i;
    }
}

__global__ void decode_from_field_int64_kernel(const int64_t* input, double* output, int size) {
    int64_t n = 8;
    int64_t p = (((int64_t)2) << 31) - 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int64_t r = input[idx];
        double r_d = (double)r;
        if (r_d >= p/2) r_d -= p;
        output[idx] = r_d / pow(2, n);
    }
}

torch::Tensor field_matmul_int64(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kInt64, "A must be int64");
    TORCH_CHECK(B.scalar_type() == torch::kInt64, "B must be int64");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kInt64).device(A.device()));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    field_matmul_i64_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<int64_t>(),
        B.data_ptr<int64_t>(),
        C.data_ptr<int64_t>(),
        M, N, K
    );

    return C;
}

// Wrapper function for PyTorch 
torch::Tensor matmul_int64(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kInt64, "A must be int64");
    TORCH_CHECK(B.scalar_type() == torch::kInt64, "B must be int64");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kInt64).device(A.device()));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_int64_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<int64_t>(),
        B.data_ptr<int64_t>(),
        C.data_ptr<int64_t>(),
        M, N, K
    );

    return C;
}

torch::Tensor matmul_int32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kInt32, "A must be int32");
    TORCH_CHECK(B.scalar_type() == torch::kInt32, "B must be int32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_int32_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<int32_t>(),
        B.data_ptr<int32_t>(),
        C.data_ptr<int32_t>(),
        M, N, K
    );

    return C;
}

torch::Tensor encode_to_field_int64(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kDouble, "A must be of type double");

    auto output = torch::empty_like(A, torch::TensorOptions().dtype(torch::kInt64));
    int64_t size = A.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    encode_to_field_int64_kernel<<<blocks, threads>>>(
        A.data_ptr<double>(),
        output.data_ptr<int64_t>(),
        size
    );

    cudaDeviceSynchronize();

    return output;
}

torch::Tensor decode_from_field_int64(torch::Tensor A) { 
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kInt64, "A must be of type double");

    auto output = torch::empty_like(A, torch::TensorOptions().dtype(torch::kDouble));
    int64_t size = A.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    decode_from_field_int64_kernel<<<blocks, threads>>>(
        A.data_ptr<int64_t>(),
        output.data_ptr<double>(),
        size
    );

    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_int64", &matmul_int64, "Matrix multiplication (int64) on CUDA");
    m.def("matmul_int32", &matmul_int32, "Matrix multiplication (int32) on CUDA");
    m.def("field_matmul_int64", &field_matmul_int64, "Fixed point field Matrix multiplication (int64) on CUDA");
    m.def("encode_to_field_int64", &encode_to_field_int64, "Encode (double) to field (int64) on CUDA");
    m.def("decode_from_field_int64", &decode_from_field_int64, "Decode from field (int64) to double on CUDA");
}