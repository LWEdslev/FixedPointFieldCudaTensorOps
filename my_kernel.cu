#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 8
#define TILE_SIZE 8
/*
constexpr int64_t n64 = 8;
constexpr int64_t p64 = (((int64_t)1) << 31) - 1;

constexpr int32_t n32 = 1;
constexpr int32_t p32 = (((int32_t)1) << 13) - 1;


__global__ void field_matmul_i64_kernel(const int64_t* A, const int64_t* B, int64_t* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // these should be moved out as constants at some point

    if (row < M && col < N) {
        int64_t sum = 0;
        for (int k = 0; k < K; ++k) {
            int64_t z = A[row * K + k];
            int64_t z_p = B[k * N + col];

            if (z >= p64/2) z -= p64;
            if (z_p >= p64/2) z_p -= p64;
            int64_t t = z * z_p;
            t >>= n64;
            t %= p64;
            if (t < 0) t += p64;

            sum += t;
        }
        C[row * N + col] = (sum % p64);
    }
}

__global__ void encode_to_field_int64_kernel(const double* input, int64_t* output, int size) {
    double p_f = (double)p64;
    double two_pow_n = (double)pow(2,n64);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double z = input[idx];
        z = fmod(z * two_pow_n, p_f);
        if (z < 0) z += p64;
        int64_t z_i = (int64_t)z;
        output[idx] = z_i;
    }
}

__global__ void decode_from_field_int64_kernel(const int64_t* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int64_t r = input[idx];
        double r_d = (double)r;
        if (r_d >= p64/2) r_d -= p64;
        output[idx] = r_d / pow(2, n64);
    }
}

__global__ void encode_to_field_int32_kernel(const double* input, int32_t* output, int size) {
    double p_f = (double)p32;
    double two_pow_n = (double)pow(2,n32);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double z = input[idx];
        z = fmod(z * two_pow_n, p_f);
        if (z < 0) z += p32;
        int32_t z_i = (int32_t)z;
        output[idx] = z_i;
    }
}

__global__ void decode_from_field_int32_kernel(const int32_t* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int32_t r = input[idx];
        double r_d = (double)r;
        if (r_d >= p32/2) r_d -= p32;
        output[idx] = r_d / pow(2, n32);
    }
}

__global__ void field_matmul_i32_kernel(const int32_t* A, const int32_t* B, int32_t* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
            int32_t z = A[row * K + k];
            int32_t z_p = B[k * N + col];

            if (z >= p32/2) z -= p32;
            if (z_p >= p32/2) z_p -= p32;
            int64_t z64 = (int64_t)z;
            int64_t z_p64 = (int64_t)z_p;
            int64_t t64 = z64 * z_p64;
            t64 >>= n32;
            
            if (t64 < 0) t64 += p32;
            int32_t t = (int32_t)t64;

            sum += t;
        }
        C[row * N + col] = sum % p32;
    }
}

__global__ void field_matmul_i32_kernel_shared(
    const int32_t* __restrict__ A,
    const int32_t* __restrict__ B,
    int32_t* C,
    int M, int N, int K)
{
    __shared__ int32_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int32_t sum = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A and B tiles into shared memory
        int tiledRow = row;
        int tiledColA = t * TILE_SIZE + threadIdx.x;
        int tiledRowB = t * TILE_SIZE + threadIdx.y;
        int tiledCol = col;

        if (tiledRow < M && tiledColA < K)
            tile_A[threadIdx.y][threadIdx.x] = A[tiledRow * K + tiledColA];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        if (tiledRowB < K && tiledCol < N)
            tile_B[threadIdx.y][threadIdx.x] = B[tiledRowB * N + tiledCol];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Multiply tiles
        for (int k = 0; k < TILE_SIZE; ++k) {
            int32_t z = tile_A[threadIdx.y][k];
            int32_t z_p = tile_B[k][threadIdx.x];

            if (z >= p32 / 2) z -= p32;
            if (z_p >= p32 / 2) z_p -= p32;

            int64_t z64 = (int64_t)z;
            int64_t z_p64 = (int64_t)z_p;
            int64_t t64 = z64 * z_p64;
            t64 >>= n32;
            if (t64 < 0) t64 += p32;

            int32_t t = (int32_t)t64;
            sum += t;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum % p32;
    }
}



// This is simply for measuring how fast f32 is against i32 when doing the same operations
__global__ void benchmark_f32_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void benchmark_f32_kernel_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for A and B tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Row and column index of C element this thread computes
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0;

    // Loop over tiles of A and B
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load elements into shared memory, with bounds check
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply tileA and tileB together
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Store the result
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}


__global__ void benchmark_i32_kernel(const int32_t* A, const int32_t* B, int32_t* C, int M, int N, int K) {
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

torch::Tensor benchmark_i32(torch::Tensor A, torch::Tensor B) {
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    benchmark_i32_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<int32_t>(),
        B.data_ptr<int32_t>(),
        C.data_ptr<int32_t>(),
        M, N, K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Operation took %f ms\n", milliseconds);

    return C;
}

torch::Tensor benchmark_f32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "A must be int32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat, "B must be int32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    benchmark_f32_kernel_tiled<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Operation took %f ms\n", milliseconds);

    return C;
}
*/
constexpr uint32_t P     = 0x7FFFFFFF;       
constexpr uint64_t R     = 1ULL << 32;
constexpr uint32_t P_inv = 2147483649;

__global__ void matvec_fp16_shared_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ __half As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __half Bs[BLOCK_SIZE];
    if (row < M && col == 0) {
        __half acc = __float2half(0.0f);

        for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
            int k = tile * BLOCK_SIZE + threadIdx.x;
            if (row < M && k < K) {
                As[threadIdx.y][threadIdx.x] = A[row * K + k];
            } else {
                As[threadIdx.y][threadIdx.x] = __float2half(0.0f);
            }

            if (k < K && threadIdx.y == 0) {
                Bs[threadIdx.x] = B[k];
            }
            __syncthreads();

            if (k < K) {
                acc = __hfma(As[threadIdx.y][threadIdx.x], Bs[threadIdx.x], acc);
            }

            __syncthreads();
        }

        C[row] = acc;
    }
}



__device__ __forceinline__ uint32_t montgomery_reduce(uint64_t T) {
    uint32_t m = (uint32_t)(T * P_inv);  
    uint64_t t = (T + (uint64_t)m * P) >> 32;
    if (t >= P) t -= P;
    return (uint32_t)t;
}

__global__ void montgomery_matvec_kernel(
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ B,
    uint32_t* C,
    int M, int N, int K
) {
    __shared__ uint32_t sA[TILE_SIZE][TILE_SIZE];   
    __shared__ uint32_t sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    uint64_t acc = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_k = t * TILE_SIZE + threadIdx.x;
        if (row < M && tiled_k < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiled_k];
        else
            sA[threadIdx.y][threadIdx.x] = 0;
        if (col < N && tiled_k < K)
            sB[threadIdx.y][threadIdx.x] = B[tiled_k * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += (uint64_t)sA[threadIdx.y][k] * (uint64_t)sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = montgomery_reduce(acc);
    }
}

__global__ void matvec_mod_kernel(
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ B,
    uint32_t* C,
    int M, int N, int K
) {
    __shared__ uint32_t sA[TILE_SIZE][TILE_SIZE];
    __shared__ uint32_t sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    uint64_t acc = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_k = t * TILE_SIZE + threadIdx.x;

        if (row < M && tiled_k < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiled_k];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        if (col < N && tiled_k < K)
            sB[threadIdx.y][threadIdx.x] = B[tiled_k * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += (uint64_t)sA[threadIdx.y][k] * (uint64_t)sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = (uint32_t)(acc % P);
    }
}

/*
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    field_matmul_i64_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<int64_t>(),
        B.data_ptr<int64_t>(),
        C.data_ptr<int64_t>(),
        M, N, K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Operation took %f ms\n", milliseconds);

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

torch::Tensor field_matmul_int32(torch::Tensor A, torch::Tensor B) {
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    montgomery_matvec_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<int32_t>(),
        B.data_ptr<int32_t>(),
        C.data_ptr<int32_t>(),
        M, N, K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Operation took %f ms\n", milliseconds);

    return C;
}

torch::Tensor encode_to_field_int32(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kDouble, "A must be of type double");

    auto output = torch::empty_like(A, torch::TensorOptions().dtype(torch::kInt32));
    int64_t size = A.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    encode_to_field_int32_kernel<<<blocks, threads>>>(
        A.data_ptr<double>(),
        output.data_ptr<int32_t>(),
        size
    );

    cudaDeviceSynchronize();

    return output;
}

torch::Tensor decode_from_field_int32(torch::Tensor A) { 
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kInt32, "A must be of type double");

    auto output = torch::empty_like(A, torch::TensorOptions().dtype(torch::kDouble));
    int64_t size = A.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    decode_from_field_int32_kernel<<<blocks, threads>>>(
        A.data_ptr<int32_t>(),
        output.data_ptr<double>(),
        size
    );

    cudaDeviceSynchronize();

    return output;
}
*/
torch::Tensor montgomery_field_matmul_int32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kUInt32, "A must be uint32");
    TORCH_CHECK(B.scalar_type() == torch::kUInt32, "B must be uint32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kUInt32).device(A.device()));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    montgomery_matvec_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<uint32_t>(),
        B.data_ptr<uint32_t>(),
        C.data_ptr<uint32_t>(),
        M, N, K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Montgomery matmul operation took %f ms\n", milliseconds);

    return C;
}

torch::Tensor field_matmul_int32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kUInt32, "A must be uint32");
    TORCH_CHECK(B.scalar_type() == torch::kUInt32, "B must be uint32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kUInt32).device(A.device()));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matvec_mod_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<uint32_t>(),
        B.data_ptr<uint32_t>(),
        C.data_ptr<uint32_t>(),
        M, N, K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Modulos matmul operation took %f ms\n", milliseconds);

    return C;
}

torch::Tensor matvec_fp16_shared(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be float16");
    TORCH_CHECK(B.dim() == 1, "B must be 1D vector");

    int M = A.size(0);
    int K = A.size(1);

    auto C = torch::empty({M}, torch::dtype(torch::kHalf).device(A.device()));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(1, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matvec_fp16_shared_kernel<<<gridDim, blockDim>>>(
        reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, 1, K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("FP16 shared memory matvec took %f ms\n", milliseconds);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //m.def("field_matmul_int64", &field_matmul_int64, "Fixed point field Matrix multiplication (int64) on CUDA");
    //m.def("encode_to_field_int64", &encode_to_field_int64, "Encode (double) to field (int64) on CUDA");
    //m.def("decode_from_field_int64", &decode_from_field_int64, "Decode from field (int64) to double on CUDA");
    //m.def("field_matmul_int32", &field_matmul_int32, "Fixed point field Matrix multiplication (int32) on CUDA");
    //m.def("encode_to_field_int32", &encode_to_field_int32, "Encode (double) to field (int32) on CUDA");
    //m.def("decode_from_field_int32", &decode_from_field_int32, "Decode from field (int32) to double on CUDA");
    //m.def("benchmark_i32", &benchmark_i32, "benchmark i32 matrix multiplication (not field)");
    //m.def("benchmark_f32", &benchmark_f32, "benchmark f32 matrix multiplication (not field)");
    m.def("montgomery_field_matmul_int32", &montgomery_field_matmul_int32, "i32 prime field multiplication using montomgery multiplication");
    m.def("field_matmul_int32", &field_matmul_int32, "i32 prime field multiplication using modulos");
    m.def("matvec_fp16_shared", &matvec_fp16_shared, "regular mat vec mult");
}