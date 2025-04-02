import torch
import time
import field_ops

for k in range(8, 14):
    M, K, N = 2 << k, 2 << k, 2 << k
    A = (torch.rand(M, K, dtype=torch.double, device="cuda") - 0.5) * 20 # uniform in [-10, 10]
    B = (torch.rand(N, 1, dtype=torch.double, device="cuda") - 0.5) * 20 # uniform in [-10, 10]
    print('pytorch f64')
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    expected = A @ B
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"Operation took: {(end_time - start_time) * 1000:.4f} ms")

    print('pytorch f32')
    Af = A.float().cuda()
    Bf = B.float().cuda()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    expected = Af @ Bf
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"Operation took: {(end_time - start_time) * 1000:.4f} ms")

    print('f32', f"2^{k}")
    C = field_ops.benchmark_f32(A.float(), B.float())
    A = torch.randint(-10, 10, (M, K), dtype=torch.int32, device="cuda") # uniform in [-10, 10]
    B = torch.randint(-10, 10, (N, 1), dtype=torch.int32, device="cuda") # uniform in [-10, 10]
    print('i32', f"2^{k}")
    C = field_ops.benchmark_i32(A, B)
    print('field arithmetic')
    C = field_ops.field_matmul_int32(A, B)
    A = torch.randint(-10, 10, (M, K), dtype=torch.int64, device="cuda") # uniform in [-10, 10]
    B = torch.randint(-10, 10, (N, 1), dtype=torch.int64, device="cuda") # uniform in [-10, 10]
    C = field_ops.field_matmul_int64(A, B)

