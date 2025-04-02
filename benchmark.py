import torch
import time
import field_ops

M, N = 5120, 13824
A = (torch.rand(M, N, dtype=torch.double) - 0.5) * 20 # uniform in [-10, 10]
B = (torch.rand(N, 1, dtype=torch.double) - 0.5) * 20 # uniform in [-10, 10]
A = A.cuda()
B = B.cuda()
print('pytorch f64')
torch.cuda.synchronize()
start_time = time.perf_counter()
expected = A @ B
torch.cuda.synchronize()
end_time = time.perf_counter()
print(f"Operation took: {(end_time - start_time) * 1000:.4f} ms")
del A, B, expected
#torch.cuda.empty_cache()
for _ in range(10):
    print('pytorch f32')
    A = (torch.rand(M, N, dtype=torch.float32, device="cuda") - 0.5) * 20 # uniform in [-10, 10]
    B = (torch.rand(N, 1, dtype=torch.float32, device="cuda") - 0.5) * 20 # uniform in [-10, 10]
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    expected = A @ B
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"Operation took: {(end_time - start_time) * 1000:.4f} ms")
    del expected
    #torch.cuda.empty_cache()

for _ in range(10):
    print('pytorch f16')
    A = (torch.rand(M, N, dtype=torch.float16, device="cuda") - 0.5) * 20 # uniform in [-10, 10]
    B = (torch.rand(N, 1, dtype=torch.float16, device="cuda") - 0.5) * 20 # uniform in [-10, 10]
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    expected = A @ B
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"Operation took: {(end_time - start_time) * 1000:.4f} ms")
    del expected
    #torch.cuda.empty_cache()

for _ in range(10):
    print('f32')
    C = field_ops.benchmark_f32(A.float(), B.float())
    #print('i32')
    #C = field_ops.benchmark_i32(A, B)
    del C
    #torch.cuda.empty_cache()

for _ in range(10):
    A = torch.randint(-10, 10, (M, N), dtype=torch.int32, device="cuda") # uniform in [-10, 10]
    B = torch.randint(-10, 10, (N, 1), dtype=torch.int32, device="cuda") # uniform in [-10, 10]
    print('field arithmetic')
    C = field_ops.field_matmul_int32(A, B)
    #torch.cuda.empty_cache()
    del A, B, C
    #torch.cuda.empty_cache()

