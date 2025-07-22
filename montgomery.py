import torch
import time
import statistics
import field_ops
p = 2**31 - 1
R = 1 << 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('start')

N = 1024 << 3  # 8192

A = torch.randint(0, p, (N, N), dtype=torch.int64, device=device)
x = torch.randint(0, p, (N,), dtype=torch.int64, device=device)

def to_montgomery(x, R, p):
    return (x * R) % p

A_mont = to_montgomery(A, R, p).to(torch.uint32)
x_mont = to_montgomery(x, R, p).to(torch.uint32)
x_mont = x_mont.unsqueeze(1)  # shape (N,1)

A_mod = (A % p).to(torch.uint32)
x_mod = (x % p).to(torch.uint32)
x_mod = x_mod.unsqueeze(1)

mod_times = []
for _ in range(10):
    torch.cuda.synchronize()
    start = time.time()

    res_mod = field_ops.field_matmul_int32(A_mod, x_mod)

    torch.cuda.synchronize()
    mod_times.append(time.time() - start)

mont_times = []
for _ in range(10):
    torch.cuda.synchronize()
    start = time.time()

    res_mont = field_ops.montgomery_field_matmul_int32(A_mont, x_mont)

    torch.cuda.synchronize()
    mont_times.append(time.time() - start)


A_fp16 = A.float().half()
x_fp16 = x.float().half()

fp16_times = []
for _ in range(10):
    torch.cuda.synchronize()
    start = time.time()

    res_fp16 = field_ops.matvec_fp16_shared(A_fp16, x_fp16)#torch.matmul(A_fp16, x_fp16)

    torch.cuda.synchronize()
    fp16_times.append(time.time() - start)

mont_median = statistics.median(mont_times)
mod_median = statistics.median(mod_times)
fp16_median = statistics.median(fp16_times)

print(f"Montgomery multiplication median time: {mont_median:.6f} s")
print(f"Modulo multiplication median time:     {mod_median:.6f} s")
print(f"FP16 multiplication median time:       {fp16_median:.6f} s")
