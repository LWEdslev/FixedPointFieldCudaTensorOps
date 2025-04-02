import time
import torch
import field_ops  # Import your compiled CUDA extension

k = 10

M, K, N = 1 << k, 1 << k, 1 << k

A = (torch.rand(M, K, dtype=torch.double, device="cuda") - 0.5) * 20 # uniform in [-10, 10]
B = (torch.rand(K, N, dtype=torch.double, device="cuda") - 0.5) * 20 # uniform in [-10, 10]
Af = A.float()
Bf = A.float()
torch.cuda.synchronize()
start_time = time.time()
expected = A @ B
torch.cuda.synchronize()
#print(expected)
time_taken_1 = time.time() - start_time
print(f"Operations took: {time_taken_1} seconds")

print('Expected result:\n', expected)

A = field_ops.encode_to_field_int32(A)
B = field_ops.encode_to_field_int32(B)
torch.cuda.synchronize()

# Perform and time first matrix multiplication
start_time = time.time()
C = field_ops.field_matmul_int32(A, B)
time_taken_1 = time.time() - start_time

C = field_ops.decode_from_field_int32(C)

# Print results
#print("Matrix A:\n", A.cpu().numpy())
#print("Matrix B:\n", B.cpu().numpy())
print("Result:\n", C)
print(f"Operations took: {time_taken_1} seconds")

e = C - expected
ae = torch.abs(e)
mae = torch.mean(ae).item()
print('MAE:', mae)
# Perform and time second matrix multiplication
#A = A.cpu()
#B = B.cpu()
#start_time = time.time()
#C = A @ B
#time_taken_2 = time.time() - start_time

# Print results
#print("Done in PyTorch:")
#print("Result C:\n", C.numpy())
#print(f"CPU matmul: {time_taken_2:.6f} seconds")

