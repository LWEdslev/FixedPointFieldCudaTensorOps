import time
import torch
import field_ops  # Import your compiled CUDA extension

M, K, N = 4096, 4096, 4096  # Example sizes

A = torch.randn(M, K, dtype=torch.double, device="cuda")
B = torch.randn(K, N, dtype=torch.double, device="cuda")

print('Expected result:\n', A @ B)

A = field_ops.encode_to_field_int64(A)
B = field_ops.encode_to_field_int64(B)


# Perform and time first matrix multiplication
start_time = time.time()
C = field_ops.field_matmul_int64(A, B)
time_taken_1 = time.time() - start_time

C = field_ops.decode_from_field_int64(C)

# Print results
#print("Matrix A:\n", A.cpu().numpy())
#print("Matrix B:\n", B.cpu().numpy())
print("Result:\n", C)
print(f"GPU matmul: {time_taken_1} seconds")

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

