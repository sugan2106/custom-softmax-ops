import csv
import numpy as np
import torch

ROWS = 1024
COLS = 1024
WARMUP = 30
REPEATS = 100

assert torch.cuda.is_available(), "CUDA-enabled PyTorch is required."

x_np = np.fromfile("input.bin", dtype=np.float32)
custom_np = np.fromfile("custom_output.bin", dtype=np.float32)

assert x_np.size == ROWS * COLS, "Input shape mismatch."
assert custom_np.size == ROWS * COLS, "Output shape mismatch."

x_np = x_np.reshape(ROWS, COLS)
custom_np = custom_np.reshape(ROWS, COLS)

x = torch.from_numpy(x_np.copy()).cuda()
custom = torch.from_numpy(custom_np.copy()).cuda()

with torch.no_grad():
    reference = torch.softmax(x, dim=-1)

    difference = (custom - reference).abs()
    max_abs_error = difference.max().item()

    max_rel_error = (
        difference / reference.abs().clamp_min(1e-7)
    ).max().item()

    row_sum_error = (
        custom.sum(dim=-1) - 1.0
    ).abs().max().item()

    correct = torch.allclose(
        custom, reference, rtol=1e-4, atol=1e-5
    )

    for _ in range(WARMUP):
        torch.softmax(x, dim=-1)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(REPEATS):
        torch.softmax(x, dim=-1)

    end.record()
    end.synchronize()

    pytorch_ms = start.elapsed_time(end) / REPEATS

with open("custom_latency.txt") as f:
    custom_ms = float(f.read().strip())

speedup = pytorch_ms / custom_ms

# Approximate bandwidth: one input read + one output write.
bytes_moved = 2 * ROWS * COLS * 4
custom_gbps = bytes_moved / (custom_ms * 1e6)
pytorch_gbps = bytes_moved / (pytorch_ms * 1e6)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Shape: ({ROWS}, {COLS})")

print("\n--- Correctness ---")
print(f"Maximum absolute error: {max_abs_error:.8e}")
print(f"Maximum relative error: {max_rel_error:.8e}")
print(f"Maximum row-sum error:  {row_sum_error:.8e}")
print(f"Allclose: {correct}")

print("\n--- Performance ---")
print(f"Custom CUDA: {custom_ms:.6f} ms")
print(f"PyTorch:     {pytorch_ms:.6f} ms")
print(f"Speedup:     {speedup:.3f}x")
print(f"Custom effective bandwidth:  {custom_gbps:.2f} GB/s")
print(f"PyTorch effective bandwidth: {pytorch_gbps:.2f} GB/s")

with open("comparison_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "rows", "cols", "custom_ms", "pytorch_ms", "speedup",
        "max_abs_error", "max_rel_error", "row_sum_error",
        "allclose", "custom_effective_GBps", "pytorch_effective_GBps"
    ])
    writer.writerow([
        ROWS, COLS, custom_ms, pytorch_ms, speedup,
        max_abs_error, max_rel_error, row_sum_error,
        correct, custom_gbps, pytorch_gbps
    ])