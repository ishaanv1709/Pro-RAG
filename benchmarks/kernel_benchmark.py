"""
Microbenchmark: Triton vs PyTorch cosine similarity
Run: python benchmarks/kernel_benchmark.py
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SIZES   = [(64, 1000, 768), (32, 5000, 768), (16, 10000, 1024)]
WARMUP  = 5
ITERS   = 50


def bench(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    if args[0].is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(*args)
    if args[0].is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000


def torch_cosine(q, d):
    q = torch.nn.functional.normalize(q, dim=-1)
    d = torch.nn.functional.normalize(d, dim=-1)
    return q @ d.T


print(f"\n{'Config':<25} {'PyTorch CPU':>14} {'PyTorch CUDA':>14} {'Triton':>12} {'Speedup':>10}")
print("-" * 80)

for M, N, D in SIZES:
    q_cpu = torch.randn(M, D)
    d_cpu = torch.randn(N, D)
    cpu_ms = bench(torch_cosine, q_cpu, d_cpu)

    cuda_ms    = None
    triton_ms  = None

    if torch.cuda.is_available():
        from src.kernels.cosine_similarity import triton_cosine, cosine_similarity
        q_gpu = q_cpu.cuda()
        d_gpu = d_cpu.cuda()
        cuda_ms   = bench(torch_cosine, q_gpu, d_gpu)
        try:
            triton_ms = bench(triton_cosine, q_gpu, d_gpu)
        except Exception as e:
            print(f"  Triton error: {e}")

    label    = f"M={M} N={N} D={D}"
    cuda_str = f"{cuda_ms:.2f}" if cuda_ms else "N/A"
    tri_str  = f"{triton_ms:.2f}" if triton_ms else "N/A"
    speedup  = f"{cuda_ms/triton_ms:.2f}x" if (cuda_ms and triton_ms) else "N/A"

    print(f"{label:<25} {cpu_ms:>12.2f}ms {cuda_str:>12}ms {tri_str:>10}ms {speedup:>10}")
