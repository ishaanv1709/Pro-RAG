import torch
import triton
import triton.language as tl


@triton.jit
def _cosine_kernel(
    Q, D, Out,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    dot   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    q_sq  = tl.zeros((BLOCK_M,),         dtype=tl.float32)
    d_sq  = tl.zeros((BLOCK_N,),         dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)

        q = tl.load(Q + m[:, None] * K + k[None, :],
                    mask=(m[:, None] < M) & (k[None, :] < K), other=0.0)
        d = tl.load(D + n[:, None] * K + k[None, :],
                    mask=(n[:, None] < N) & (k[None, :] < K), other=0.0)

        dot  += tl.dot(q, tl.trans(d))
        q_sq += tl.sum(q * q, axis=1)
        d_sq += tl.sum(d * d, axis=1)

    result = dot / (tl.sqrt(q_sq[:, None] * d_sq[None, :]) + 1e-8)
    tl.store(Out + m[:, None] * N + n[None, :], result,
             mask=(m[:, None] < M) & (n[None, :] < N))


def triton_cosine(queries, docs):
    M, K = queries.shape
    N     = docs.shape[0]
    out   = torch.empty((M, N), device=queries.device, dtype=torch.float32)
    grid  = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    _cosine_kernel[grid](queries, docs, out, M, N, K, 32, 32, 64)
    return out


def torch_cosine(queries, docs):
    q = torch.nn.functional.normalize(queries.float(), dim=-1)
    d = torch.nn.functional.normalize(docs.float(),    dim=-1)
    return q @ d.T


_notified = set()

def _notify_once(key, msg):
    if key not in _notified:
        _notified.add(key)
        print(msg)


def cosine_similarity(queries, docs, use_triton=True):
    if use_triton and queries.is_cuda and docs.is_cuda:
        try:
            out = triton_cosine(queries, docs)
            _notify_once("triton", "[kernel] Triton cosine kernel ACTIVE (GPU)")
            return out
        except Exception as e:
            _notify_once("fallback", f"[kernel] Triton FAILED → PyTorch fallback: {e}")
    elif use_triton:
        _notify_once("cpu", "[kernel] use_triton=True but tensors not on CUDA → PyTorch fallback")
    return torch_cosine(queries, docs)
