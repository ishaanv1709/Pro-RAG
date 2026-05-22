import torch
import pytest
from src.kernels.cosine_similarity import cosine_similarity


def ref(q, d):
    q = torch.nn.functional.normalize(q, dim=-1)
    d = torch.nn.functional.normalize(d, dim=-1)
    return q @ d.T


@pytest.mark.parametrize("M,N,D", [(4, 8, 64), (1, 5, 128), (16, 16, 256)])
def test_cpu_matches_reference(M, N, D):
    q, d = torch.randn(M, D), torch.randn(N, D)
    assert torch.allclose(cosine_similarity(q, d), ref(q, d), atol=1e-5)


def test_output_shape():
    out = cosine_similarity(torch.randn(3, 64), torch.randn(10, 64))
    assert out.shape == (3, 10)


def test_self_similarity_is_one():
    v = torch.nn.functional.normalize(torch.randn(5, 32), dim=-1)
    assert torch.allclose(cosine_similarity(v, v).diagonal(), torch.ones(5), atol=1e-5)


def test_orthogonal_is_zero():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    d = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert abs(cosine_similarity(q, d).item()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_triton_matches_torch():
    from src.kernels.cosine_similarity import triton_cosine
    q = torch.randn(8, 128, device="cuda")
    d = torch.randn(16, 128, device="cuda")
    assert torch.allclose(triton_cosine(q, d), ref(q.cpu(), d.cpu()).cuda(), atol=1e-4)
