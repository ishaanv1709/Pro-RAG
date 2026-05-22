import torch
print("Compiling flashinfer sampling JIT -- takes 5-10 min, one time only...")
from flashinfer.sampling import top_k_top_p_sampling_from_probs
b = 1
p = torch.softmax(torch.randn(b, 151936, device="cuda"), -1).float()
u = torch.rand(b, device="cuda")
ks = torch.full((b,), 1, dtype=torch.int32, device="cuda")
ps = torch.full((b,), 1.0, dtype=torch.float32, device="cuda")
out, ok = top_k_top_p_sampling_from_probs(p, u, ks, ps)
print("Done — JIT cached. out shape:", out.shape)
