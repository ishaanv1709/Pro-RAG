#!/usr/bin/env python3
"""
Patch SGLang sampler.py to use PyTorch multinomial as fallback when flashinfer JIT fails.
Run from WSL2: python3 /mnt/c/Users/ishaa_04bpft8/Desktop/adaptive-rag-system/fix_flashinfer.py
(curand headers and flashinfer cache already cleaned in previous run)
"""
from pathlib import Path

sampler = Path("/home/ishaa_04bpft8/adaptive-rag-system/.venv/lib/python3.10/site-packages/sglang/srt/layers/sampler.py")
if not sampler.exists():
    print(f"ERROR: {sampler} not found"); exit(1)

src = sampler.read_text()

if "PYTORCH_FALLBACK_PATCH" in src:
    print("sampler.py already patched."); exit(0)

lines = src.splitlines(keepends=True)

# Find the call: batch_next_token_ids = top_k_top_p_sampling_from_probs(
target = "batch_next_token_ids = top_k_top_p_sampling_from_probs("
call_start = None
for i, line in enumerate(lines):
    if target in line:
        call_start = i
        break

if call_start is None:
    print("ERROR: call site not found"); exit(1)

# Find end of call using parenthesis counting
depth, call_end = 0, call_start
for i in range(call_start, len(lines)):
    depth += lines[i].count('(') - lines[i].count(')')
    if depth <= 0:
        call_end = i
        break

# Determine indentation of the call line
indent = len(lines[call_start]) - len(lines[call_start].lstrip())
pad = ' ' * indent

print(f"Wrapping lines {call_start+1}–{call_end+1}:")
print("".join(lines[call_start:call_end+1]))

# Build replacement: try/except wrapping the original call
call_block = "".join("    " + l for l in lines[call_start:call_end+1])
replacement = (
    f"{pad}# PYTORCH_FALLBACK_PATCH\n"
    f"{pad}try:\n"
    + call_block +
    f"{pad}except Exception:\n"
    f"{pad}    import torch as _t\n"
    f"{pad}    batch_next_token_ids = _t.multinomial(\n"
    f"{pad}        probs.float().clamp(min=1e-10), 1\n"
    f"{pad}    ).squeeze(-1).to(_t.int32)\n"
)

new_src = "".join(lines[:call_start]) + replacement + "".join(lines[call_end+1:])
sampler.write_text(new_src)
print("Patched! PyTorch fallback added to sampler.py")
print("\nNow start SGLang:")
print("  python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct-AWQ --port 30000 --disable-radix-cache")
