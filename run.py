import torch
import triton
import triton.language as tl

@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
print("here5")
device = 'cuda'

import triton.profiler as proton
proton.start("kernela", hook="structured_logging")
inp = torch.randn(10, device=device)
out = torch.zeros(10, device=device)
kernel[(10, )](inp, out, 10, XBLOCK=16)
proton.finalize()
