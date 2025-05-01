import os
import json
import tempfile
from pathlib import Path
import pytest
import torch
import triton
import triton.language as tl
target = triton.runtime.driver.active.get_current_target()


def test_compilation_tracing(device):
    # Test tracing during actual kernel compilation
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nLogs will be saved at: {log_dir.absolute()}")
        print("=====")
        triton.knobs.structured_logging.triton_trace = temp_dir

        # Define a simple kernel
        @triton.jit
        def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
            xnumel = 10
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:]
            xmask = xindex < xnumel
            x0 = xindex
            tmp0 = tl.load(in_ptr0 + (x0), xmask)
            tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

        # Create tensors and run kernel
        inp = torch.randn(10, device=device)
        out = torch.zeros(10, device=device)
        kernel[(10, )](inp, out, 10, XBLOCK=16)

        # Check that log files were created
        log_files = list(Path(temp_dir).glob("dedicated_log_triton_trace_*.json"))
        assert len(log_files) >= 1

        # Check log content
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                assert "payload" in data
                assert "file_path" in data['payload']
            except json.JSONDecodeError as e:
                print(e)
                pytest.fail("Log file does not contain valid JSON content")
