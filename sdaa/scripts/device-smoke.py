#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
import torch_sdaa

device = torch.device("sdaa:0")
left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
right = torch.tensor([[2.0, 0.0], [1.0, 2.0]], device=device)
product = left @ right
torch.sdaa.synchronize()
observed = product.cpu().tolist()
if observed != [[4.0, 4.0], [10.0, 8.0]]:
    raise RuntimeError(f"Unexpected SDAA matrix result: {observed}")

rng = {}
try:
    torch.sdaa.manual_seed_all(2026)
    first = torch.rand(4, device=device)
    torch.sdaa.synchronize()
    torch.sdaa.manual_seed_all(2026)
    second = torch.rand(4, device=device)
    torch.sdaa.synchronize()
    rng["global_seed_repeatable"] = bool(torch.equal(first, second))
    rng["values"] = first.cpu().tolist()
except Exception as exc:
    rng["global_seed_error"] = f"{type(exc).__name__}: {exc}"
try:
    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    values = torch.rand(4, device=device, generator=generator)
    torch.sdaa.synchronize()
    rng["device_generator_values"] = values.cpu().tolist()
except Exception as exc:
    rng["device_generator_error"] = f"{type(exc).__name__}: {exc}"

result = {
    "uid": os.getuid(),
    "gid": os.getgid(),
    "torch": torch.__version__,
    "torch_sdaa": torch_sdaa.__version__,
    "sdaa_available": torch.sdaa.is_available(),
    "device_count": torch.sdaa.device_count(),
    "device_0": torch.sdaa.get_device_name(0),
    "matrix_product": observed,
    "rng": rng,
}
destination = Path("/workspace/repro/work/device-smoke.json")
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
