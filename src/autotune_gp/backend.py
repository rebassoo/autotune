from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any

BackendName = Literal["numpy", "torch", "cupy"]
DeviceName = Literal["cpu", "cuda"]

@dataclass
class Backend:
    name: BackendName
    device: str
    xp: Any = None
    torch_device: Any = None

def get_backend(name: BackendName, device: DeviceName = "cpu") -> Backend:
    name = str(name).lower()
    device = str(device).lower()

    if name == "numpy":
        import numpy as np
        return Backend(name="numpy", device="cpu", xp=np)

    if name == "cupy":
        try:
            import cupy as cp
        except Exception as e:
            raise RuntimeError("backend=cupy requested but CuPy is not importable. Install a matching cupy-cuda package.") from e
        return Backend(name="cupy", device="cuda", xp=cp)

    if name == "torch":
        try:
            import torch
        except Exception as e:
            raise RuntimeError("backend=torch requested but PyTorch is not importable. Install torch (CUDA build if using cuda).") from e
        if device not in ("cpu", "cuda"):
            raise ValueError("torch device must be 'cpu' or 'cuda'")
        td = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        return Backend(name="torch", device=str(td), xp=torch, torch_device=td)

    raise ValueError("Unknown backend %r (expected numpy|torch|cupy)" % name)
