import argparse
import os
import sys
import traceback
from pathlib import Path

import torch

_LOCAL_VLLM_REPO = Path(__file__).resolve().parent
if os.environ.get("VLLM_REPRO_USE_INSTALLED") == "1":
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != _LOCAL_VLLM_REPO]
elif (_LOCAL_VLLM_REPO / "vllm").is_dir():
    sys.path.insert(0, str(_LOCAL_VLLM_REPO))


def describe_norm(norm, label):
    weight = getattr(norm, "weight", None)
    params = dict(norm.named_parameters())
    buffers = dict(norm.named_buffers())
    print(f"[{label}] type={type(norm).__name__}")
    print(f"[{label}] weight_type={type(weight).__name__ if weight is not None else None}")
    print(f"[{label}] weight_device={getattr(weight, 'device', None)}")
    print(f"[{label}] in_parameters={'weight' in params}")
    print(f"[{label}] in_buffers={'weight' in buffers}")
    print(f"[{label}] in_state_dict={'weight' in norm.state_dict()}")


def make_config():
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig

    cfg = VllmConfig()
    cfg.kernel_config.ir_op_priority = IrOpPriorityConfig.with_default(
        ["vllm_c", "native"],
        rms_norm=["vllm_c", "native"],
        fused_add_rms_norm=["vllm_c", "native"],
    )
    return cfg


def run_direct_rmsnorm(device):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    with set_current_vllm_config(make_config()):
        with torch.device("cpu"):
            norm = RMSNorm(128, eps=1e-6, has_weight=False)

    describe_norm(norm, "direct_before_to")
    norm.to(device)
    describe_norm(norm, "direct_after_to")

    x = torch.randn(4, 128, device=device)
    print("[direct] running forward")
    y = norm(x)
    torch.cuda.synchronize()
    print(f"[direct] ok output_shape={tuple(y.shape)} output_device={y.device}")


def run_gemma4_router_norm(device):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.gemma4 import Gemma4Router

    config = SimpleNamespace(
        hidden_size=128,
        rms_norm_eps=1e-6,
        num_experts=4,
    )
    with set_current_vllm_config(make_config()):
        with torch.device("cpu"):
            router = Gemma4Router(config)

    norm = router.norm
    describe_norm(norm, "gemma4_router_before_to")
    router.to(device)
    describe_norm(norm, "gemma4_router_after_to")

    x = torch.randn(4, 128, device=device)
    print("[gemma4_router] running router.norm forward")
    y = norm(x)
    torch.cuda.synchronize()
    print(f"[gemma4_router] ok output_shape={tuple(y.shape)} output_device={y.device}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["direct", "gemma4-router"], default="direct")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This repro needs CUDA."
    try:
        if args.case == "direct":
            run_direct_rmsnorm(args.device)
        else:
            run_gemma4_router_norm(args.device)
    except Exception:
        print("[repro] failed with exception:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
