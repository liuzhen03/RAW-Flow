import os
from typing import Any, Dict, Optional

import torch

def save_checkpoint(state: Dict[str, Any], save_dir: str, filename: str = "checkpoint.pth", is_best: bool = False) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)
    return path

def _resolve_state_dict(ckpt, candidates):
    if isinstance(ckpt, dict):
        for key in candidates:
            if key in ckpt:
                return ckpt[key]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt

def load_checkpoint(
    path: str,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=weights_only)
    return ckpt

def load_into_module(
    module: torch.nn.Module,
    ckpt: Dict[str, Any],
    key_candidates: list,
    strict: bool = False,
) -> None:
    state = _resolve_state_dict(ckpt, key_candidates)
    module.load_state_dict(state, strict=strict)

def strip_optimizer(state: Dict[str, Any], extra_keys_to_drop=()) -> Dict[str, Any]:
    drop = {"optimizer_state_dict", "rgb_optimizer_state_dict", "raw_optimizer_state_dict",
            "scheduler_state_dict", "epoch", "best_psnr", "best_rggb_psnr",
            "best_rgb_psnr", "best_raw_psnr", "best_raw_psnr_rgb", "best_raw_psnr_rggb"}
    drop.update(extra_keys_to_drop)
    return {k: v for k, v in state.items() if k not in drop}
