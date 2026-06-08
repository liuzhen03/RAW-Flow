#!/usr/bin/env python
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

from config import EvalCfg, parse_config_arg
from data.factory import build_eval_loader
from engine import seed_everything
from models import RawFlowWrapper
from models.utils import rggb_to_rgb

VIS_GAMMA = 0.2

def _to_chw(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor

def _to_numpy_hwc(tensor):
    return tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def _metrics_pair(pred: np.ndarray, target: np.ndarray, channel_axis: int):
    psnr = float(compute_psnr(target, pred, data_range=1.0))
    ssim = float(
        compute_ssim(target, pred, data_range=1.0, channel_axis=channel_axis, win_size=7)
    )
    mse = float(np.mean((pred - target) ** 2))
    pearson = _pearson(pred, target)
    return {"psnr": psnr, "ssim": ssim, "mse": mse, "pearson": pearson}

def _gamma_uint8(img_01: np.ndarray) -> np.ndarray:
    img = np.clip(img_01, 0.0, 1.0) ** VIS_GAMMA
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def _maybe_resize_rgb_to_raw(rgb_chw: torch.Tensor, target_hw: tuple) -> torch.Tensor:
    if rgb_chw.shape[1:] == target_hw:
        return rgb_chw
    t = rgb_chw.unsqueeze(0)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0)

def _save_visualization(
    vis_dir: str,
    base_name: str,
    rgb_input: torch.Tensor,
    pred_raw: torch.Tensor,
    gt_raw: torch.Tensor,
) -> None:
    os.makedirs(vis_dir, exist_ok=True)

    raw_hw = gt_raw.shape[1:]
    rgb_at_raw = _maybe_resize_rgb_to_raw(rgb_input, raw_hw)
    rgb_uint8 = np.clip(_to_numpy_hwc(rgb_at_raw) * 255.0, 0, 255).astype(np.uint8)

    pred_rgb_from_raw = rggb_to_rgb(pred_raw.unsqueeze(0))[0]
    gt_rgb_from_raw = rggb_to_rgb(gt_raw.unsqueeze(0))[0]
    pred_uint8 = _gamma_uint8(_to_numpy_hwc(pred_rgb_from_raw))
    gt_uint8 = _gamma_uint8(_to_numpy_hwc(gt_rgb_from_raw))

    imageio.imwrite(os.path.join(vis_dir, f"{base_name}_input_rgb.png"), rgb_uint8)
    imageio.imwrite(os.path.join(vis_dir, f"{base_name}_pred_raw.png"), pred_uint8)
    imageio.imwrite(os.path.join(vis_dir, f"{base_name}_gt_raw.png"), gt_uint8)

    if rgb_uint8.shape == pred_uint8.shape == gt_uint8.shape:
        comparison = np.concatenate([rgb_uint8, pred_uint8, gt_uint8], axis=1)
        imageio.imwrite(os.path.join(vis_dir, f"{base_name}_comparison.png"), comparison)

def _save_raw_npy(raw_dir: str, base_name: str, pred_raw: torch.Tensor) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    np.save(os.path.join(raw_dir, f"{base_name}_pred_raw.npy"), pred_raw.detach().cpu().numpy())

def evaluate(cfg: EvalCfg) -> dict:
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    vis_dir = os.path.join(cfg.output_dir, "visualization")
    raw_dir = os.path.join(cfg.output_dir, "raw")

    model = RawFlowWrapper(
        model_path=cfg.checkpoint,
        camera=cfg.camera,
        use_fusion=cfg.use_fusion,
        context_size=cfg.context_size,
        patch_size=cfg.patch_size,
        overlap=cfg.overlap,
        device=cfg.device,
    )

    loader = build_eval_loader(cfg)
    aggregate = {"rgb": [], "rggb": []}
    per_image = []

    pbar = tqdm(loader, desc="evaluate")
    for i, batch in enumerate(pbar):
        if cfg.limit and i >= cfg.limit:
            break
        rgb = _to_chw(batch["guidance_data"])
        raw_gt = _to_chw(batch["raw_data"])
        path = batch["path"][0] if isinstance(batch["path"], list) else batch["path"]
        base_name = os.path.splitext(os.path.basename(path))[0]

        rgb_01 = (rgb + 1.0) / 2.0
        raw_gt_01 = (raw_gt + 1.0) / 2.0

        with torch.no_grad():
            raw_pred_01 = model(rgb_01.to(cfg.device))
        if raw_pred_01.dim() == 4:
            raw_pred_01 = raw_pred_01.squeeze(0)

        pred_hwc = _to_numpy_hwc(raw_pred_01)
        gt_hwc = _to_numpy_hwc(raw_gt_01)
        rggb_metrics = _metrics_pair(pred_hwc, gt_hwc, channel_axis=2)
        pred_rgb_np = rggb_to_rgb(raw_pred_01.unsqueeze(0))[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        gt_rgb_np = rggb_to_rgb(raw_gt_01.unsqueeze(0))[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        rgb_metrics = _metrics_pair(pred_rgb_np, gt_rgb_np, channel_axis=2)

        aggregate["rggb"].append(rggb_metrics)
        aggregate["rgb"].append(rgb_metrics)
        per_image.append({"path": path, "rgb": rgb_metrics, "rggb": rggb_metrics})

        pbar.set_postfix(
            rgb_psnr=f"{rgb_metrics['psnr']:.2f}",
            rgb_ssim=f"{rgb_metrics['ssim']:.4f}",
            refresh=False,
        )
        tqdm.write(
            f"  {base_name}  rgb=PSNR {rgb_metrics['psnr']:.2f} SSIM {rgb_metrics['ssim']:.4f} "
            f"Pearson {rgb_metrics['pearson']:.4f} | rggb=PSNR {rggb_metrics['psnr']:.2f} "
            f"SSIM {rggb_metrics['ssim']:.4f}"
        )

        if cfg.save_visualizations:
            _save_visualization(
                vis_dir=vis_dir,
                base_name=base_name,
                rgb_input=rgb_01.detach().cpu(),
                pred_raw=raw_pred_01.detach().cpu(),
                gt_raw=raw_gt_01.detach().cpu(),
            )
        if cfg.save_raw_outputs:
            _save_raw_npy(raw_dir, base_name, raw_pred_01)

    summary = {}
    for domain in ("rgb", "rggb"):
        if not aggregate[domain]:
            continue
        keys = aggregate[domain][0].keys()
        summary[domain] = {k: float(np.mean([m[k] for m in aggregate[domain]])) for k in keys}

    out_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)

    print()
    print("=" * 60)
    print(f"Evaluated {len(per_image)} images")
    for domain, m in summary.items():
        kv = " ".join(f"{k}={v:.4f}" for k, v in m.items())
        print(f"[{domain}] {kv}")
    print(f"Metrics:        {out_path}")
    if cfg.save_visualizations:
        print(f"Visualizations: {vis_dir}/")
    if cfg.save_raw_outputs:
        print(f"Raw outputs:    {raw_dir}/")
    print("=" * 60)
    return summary

def main():
    cfg = parse_config_arg("Full-image RawFlow evaluation", EvalCfg)
    evaluate(cfg)

if __name__ == "__main__":
    main()
