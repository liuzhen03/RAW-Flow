#!/usr/bin/env python
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Stage2Cfg, parse_config_arg
from data.factory import build_loaders
from engine import BaseTrainer, Logger, load_checkpoint, seed_everything
from models import (
    CrossScaleContextGuidance,
    DeterministicLatentFlowMatching,
    RawAE,
    RawFlowMatchingModel,
    RgbAE,
)
from models.utils import calculate_metrics, rggb_to_rgb

def _preprocess(batch):
    return {
        "rgb": batch["guidance_data"],
        "raw_rggb": batch["raw_data"],
        "context": batch.get("context_data"),
        "coordinates": batch.get("norm_coordinates"),
    }

def _load_ae(model: torch.nn.Module, path: str, device: str, key_candidates) -> torch.nn.Module:
    ckpt = load_checkpoint(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for k in key_candidates:
            if k in ckpt:
                state = ckpt[k]
                break
        else:
            state = ckpt.get("state_dict", ckpt)
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

class Stage2Trainer(BaseTrainer):
    def __init__(self, cfg, components, optimizers, train_loader, val_loader, logger, fm):
        super().__init__(
            components=components,
            optimizers=optimizers,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            device=cfg.trainer.device,
            epochs=cfg.trainer.epochs,
            clip_grad_norm=cfg.trainer.clip_grad_norm,
            val_freq=cfg.trainer.val_freq,
            quick_val_samples=cfg.trainer.quick_val_samples,
            save_dir=cfg.paths.save_dir,
            log_interval=cfg.trainer.log_interval,
        )
        self.cfg = cfg
        self.fm = fm

    def training_step(self, batch):
        data = _preprocess(batch)
        rgb_ae = self.components["rgb_ae"]
        raw_ae = self.components["raw_ae"]
        with torch.no_grad():
            rgb_latent, _, _ = rgb_ae.encode(data["rgb"])
            raw_latent, _, _ = raw_ae.encode(data["raw_rggb"])
        t = torch.rand(rgb_latent.shape[0], device=self.device)
        fm_loss, cond_loss = self.fm.compute_loss(
            raw_latent,
            t,
            rgb_latent,
            context=data["context"],
            coordinates=data["coordinates"],
        )
        total = fm_loss + self.cfg.losses.feature_weight * cond_loss
        return {"loss/main": total, "fm_loss": fm_loss.detach(), "cond_loss": cond_loss.detach()}

    def validation_step(self, batch, quick):
        data = _preprocess(batch)
        rgb_ae = self.components["rgb_ae"]
        raw_ae = self.components["raw_ae"]
        rgb_latent, rgb_fea2, rgb_fea4 = rgb_ae.encode(data["rgb"])
        raw_latent, _, _ = raw_ae.encode(data["raw_rggb"])
        steps = max(2, self.cfg.model.flow_steps // 4) if quick else self.cfg.model.flow_steps
        pred_raw_latent, _ = self.fm.sample_ode(
            shape=raw_latent.shape,
            rgb_latent=rgb_latent,
            num_steps=steps,
            context=data["context"],
            coordinates=data["coordinates"],
        )
        pred_raw = raw_ae.decode(pred_raw_latent, rgb_fea2, rgb_fea4)
        l2 = F.mse_loss(pred_raw, data["raw_rggb"])
        rggb_psnr, rggb_ssim = calculate_metrics(pred_raw, data["raw_rggb"], mode="rggb")
        pred_rgb = rggb_to_rgb(pred_raw)
        target_rgb = rggb_to_rgb(data["raw_rggb"])
        rgb_psnr, rgb_ssim = calculate_metrics(pred_rgb, target_rgb)
        return {
            "l2_loss": l2.item(),
            "rggb_psnr": float(rggb_psnr),
            "rggb_ssim": float(rggb_ssim),
            "psnr": float(rgb_psnr),
            "rgb_ssim": float(rgb_ssim),
        }

def main():
    cfg = parse_config_arg("Train stage 2 dual UNet", Stage2Cfg)
    seed_everything(cfg.trainer.seed)
    os.makedirs(cfg.paths.save_dir, exist_ok=True)

    train_loader, val_loader = build_loaders(cfg.data, cfg.trainer)

    rgb_ae = RgbAE(channels=cfg.model.channels, down_channels=cfg.model.down_channels)
    raw_ae = RawAE(channels=cfg.model.channels, down_channels=cfg.model.down_channels)
    if not cfg.paths.stage1_ckpt or not os.path.exists(cfg.paths.stage1_ckpt):
        raise FileNotFoundError(f"Missing stage1 checkpoint: {cfg.paths.stage1_ckpt}")
    rgb_ae = _load_ae(rgb_ae, cfg.paths.stage1_ckpt, cfg.trainer.device, ["rgb_model_state_dict", "rgb_state_dict"])
    raw_ae = _load_ae(raw_ae, cfg.paths.stage1_ckpt, cfg.trainer.device, ["raw_model_state_dict", "raw_state_dict"])

    cond_channels = [
        cfg.model.cond_unet_channels,
        cfg.model.cond_unet_channels * 2,
        cfg.model.cond_unet_channels * 4,
        cfg.model.cond_unet_channels * 4,
        cfg.model.cond_unet_channels * 4,
        cfg.model.cond_unet_channels * 2,
        cfg.model.cond_unet_channels,
    ]
    flow_unet = RawFlowMatchingModel(
        image_size=64,
        in_channels=cfg.model.down_channels,
        model_channels=cfg.model.flow_unet_channels,
        out_channels=cfg.model.down_channels,
        num_res_blocks=cfg.model.flow_unet_nums_res,
        channel_mult=(1, 2, 4),
        cond_channels=cond_channels,
    )
    cond_unet = CrossScaleContextGuidance(
        image_size=64,
        in_channels=cfg.model.down_channels,
        model_channels=cfg.model.cond_unet_channels,
        out_channels=cfg.model.down_channels,
        use_context=cfg.data.use_context,
        use_cond_loss=cfg.model.use_cond_loss,
    )
    flow_unet.to(cfg.trainer.device)
    cond_unet.to(cfg.trainer.device)

    fm = DeterministicLatentFlowMatching(
        flow_unet,
        cond_unet,
        mode=cfg.model.flow_matching_mode,
        use_context=cfg.data.use_context > 0,
        use_cond_loss=cfg.model.use_cond_loss,
    )

    optimizer = optim.Adam(list(flow_unet.parameters()) + list(cond_unet.parameters()), lr=cfg.trainer.lr)
    logger = Logger(save_dir=cfg.paths.save_dir)

    trainer = Stage2Trainer(
        cfg=cfg,
        components={"rgb_ae": rgb_ae, "raw_ae": raw_ae, "flow_unet": flow_unet, "cond_unet": cond_unet},
        optimizers={"main": optimizer},
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        fm=fm,
    )
    if cfg.paths.resume and os.path.exists(cfg.paths.resume):
        ckpt = load_checkpoint(cfg.paths.resume, map_location=cfg.trainer.device)
        trainer.load_state_dict(ckpt)
        logger.info(f"Resumed from {cfg.paths.resume} at epoch {trainer.start_epoch}")
    trainer.fit()

if __name__ == "__main__":
    main()
