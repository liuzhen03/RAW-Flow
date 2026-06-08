#!/usr/bin/env python
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import torch.optim as optim

from config import Stage3Cfg, parse_config_arg
from data.factory import build_loaders
from engine import BaseTrainer, Logger, load_checkpoint, seed_everything
from models import (
    CrossScaleContextGuidance,
    FusionModule,
    RawAE,
    RawFlowMatchingModel,
    RawFlowModel,
    RgbAE,
)

def _preprocess(batch):
    return {
        "rgb": batch["guidance_data"],
        "raw_rggb": batch["raw_data"],
        "context": batch.get("context_data"),
        "coordinates": batch.get("norm_coordinates"),
    }

def _load_state(path, key_candidates, device):
    ckpt = load_checkpoint(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        return ckpt
    for k in key_candidates:
        if k in ckpt:
            return ckpt[k]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt

def _apply_freeze(model: RawFlowModel, freeze_spec) -> None:
    spec = {
        "rgb_encoder": freeze_spec.rgb_encoder,
        "raw_decoder": freeze_spec.raw_decoder,
        "flow_model": freeze_spec.flow_model,
        "cond_unet": freeze_spec.cond_unet,
        "fusion": freeze_spec.fusion,
    }
    name_map = {
        "rgb_encoder": "rgb_encoder",
        "raw_decoder": "raw_decoder",
        "flow_model": "flow_model",
        "cond_unet": "cond_unet",
        "fusion": "fusion_module",
    }
    for key, freeze in spec.items():
        attr = name_map[key]
        module = getattr(model, attr, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = not freeze

class Stage3Trainer(BaseTrainer):
    def __init__(self, cfg, model: RawFlowModel, optimizer, train_loader, val_loader, logger):
        super().__init__(
            components={"model": model},
            optimizers={"main": optimizer},
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

    def training_step(self, batch):
        data = _preprocess(batch)
        model: RawFlowModel = self.components["model"]
        outputs = model(data["rgb"], data["raw_rggb"], context=data["context"], coordinates=data["coordinates"])
        return {
            "loss/main": outputs["total_loss"],
            "l1": outputs["l1_loss"].detach(),
            "l2": outputs["l2_loss"].detach(),
            "log_l1": outputs["log_l1_loss"].detach(),
            "perceptual": outputs["perceptual_loss"].detach(),
        }

    def validation_step(self, batch, quick):
        data = _preprocess(batch)
        model: RawFlowModel = self.components["model"]
        outputs = model(data["rgb"], data["raw_rggb"], context=data["context"], coordinates=data["coordinates"])
        return {
            "l1_loss": float(outputs["l1_loss"].item()),
            "l2_loss": float(outputs["l2_loss"].item()),
            "total_loss": float(outputs["total_loss"].item()),
            "rggb_psnr": float(outputs["psnr_rggb"]),
            "rggb_ssim": float(outputs["ssim_rggb"]),
            "psnr": float(outputs["psnr_rgb"]),
            "rgb_ssim": float(outputs["ssim_rgb"]),
        }

def main():
    cfg = parse_config_arg("Train stage 3 end-to-end fine-tuning", Stage3Cfg)
    seed_everything(cfg.trainer.seed)
    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    device = cfg.trainer.device

    train_loader, val_loader = build_loaders(cfg.data, cfg.trainer)

    rgb_ae = RgbAE(channels=cfg.model.channels, down_channels=cfg.model.down_channels).to(device)
    raw_ae = RawAE(channels=cfg.model.channels, down_channels=cfg.model.down_channels).to(device)

    if cfg.paths.stage1_ckpt and os.path.exists(cfg.paths.stage1_ckpt):
        rgb_ae.load_state_dict(
            _load_state(cfg.paths.stage1_ckpt, ["rgb_model_state_dict", "rgb_state_dict"], device),
            strict=False,
        )
        raw_ae.load_state_dict(
            _load_state(cfg.paths.stage1_ckpt, ["raw_model_state_dict", "raw_state_dict"], device),
            strict=False,
        )

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
    ).to(device)
    cond_unet = CrossScaleContextGuidance(
        image_size=64,
        in_channels=cfg.model.down_channels,
        model_channels=cfg.model.cond_unet_channels,
        out_channels=cfg.model.down_channels,
        use_context=cfg.data.use_context,
        use_cond_loss=cfg.model.use_cond_loss,
    ).to(device)
    if cfg.paths.stage2_ckpt and os.path.exists(cfg.paths.stage2_ckpt):
        flow_unet.load_state_dict(
            _load_state(cfg.paths.stage2_ckpt, ["flow_unet_state_dict", "model_state_dict", "flow_model_state_dict"], device),
            strict=False,
        )
        cond_unet.load_state_dict(
            _load_state(cfg.paths.stage2_ckpt, ["cond_unet_state_dict", "cond_model_state_dict"], device),
            strict=False,
        )

    fusion = None
    if cfg.model.use_fusion:
        fusion = FusionModule(
            in_channels=cfg.model.down_channels,
            cond_channels=cfg.model.down_channels,
            n_feat=cfg.model.fusion_hidden_dim,
            num_blocks=cfg.model.fusion_num_heads,
        ).to(device)

    model = RawFlowModel(
        rgb_encoder=rgb_ae.encoder,
        raw_decoder=raw_ae.decoder,
        flow_model=flow_unet,
        cond_unet=cond_unet,
        fusion_module=fusion,
        flow_steps=cfg.model.flow_steps,
        flow_matching_mode=cfg.model.flow_matching_mode,
        use_fusion=cfg.model.use_fusion,
        use_context=cfg.data.use_context > 0,
        use_cond_loss=cfg.model.use_cond_loss,
    ).to(device)

    _apply_freeze(model, cfg.model.freeze)

    if fusion is not None and cfg.fusion_lr != cfg.trainer.lr:
        groups = [
            {"params": [p for n, p in model.named_parameters() if not n.startswith("fusion_module") and p.requires_grad], "lr": cfg.trainer.lr},
            {"params": [p for p in model.fusion_module.parameters() if p.requires_grad], "lr": cfg.fusion_lr},
        ]
        optimizer = optim.Adam(groups)
    else:
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.trainer.lr)

    logger = Logger(save_dir=cfg.paths.save_dir)
    trainer = Stage3Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
    )
    if cfg.paths.resume and os.path.exists(cfg.paths.resume):
        ckpt = load_checkpoint(cfg.paths.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "main_optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["main_optimizer_state_dict"])
            trainer.start_epoch = int(ckpt.get("epoch", 0))
            trainer.global_step = int(ckpt.get("global_step", 0))
            trainer.best_metric = float(ckpt.get("best_metric", ckpt.get("best_rggb_psnr", ckpt.get("best_psnr", 0.0))))
        else:
            trainer.load_state_dict(ckpt)
        logger.info(f"Resumed from {cfg.paths.resume} at epoch {trainer.start_epoch}")
    trainer.fit()

if __name__ == "__main__":
    main()
