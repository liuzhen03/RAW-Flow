#!/usr/bin/env python
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import torch.optim as optim

from config import Stage1Cfg, parse_config_arg
from data.factory import build_loaders
from engine import BaseTrainer, Logger, load_checkpoint, seed_everything
from metrics import CollectionMetric, PSNRMetric, SSIMMetric
from models import RawAE, RgbAE
from models.utils import calculate_metrics, rggb_to_rgb

def _preprocess(batch):
    return {"rgb": batch["guidance_data"], "raw_rggb": batch["raw_data"]}

class Stage1Trainer(BaseTrainer):
    def __init__(self, cfg, components, optimizers, train_loader, val_loader, logger):
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

    def training_step(self, batch):
        data = _preprocess(batch)
        rgb_model = self.components["rgb"]
        raw_model = self.components["raw"]
        with torch.no_grad():
            rgb_latent, rgb_fea2, rgb_fea4 = rgb_model.encode(data["rgb"])
        rgb_out = rgb_model(data["rgb"], data["rgb"])
        rgb_l2 = rgb_out["l2_loss"]
        rgb_perceptual = rgb_out["perceptual_loss"]
        rgb_loss = self.cfg.losses.l2_weight * rgb_l2 + self.cfg.losses.perceptual_weight * rgb_perceptual

        raw_out = raw_model(
            data["raw_rggb"],
            data["raw_rggb"],
            rgb_features=(rgb_fea2, rgb_fea4),
            rgb_latent=rgb_latent,
        )
        feature_loss = raw_out.get("feature_loss", torch.tensor(0.0, device=self.device))
        raw_l2 = raw_out["l2_loss"]
        raw_perceptual = raw_out["perceptual_loss"]
        raw_loss = (
            self.cfg.losses.l2_weight * raw_l2
            + self.cfg.losses.perceptual_weight * raw_perceptual
            + self.cfg.losses.feature_weight * feature_loss
        )
        return {
            "loss/rgb": rgb_loss,
            "loss/raw": raw_loss,
            "rgb_l2": rgb_l2.detach(),
            "rgb_perceptual": rgb_perceptual.detach(),
            "raw_l2": raw_l2.detach(),
            "raw_perceptual": raw_perceptual.detach(),
            "raw_feature": feature_loss.detach() if isinstance(feature_loss, torch.Tensor) else torch.tensor(0.0),
        }

    def validation_step(self, batch, quick):
        data = _preprocess(batch)
        rgb_model = self.components["rgb"]
        raw_model = self.components["raw"]
        rgb_out = rgb_model(data["rgb"], data["rgb"])
        rgb_pred = rgb_out["recon_img"]
        rgb_latent, rgb_fea2, rgb_fea4 = rgb_model.encode(data["rgb"])
        raw_out = raw_model(
            data["raw_rggb"],
            data["raw_rggb"],
            rgb_features=(rgb_fea2, rgb_fea4),
            rgb_latent=rgb_latent,
        )
        raw_pred = raw_out["recon_img"]

        rgb_metrics = CollectionMetric()
        rgb_metrics.add(PSNRMetric(), "psnr")
        rgb_metrics.add(SSIMMetric(), "ssim")
        rgb_metrics.update((rgb_pred + 1) / 2, (data["rgb"] + 1) / 2)
        rgb_m = rgb_metrics.compute()

        raw_pred_rgb = rggb_to_rgb(raw_pred)
        raw_target_rgb = rggb_to_rgb(data["raw_rggb"])
        raw_metrics = CollectionMetric()
        raw_metrics.add(PSNRMetric(), "psnr")
        raw_metrics.add(SSIMMetric(), "ssim")
        raw_metrics.update((raw_pred_rgb + 1) / 2, (raw_target_rgb + 1) / 2)
        raw_m = raw_metrics.compute()

        return {
            "rgb_psnr": float(rgb_m["psnr"]),
            "rgb_ssim": float(rgb_m["ssim"]),
            "psnr": float(raw_m["psnr"]),
            "raw_ssim": float(raw_m["ssim"]),
        }

def main():
    cfg = parse_config_arg("Train stage 1 autoencoder", Stage1Cfg)
    seed_everything(cfg.trainer.seed)
    os.makedirs(cfg.paths.save_dir, exist_ok=True)

    train_loader, val_loader = build_loaders(cfg.data, cfg.trainer)

    rgb_model = RgbAE(channels=cfg.model.channels, down_channels=cfg.model.down_channels)
    raw_model = RawAE(channels=cfg.model.channels, down_channels=cfg.model.down_channels)
    rgb_opt = optim.Adam(rgb_model.parameters(), lr=cfg.trainer.lr)
    raw_opt = optim.Adam(raw_model.parameters(), lr=cfg.trainer.lr)

    logger = Logger(save_dir=cfg.paths.save_dir)
    trainer = Stage1Trainer(
        cfg=cfg,
        components={"rgb": rgb_model, "raw": raw_model},
        optimizers={"rgb": rgb_opt, "raw": raw_opt},
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
    )
    if cfg.paths.resume and os.path.exists(cfg.paths.resume):
        ckpt = load_checkpoint(cfg.paths.resume, map_location=cfg.trainer.device)
        trainer.load_state_dict(ckpt)
        logger.info(f"Resumed from {cfg.paths.resume} at epoch {trainer.start_epoch}")
    trainer.fit()

if __name__ == "__main__":
    main()
