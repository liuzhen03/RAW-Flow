from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .logger import Logger

class BaseTrainer(ABC):

    def __init__(
        self,
        components: Dict[str, torch.nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: Logger,
        device: str,
        epochs: int,
        clip_grad_norm: float = 1.0,
        val_freq: int = 1,
        quick_val_samples: int = 20,
        save_dir: str = "checkpoints/run",
        log_interval: int = 200,
    ):
        self.components = components
        self.optimizers = optimizers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = device
        self.epochs = epochs
        self.clip_grad_norm = clip_grad_norm
        self.val_freq = val_freq
        self.quick_val_samples = quick_val_samples
        self.save_dir = save_dir
        self.log_interval = max(1, log_interval)
        self.start_epoch = 0
        self.best_metric: float = float("-inf")
        self.global_step = 0
        for module in components.values():
            module.to(device)

    @abstractmethod
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def validation_step(self, batch: Dict[str, Any], quick: bool) -> Dict[str, float]:
        ...

    def state_dict(self) -> Dict[str, Any]:
        out = {"epoch": self.start_epoch, "global_step": self.global_step}
        for name, module in self.components.items():
            out[f"{name}_state_dict"] = module.state_dict()
        for name, opt in self.optimizers.items():
            out[f"{name}_optimizer_state_dict"] = opt.state_dict()
        out["best_metric"] = self.best_metric
        return out

    def load_state_dict(self, ckpt: Dict[str, Any]) -> None:
        for name, module in self.components.items():
            key = f"{name}_state_dict"
            if key in ckpt:
                module.load_state_dict(ckpt[key], strict=False)
        for name, opt in self.optimizers.items():
            key = f"{name}_optimizer_state_dict"
            if key in ckpt:
                opt.load_state_dict(ckpt[key])
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_metric = float(ckpt.get("best_metric", float("-inf")))

    def _route_losses_and_step(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        scalars: Dict[str, float] = {}
        for key, loss in losses.items():
            if not key.startswith("loss/"):
                if isinstance(loss, torch.Tensor):
                    scalars[key] = loss.detach().item()
                else:
                    scalars[key] = float(loss)
                continue
            opt_name = key[len("loss/"):]
            if opt_name not in self.optimizers:
                continue
            opt = self.optimizers[opt_name]
            opt.zero_grad()
            loss.backward()
            if self.clip_grad_norm > 0:
                params = []
                for module in self.components.values():
                    params.extend(p for p in module.parameters() if p.requires_grad)
                torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
            opt.step()
            scalars[key] = loss.detach().item()
        return scalars

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        for module in self.components.values():
            module.train()
        running: Dict[str, float] = {}
        n_batches = 0
        n_steps_in_window = 0
        window: Dict[str, float] = {}

        pbar = tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{self.epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = _move_to_device(batch, self.device)
            losses = self.training_step(batch)
            scalars = self._route_losses_and_step(losses)
            n_batches += 1
            n_steps_in_window += 1
            self.global_step += 1
            for k, v in scalars.items():
                running[k] = running.get(k, 0.0) + v
                window[k] = window.get(k, 0.0) + v

            loss_keys = [k for k in scalars if k.startswith("loss/")]
            if loss_keys:
                pbar.set_postfix(
                    {k: f"{running[k] / n_batches:.4f}" for k in loss_keys},
                    refresh=False,
                )

            if self.global_step % self.log_interval == 0:
                avg = {k: v / n_steps_in_window for k, v in window.items()}
                self.logger.log(avg, step=self.global_step, phase="train")
                self.logger.info(
                    f"epoch {epoch + 1}/{self.epochs} step {self.global_step}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
                )
                window = {}
                n_steps_in_window = 0

        pbar.close()
        if n_batches > 0:
            running = {k: v / n_batches for k, v in running.items()}
        return running

    @torch.no_grad()
    def _validate(self, epoch: int, quick: bool) -> Dict[str, float]:
        for module in self.components.values():
            module.eval()
        running: Dict[str, float] = {}
        n = 0
        max_n = self.quick_val_samples if quick else None
        for batch in self.val_loader:
            if max_n is not None and n >= max_n:
                break
            batch = _move_to_device(batch, self.device)
            metrics = self.validation_step(batch, quick=quick)
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + float(v)
            n += 1
        if n > 0:
            running = {k: v / n for k, v in running.items()}
        running["val_samples"] = n
        phase = "val_quick" if quick else "val"
        self.logger.log(running, step=self.global_step, phase=phase)
        return running

    def _select_metric(self, val_metrics: Dict[str, float]) -> Optional[float]:
        for key in ("psnr", "rgb_psnr", "raw_psnr", "psnr_rgb", "psnr_rggb"):
            if key in val_metrics:
                return val_metrics[key]
        return None

    def _format_metrics(self, m: Dict[str, float]) -> str:
        return "  ".join(
            f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, (int, float)) and k != "val_samples"
        )

    def _print_banner(self) -> None:
        bar = "=" * 60
        self.logger.info(bar)
        self.logger.info(f"Trainer       : {type(self).__name__}")
        self.logger.info(f"Save dir      : {self.save_dir}")
        self.logger.info(f"Device        : {self.device}")
        self.logger.info(f"Epochs        : {self.epochs}  (resume from {self.start_epoch})")
        self.logger.info(
            f"Train batches : {len(self.train_loader)}  |  Val batches : {len(self.val_loader)}"
        )
        for name, opt in self.optimizers.items():
            lrs = [group["lr"] for group in opt.param_groups]
            self.logger.info(f"Optimizer[{name}] : {type(opt).__name__}  lr={lrs}")
        for name, module in self.components.items():
            n_params = sum(p.numel() for p in module.parameters())
            n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            self.logger.info(
                f"Component[{name}] : params={n_params:,}  trainable={n_train:,}"
            )
        self.logger.info(f"Log interval  : every {self.log_interval} steps")
        self.logger.info(f"Val freq      : every {self.val_freq} epoch(s), quick={self.quick_val_samples} samples")
        self.logger.info(bar)

    def fit(self) -> None:
        self._print_banner()
        for epoch in range(self.start_epoch, self.epochs):
            self.start_epoch = epoch
            train_metrics = self._train_one_epoch(epoch)
            self.logger.log(train_metrics, step=self.global_step, phase="train_epoch")
            self.logger.info(
                f"[epoch {epoch + 1}/{self.epochs} train]  {self._format_metrics(train_metrics)}"
            )

            quick = (epoch + 1) % self.val_freq != 0
            val_metrics = self._validate(epoch, quick=quick)
            phase = "val_quick" if quick else "val"
            metric = self._select_metric(val_metrics)
            is_best = metric is not None and metric > self.best_metric
            if is_best:
                self.best_metric = metric
            best_tag = "  *new best*" if is_best else ""
            self.logger.info(
                f"[epoch {epoch + 1}/{self.epochs} {phase}]   {self._format_metrics(val_metrics)}{best_tag}"
            )

            save_checkpoint(self.state_dict(), self.save_dir, filename="checkpoint.pth", is_best=is_best)

        self.logger.info(f"Training finished. Best metric: {self.best_metric:.4f}")
        self.logger.close()

def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(b, device) for b in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch
