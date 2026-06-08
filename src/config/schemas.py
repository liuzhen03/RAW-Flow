from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf

@dataclass
class DataCfg:
    dataset: str = "fivek"
    camera: str = "NIKON_D700"
    data_dir: str = "./data/fivek_dataset_processed/fivek_patches_3_complete_dataset"
    file_list_train: str = "train"
    file_list_val: str = "test"
    patch_size: int = 256
    rgb_size: str = "2X"
    use_context: int = 0
    use_coordinate: bool = False
    is_full: bool = False
    num_workers: int = 4
    min_mode: str = "black_level"

@dataclass
class TrainerCfg:
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1.0e-4
    seed: int = 42
    clip_grad_norm: float = 1.0
    val_freq: int = 1
    quick_val_samples: int = 20
    amp: bool = False
    log_interval: int = 200
    device: str = "cuda:0"
    patience: int = 5

@dataclass
class PathsCfg:
    save_dir: str = "checkpoints/run"
    resume: Optional[str] = None
    stage1_ckpt: Optional[str] = None
    stage2_ckpt: Optional[str] = None

@dataclass
class Stage1ModelCfg:
    channels: int = 96
    down_channels: int = 6

@dataclass
class Stage1LossCfg:
    l2_weight: float = 1.0
    perceptual_weight: float = 0.01
    feature_weight: float = 0.1

@dataclass
class Stage1Cfg:
    data: DataCfg = field(default_factory=DataCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    model: Stage1ModelCfg = field(default_factory=Stage1ModelCfg)
    losses: Stage1LossCfg = field(default_factory=Stage1LossCfg)

@dataclass
class Stage2ModelCfg:
    channels: int = 96
    down_channels: int = 6
    flow_unet_channels: int = 96
    cond_unet_channels: int = 48
    unet_nums_res: int = 3
    flow_unet_nums_res: int = 3
    flow_steps: int = 20
    flow_matching_mode: str = "rgb_latent"
    use_cond_loss: bool = False

@dataclass
class Stage2LossCfg:
    feature_weight: float = 0.5

@dataclass
class Stage2Cfg:
    data: DataCfg = field(default_factory=DataCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    model: Stage2ModelCfg = field(default_factory=Stage2ModelCfg)
    losses: Stage2LossCfg = field(default_factory=Stage2LossCfg)

@dataclass
class Stage3FreezeCfg:
    rgb_encoder: bool = False
    raw_decoder: bool = False
    flow_model: bool = False
    cond_unet: bool = False
    fusion: bool = False

@dataclass
class Stage3ModelCfg:
    channels: int = 96
    down_channels: int = 6
    flow_unet_channels: int = 96
    cond_unet_channels: int = 48
    unet_nums_res: int = 3
    flow_unet_nums_res: int = 3
    flow_steps: int = 20
    flow_matching_mode: str = "rgb_latent"
    use_cond_loss: bool = False
    use_fusion: bool = False
    fusion_hidden_dim: int = 128
    fusion_num_heads: int = 4
    freeze: Stage3FreezeCfg = field(default_factory=Stage3FreezeCfg)

@dataclass
class Stage3Cfg:
    data: DataCfg = field(default_factory=DataCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    model: Stage3ModelCfg = field(default_factory=Stage3ModelCfg)
    fusion_lr: float = 1.0e-4

@dataclass
class EvalCfg:
    data: DataCfg = field(default_factory=DataCfg)
    checkpoint: str = "./ckpt/nikon_d700.pth"
    camera: str = "NIKON_D700"
    use_fusion: bool = False
    context_size: int = 256
    patch_size: int = 512
    overlap: int = 16
    device: str = "cuda:0"
    batch_size: int = 1
    num_workers: int = 4
    seed: int = 42
    limit: int = 0
    output_dir: str = "./test_output/rawflow"
    save_visualizations: bool = True
    save_raw_outputs: bool = False
    visualization_quality: int = 95

def load_config(yaml_path: str, schema_cls):
    schema = OmegaConf.structured(schema_cls)
    file_cfg = OmegaConf.load(yaml_path) if yaml_path else OmegaConf.create({})
    cli_cfg = OmegaConf.from_cli()
    merged = OmegaConf.merge(schema, file_cfg, cli_cfg)
    OmegaConf.resolve(merged)
    return OmegaConf.to_object(merged)

def parse_config_arg(description: str, schema_cls):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args, overrides = parser.parse_known_args()

    schema = OmegaConf.structured(schema_cls)
    file_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(schema, file_cfg, cli_cfg)
    OmegaConf.resolve(merged)
    return OmegaConf.to_object(merged)
