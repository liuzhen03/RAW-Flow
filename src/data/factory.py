from torch.utils.data import DataLoader

from .camera import resolve_raw_range
from .full_image_dataset import FullImageDataset
from .patch_dataset import PatchDataset
from .transforms import ImageTransforms, ImageTransformsContext

def _build_transform(patch_size, is_train, use_context_or_coord, rgb_size):
    if use_context_or_coord or rgb_size == "2X":
        return ImageTransformsContext(patch_size=patch_size, is_train=is_train)
    return ImageTransforms(patch_size=patch_size, is_train=is_train)

def build_patch_dataset(data_cfg, split, is_train, transform=True):
    file_list = data_cfg.file_list_train if split == "train" else data_cfg.file_list_val
    raw_min, raw_max = resolve_raw_range(
        data_cfg.dataset, data_cfg.camera, getattr(data_cfg, "min_mode", "black_level")
    )
    needs_context = data_cfg.use_context > 0 or data_cfg.use_coordinate or data_cfg.rgb_size == "2X"
    transforms = (
        _build_transform(data_cfg.patch_size, is_train, needs_context, data_cfg.rgb_size)
        if transform
        else None
    )
    return PatchDataset(
        dataset_path=data_cfg.data_dir,
        file_list=file_list,
        raw_min_value=raw_min,
        raw_max_value=raw_max,
        transforms=transforms,
        is_full=data_cfg.is_full,
        rgb_size=data_cfg.rgb_size,
        use_context=data_cfg.use_context,
        use_coordinate=data_cfg.use_coordinate,
    )

def build_full_image_dataset(data_cfg, split="val"):
    file_list = data_cfg.file_list_train if split == "train" else data_cfg.file_list_val
    raw_min, raw_max = resolve_raw_range(
        data_cfg.dataset, data_cfg.camera, getattr(data_cfg, "min_mode", "black_level")
    )
    return FullImageDataset(
        dataset_path=data_cfg.data_dir,
        file_list=file_list,
        raw_min_value=raw_min,
        raw_max_value=raw_max,
        transforms=None,
    )

def build_loaders(data_cfg, trainer_cfg):
    train_ds = build_patch_dataset(data_cfg, split="train", is_train=True)
    val_ds = build_patch_dataset(data_cfg, split="val", is_train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=trainer_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

def build_eval_loader(eval_cfg):
    ds = build_full_image_dataset(eval_cfg.data, split="val")
    return DataLoader(
        ds,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=eval_cfg.num_workers,
        pin_memory=True,
    )
