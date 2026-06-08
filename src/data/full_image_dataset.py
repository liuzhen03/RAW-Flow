import os

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

class FullImageDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        file_list,
        raw_min_value=0,
        raw_max_value=16383,
        transforms=None,
        rgb_only=False,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.file_list_path = (
            file_list if os.path.isabs(file_list) else os.path.join(dataset_path, file_list)
        )
        self.rgb_only = rgb_only
        self.transforms = transforms
        self.raw_min_value = raw_min_value
        self.raw_max_value = raw_max_value
        self.data = self._load()

    def _load(self):
        data = []
        if not os.path.exists(self.file_list_path):
            return data
        with open(self.file_list_path, "r") as f:
            items = [line.strip() for line in f.readlines() if line.strip()]
        for item in items:
            parts = item.split(",")
            if len(parts) != 2:
                continue
            raw_rel, rgb_rel = parts
            raw_path = os.path.join(self.dataset_path, raw_rel)
            rgb_path = os.path.join(self.dataset_path, rgb_rel)
            if not os.path.exists(rgb_path):
                continue
            if not self.rgb_only and not os.path.exists(raw_path):
                continue
            data.append((raw_path, rgb_path))
        return data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _np2tensor(arr):
        return torch.from_numpy(arr.copy()).permute(2, 0, 1)

    def __getitem__(self, idx):
        raw_path, rgb_path = self.data[idx]

        if os.path.splitext(rgb_path)[1].lower() == ".npy":
            rgb_loaded = np.load(rgb_path)
        else:
            rgb_loaded = imageio.imread(rgb_path)

        if isinstance(rgb_loaded, np.lib.npyio.NpzFile):
            if "arr_0" in rgb_loaded:
                rgb_data = rgb_loaded["arr_0"]
            elif "rgb" in rgb_loaded:
                rgb_data = rgb_loaded["rgb"]
            else:
                raise ValueError(f"Cannot find image data in {rgb_path}")
        else:
            rgb_data = rgb_loaded

        if rgb_data.dtype != np.float32 or rgb_data.max() > 1.0:
            rgb_data = rgb_data.astype(np.float32) / 255.0
        rgb_data = np.clip(rgb_data, 0.0, 1.0)

        if not self.rgb_only:
            raw_npz = np.load(raw_path)
            if "raw" in raw_npz:
                raw_data = raw_npz["raw"]
            elif "arr_0" in raw_npz:
                raw_data = raw_npz["arr_0"]
            else:
                raise ValueError(f"Cannot find 'raw' or 'arr_0' in {raw_path}")
            raw_data = (raw_data.astype(np.float32) - self.raw_min_value) / (
                self.raw_max_value - self.raw_min_value
            )
            raw_data = np.clip(raw_data, 0.0, 1.0)
        else:
            raw_data = np.zeros((rgb_data.shape[0], rgb_data.shape[1], 4), dtype=np.float32)

        if self.transforms is not None:
            raw_data, rgb_data = self.transforms(raw_data, rgb_data)

        raw_t = self._np2tensor(raw_data).float() * 2.0 - 1.0
        rgb_t = self._np2tensor(rgb_data).float() * 2.0 - 1.0

        return {
            "raw_data": raw_t,
            "guidance_data": rgb_t,
            "path": os.path.relpath(
                rgb_path, self.dataset_path if self.dataset_path else os.path.dirname(rgb_path)
            ),
        }
