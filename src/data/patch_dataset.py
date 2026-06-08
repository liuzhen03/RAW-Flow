import os
import re

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

_PATCH_RE = re.compile(r"_[0-9]\.npz$")

def _is_patch_file(filename: str) -> bool:
    return bool(_PATCH_RE.search(filename))

def _base_filename(filename: str) -> str:
    if _is_patch_file(filename):
        return _PATCH_RE.sub(".npz", filename)
    return filename

class PatchDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        file_list,
        raw_min_value=0,
        raw_max_value=16383,
        transforms=None,
        rgb_only=False,
        is_full=False,
        rgb_size="1X",
        use_context=0,
        use_coordinate=False,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.file_list = os.path.join(dataset_path, file_list)
        self.rgb_only = rgb_only
        self.is_full = is_full
        self.rgb_size = rgb_size
        self.use_context = use_context
        self.use_coordinate = use_coordinate
        self.transforms = transforms
        self.raw_min_value = raw_min_value
        self.raw_max_value = raw_max_value
        self.data = self._load()

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _np2tensor(arr):
        return torch.from_numpy(arr.copy()).permute(2, 0, 1)

    def _load(self):
        with open(self.file_list, "r") as f:
            items = [line.strip() for line in f.readlines() if line.strip()]

        data = []
        for item in items:
            parts = item.split(",")
            if len(parts) != 2:
                continue
            raw_rel, rgb_rel = parts

            raw_path = os.path.join(self.dataset_path, raw_rel)
            rgb_path = os.path.join(self.dataset_path, rgb_rel)
            is_patch = _is_patch_file(os.path.basename(raw_rel))

            context_path = None
            coord_path = None

            if self.is_full:
                if is_patch:
                    base = os.path.basename(_base_filename(raw_rel)).rsplit(".", 1)[0]
                    raw_dir = os.path.dirname(raw_rel)
                    raw_rel = os.path.join(raw_dir, f"{base}.npz")
                    raw_path = os.path.join(self.dataset_path, raw_rel)

                    rgb_dir = os.path.dirname(rgb_rel)
                    rgb_base = os.path.basename(rgb_rel)
                    for suffix in ("_1X.npy", "_2X.npy"):
                        if suffix in rgb_base:
                            rgb_base = rgb_base.replace(suffix, "")
                            break
                    rgb_base = re.sub(r"_[0-9]$", "", rgb_base)
                    rgb_rel = os.path.join(rgb_dir, f"{rgb_base}_{self.rgb_size}.npy")
                    rgb_path = os.path.join(self.dataset_path, rgb_rel)
                    if self.use_context > 0:
                        ctx_rel = os.path.join(rgb_dir, f"{rgb_base}_context_{self.use_context}.npy")
                        context_path = os.path.join(self.dataset_path, ctx_rel)
                else:
                    if self.rgb_size == "2X" and "_1X" in rgb_rel:
                        rgb_rel = rgb_rel.replace("_1X", "_2X")
                        rgb_path = os.path.join(self.dataset_path, rgb_rel)
                    if self.use_context > 0:
                        rgb_filename = os.path.basename(rgb_rel)
                        if "_1X.npy" in rgb_filename:
                            base_rgb = rgb_filename.replace("_1X.npy", "")
                        elif "_2X.npy" in rgb_filename:
                            base_rgb = rgb_filename.replace("_2X.npy", "")
                        else:
                            base_rgb = os.path.splitext(rgb_filename)[0]
                        rgb_dir = os.path.dirname(rgb_rel)
                        ctx_rel = os.path.join(rgb_dir, f"{base_rgb}_context_{self.use_context}.npy")
                        context_path = os.path.join(self.dataset_path, ctx_rel)
            else:
                if not is_patch:
                    continue
                if self.rgb_size == "2X" and "_1X" in rgb_rel:
                    rgb_rel = rgb_rel.replace("_1X", "_2X")
                    rgb_path = os.path.join(self.dataset_path, rgb_rel)

                if self.use_coordinate:
                    base = os.path.basename(raw_rel).rsplit(".", 1)[0]
                    raw_dir = os.path.dirname(raw_rel)
                    coord_rel = os.path.join(raw_dir, f"{base}_coordinates.txt")
                    coord_path = os.path.join(self.dataset_path, coord_rel)

                if self.use_context > 0:
                    raw_base = os.path.basename(raw_rel)
                    match = re.search(r"(.+?)_[0-9]\.npz$", raw_base)
                    base_name = match.group(1) if match else raw_base.rsplit(".", 1)[0]
                    rgb_dir = os.path.dirname(rgb_rel)
                    ctx_rel = os.path.join(rgb_dir, f"{base_name}_context_{self.use_context}.npy")
                    context_path = os.path.join(self.dataset_path, ctx_rel)

            if not ((self.rgb_only or os.path.exists(raw_path)) and os.path.exists(rgb_path)):
                continue

            entry = {"raw_path": raw_path, "rgb_path": rgb_path}
            if context_path is not None and (os.path.exists(context_path) or not self.use_context):
                entry["context_path"] = context_path
            if coord_path is not None and (os.path.exists(coord_path) or not self.use_coordinate):
                entry["coord_path"] = coord_path
            data.append(entry)

        return data

    @staticmethod
    def _load_coordinates(coord_path):
        coords = {}
        with open(coord_path, "r") as f:
            content = f.read().strip()
        lines = content.split("\n")
        if len(lines) == 1 and ":" not in content:
            parts = content.split(",")
            if len(parts) == 4:
                x1, y1, x2, y2 = map(int, parts)
                coords["raw"] = [x1, y1, x2, y2]
                width = x2 - x1
                height = y2 - y1
                est = max(width * 4, height * 4)
                coords["normalized"] = [
                    max(0.0, min(1.0, x1 / est)),
                    max(0.0, min(1.0, y1 / est)),
                    max(0.0, min(1.0, x2 / est)),
                    max(0.0, min(1.0, y2 / est)),
                ]
                coords["rgb_2x"] = [x1 * 2, y1 * 2, x2 * 2, y2 * 2]
            else:
                coords["normalized"] = [0.0, 0.0, 1.0, 1.0]
        else:
            for line in lines:
                line = line.strip()
                if line.startswith("normalized:"):
                    coords["normalized"] = [float(p) for p in line.replace("normalized:", "").split(",")]
                elif line.startswith("raw:"):
                    coords["raw"] = [int(p) for p in line.replace("raw:", "").split(",")]
                elif line.startswith("rgb_2x:"):
                    coords["rgb_2x"] = [int(p) for p in line.replace("rgb_2x:", "").split(",")]
        if "normalized" not in coords:
            coords["normalized"] = [0.0, 0.0, 1.0, 1.0]
        return coords

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_path = item["raw_path"]
        rgb_path = item["rgb_path"]

        if os.path.splitext(rgb_path)[1] == ".npy":
            rgb_data = np.load(rgb_path)
        else:
            rgb_data = imageio.imread(rgb_path)
        rgb_data = rgb_data.astype(np.float32) / 255

        cwb = None
        if not self.rgb_only:
            raw_npz = np.load(raw_path)
            raw_data = raw_npz["raw"]
            raw_data = (raw_data.astype(np.float32) - self.raw_min_value) / (
                self.raw_max_value - self.raw_min_value
            )
            cwb = raw_npz.get("cwb", None)
        else:
            raw_data = np.zeros((rgb_data.shape[0], rgb_data.shape[1], 4), dtype=np.float32)

        context_data = None
        if self.use_context > 0 and "context_path" in item and os.path.exists(item["context_path"]):
            context_data = np.load(item["context_path"]).astype(np.float32) / 255

        coordinates = None
        if self.is_full and self.use_coordinate:
            coordinates = {
                "normalized": [0.0, 0.0, 1.0, 1.0],
                "raw": [0, 0, raw_data.shape[1], raw_data.shape[0]],
            }
            if self.rgb_size == "2X":
                coordinates["rgb_2x"] = [0, 0, rgb_data.shape[1], rgb_data.shape[0]]
        elif self.use_coordinate and "coord_path" in item and os.path.exists(item["coord_path"]):
            coordinates = self._load_coordinates(item["coord_path"])

        if self.transforms is not None:
            result = self.transforms(
                raw_data=raw_data,
                rgb_data=rgb_data,
                context_data=context_data,
                coordinates=coordinates,
                rgb_size=self.rgb_size,
            )
            if self.use_context > 0 and self.use_coordinate:
                raw_data, rgb_data, context_data, coordinates = result
            elif self.use_context > 0:
                raw_data, rgb_data, context_data = result
            elif self.use_coordinate:
                raw_data, rgb_data, coordinates = result
            else:
                raw_data, rgb_data = result

        raw_data = np.clip(raw_data, 0, 1)
        rgb_data = np.clip(rgb_data, 0, 1)
        if context_data is not None:
            context_data = np.clip(context_data, 0, 1)

        raw_t = self._np2tensor(raw_data).float() * 2 - 1
        rgb_t = self._np2tensor(rgb_data).float() * 2 - 1

        out = {
            "raw_data": raw_t,
            "guidance_data": rgb_t,
            "path": os.path.relpath(rgb_path, self.dataset_path),
        }
        if context_data is not None:
            out["context_data"] = self._np2tensor(context_data).float() * 2 - 1
        if coordinates is not None:
            for key, value in coordinates.items():
                if key == "normalized":
                    out["norm_coordinates"] = torch.tensor(value, dtype=torch.float32)
                elif key in ("raw", "rgb_2x"):
                    out[f"{key}_coordinates"] = torch.tensor(value, dtype=torch.int32)
        if cwb is not None:
            out["cwb"] = torch.tensor(cwb, dtype=torch.float32)
        return out
