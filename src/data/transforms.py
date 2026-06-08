import random

import numpy as np

class ImageTransforms:
    def __init__(self, patch_size, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train

    def random_flip(self, raw_data, rgb_data):
        idx = np.random.randint(2)
        return np.flip(raw_data, axis=idx).copy(), np.flip(rgb_data, axis=idx).copy()

    def random_rotate(self, raw_data, rgb_data):
        k = np.random.randint(4)
        return np.rot90(raw_data, k=k), np.rot90(rgb_data, k=k)

    def random_crop(self, raw_data, rgb_data):
        h, w, _ = raw_data.shape
        rh = random.randint(0, max(0, h - self.patch_size))
        rw = random.randint(0, max(0, w - self.patch_size))
        return (
            raw_data[rh:rh + self.patch_size, rw:rw + self.patch_size, :],
            rgb_data[rh:rh + self.patch_size, rw:rw + self.patch_size, :],
        )

    def center_crop(self, raw_data, rgb_data):
        h, w, _ = raw_data.shape
        oy = (h - self.patch_size) // 2
        ox = (w - self.patch_size) // 2
        return (
            raw_data[oy:oy + self.patch_size, ox:ox + self.patch_size, :],
            rgb_data[oy:oy + self.patch_size, ox:ox + self.patch_size, :],
        )

    def __call__(self, raw_data, rgb_data):
        assert raw_data.shape[:2] == rgb_data.shape[:2]
        if self.is_train:
            raw_data, rgb_data = self.random_crop(raw_data, rgb_data)
            raw_data, rgb_data = self.random_rotate(raw_data, rgb_data)
            raw_data, rgb_data = self.random_flip(raw_data, rgb_data)
        else:
            raw_data, rgb_data = self.center_crop(raw_data, rgb_data)
        return raw_data, rgb_data

def _pack(raw, rgb, context, coords):
    if context is not None and coords is not None:
        return raw, rgb, context, coords
    if context is not None:
        return raw, rgb, context
    if coords is not None:
        return raw, rgb, coords
    return raw, rgb

class ImageTransformsContext:
    def __init__(self, patch_size, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train

    def random_flip(self, raw_data, rgb_data, context_data=None, coordinates=None, rgb_size=None):
        idx = np.random.randint(2)
        h, w = raw_data.shape[:2]
        raw_data = np.flip(raw_data, axis=idx).copy()
        rgb_data = np.flip(rgb_data, axis=idx).copy()
        if context_data is not None:
            context_data = np.flip(context_data, axis=idx).copy()
        if coordinates is not None:
            if idx == 0:
                if "normalized" in coordinates:
                    x1, y1, x2, y2 = coordinates["normalized"]
                    coordinates["normalized"] = [1.0 - x2, y1, 1.0 - x1, y2]
                if "raw" in coordinates:
                    x1, y1, x2, y2 = coordinates["raw"]
                    coordinates["raw"] = [w - x2, y1, w - x1, y2]
                if "rgb_2x" in coordinates:
                    x1, y1, x2, y2 = coordinates["rgb_2x"]
                    rw = w * 2 if rgb_size == "2X" else w
                    coordinates["rgb_2x"] = [rw - x2, y1, rw - x1, y2]
            else:
                if "normalized" in coordinates:
                    x1, y1, x2, y2 = coordinates["normalized"]
                    coordinates["normalized"] = [x1, 1.0 - y2, x2, 1.0 - y1]
                if "raw" in coordinates:
                    x1, y1, x2, y2 = coordinates["raw"]
                    coordinates["raw"] = [x1, h - y2, x2, h - y1]
                if "rgb_2x" in coordinates:
                    x1, y1, x2, y2 = coordinates["rgb_2x"]
                    rh = h * 2 if rgb_size == "2X" else h
                    coordinates["rgb_2x"] = [x1, rh - y2, x2, rh - y1]
        return _pack(raw_data, rgb_data, context_data, coordinates)

    def random_rotate(self, raw_data, rgb_data, context_data=None, coordinates=None, rgb_size=None):
        k = np.random.randint(4)
        if k == 0:
            return _pack(raw_data, rgb_data, context_data, coordinates)
        h, w = raw_data.shape[:2]
        raw_data = np.rot90(raw_data, k=k)
        rgb_data = np.rot90(rgb_data, k=k)
        if context_data is not None:
            context_data = np.rot90(context_data, k=k)
        if coordinates is not None:
            rh = h * 2 if rgb_size == "2X" else h
            rw = w * 2 if rgb_size == "2X" else w
            if k == 1:
                if "normalized" in coordinates:
                    x1, y1, x2, y2 = coordinates["normalized"]
                    coordinates["normalized"] = [1.0 - y2, x1, 1.0 - y1, x2]
                if "raw" in coordinates:
                    x1, y1, x2, y2 = coordinates["raw"]
                    coordinates["raw"] = [h - y2, x1, h - y1, x2]
                if "rgb_2x" in coordinates:
                    x1, y1, x2, y2 = coordinates["rgb_2x"]
                    coordinates["rgb_2x"] = [rh - y2, x1, rh - y1, x2]
            elif k == 2:
                if "normalized" in coordinates:
                    x1, y1, x2, y2 = coordinates["normalized"]
                    coordinates["normalized"] = [1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1]
                if "raw" in coordinates:
                    x1, y1, x2, y2 = coordinates["raw"]
                    coordinates["raw"] = [w - x2, h - y2, w - x1, h - y1]
                if "rgb_2x" in coordinates:
                    x1, y1, x2, y2 = coordinates["rgb_2x"]
                    coordinates["rgb_2x"] = [rw - x2, rh - y2, rw - x1, rh - y1]
            else:
                if "normalized" in coordinates:
                    x1, y1, x2, y2 = coordinates["normalized"]
                    coordinates["normalized"] = [y1, 1.0 - x2, y2, 1.0 - x1]
                if "raw" in coordinates:
                    x1, y1, x2, y2 = coordinates["raw"]
                    coordinates["raw"] = [y1, w - x2, y2, w - x1]
                if "rgb_2x" in coordinates:
                    x1, y1, x2, y2 = coordinates["rgb_2x"]
                    coordinates["rgb_2x"] = [y1, rw - x2, y2, rw - x1]
        return _pack(raw_data, rgb_data, context_data, coordinates)

    def _crop_coords(self, coordinates, image_h, image_w, oy, ox, h_new, w_new, rgb_size):
        if coordinates is None:
            return None
        new_coords = {}
        if "normalized" in coordinates:
            ox1, oy1, ox2, oy2 = coordinates["normalized"]
            ow, oh = ox2 - ox1, oy2 - oy1
            new_coords["normalized"] = [
                ox1 + (ox / image_w) * ow,
                oy1 + (oy / image_h) * oh,
                ox1 + ((ox + w_new) / image_w) * ow,
                oy1 + ((oy + h_new) / image_h) * oh,
            ]
        if "raw" in coordinates:
            ox1, oy1, ox2, oy2 = coordinates["raw"]
            ow, oh = ox2 - ox1, oy2 - oy1
            new_coords["raw"] = [
                ox1 + int((ox / image_w) * ow),
                oy1 + int((oy / image_h) * oh),
                ox1 + int(((ox + w_new) / image_w) * ow),
                oy1 + int(((oy + h_new) / image_h) * oh),
            ]
        if "rgb_2x" in coordinates:
            ox1, oy1, ox2, oy2 = coordinates["rgb_2x"]
            ow, oh = ox2 - ox1, oy2 - oy1
            if rgb_size == "2X":
                rgb_ox = ox * 2
                rgb_oy = oy * 2
                rgb_w = w_new * 2
                rgb_h = h_new * 2
                new_coords["rgb_2x"] = [
                    ox1 + int((rgb_ox / (image_w * 2)) * ow),
                    oy1 + int((rgb_oy / (image_h * 2)) * oh),
                    ox1 + int(((rgb_ox + rgb_w) / (image_w * 2)) * ow),
                    oy1 + int(((rgb_oy + rgb_h) / (image_h * 2)) * oh),
                ]
            else:
                new_coords["rgb_2x"] = [
                    ox1 + int((ox / image_w) * ow),
                    oy1 + int((oy / image_h) * oh),
                    ox1 + int(((ox + w_new) / image_w) * ow),
                    oy1 + int(((oy + h_new) / image_h) * oh),
                ]
        return new_coords

    def random_crop(self, raw_data, rgb_data, context_data=None, coordinates=None, rgb_size=None):
        h, w, _ = raw_data.shape
        rh = random.randint(0, max(0, h - self.patch_size))
        rw = random.randint(0, max(0, w - self.patch_size))
        patch_raw = raw_data[rh:rh + self.patch_size, rw:rw + self.patch_size, :]
        if rgb_size == "2X":
            rgb_rh, rgb_rw = rh * 2, rw * 2
            rgb_ps = self.patch_size * 2
            patch_rgb = rgb_data[rgb_rh:rgb_rh + rgb_ps, rgb_rw:rgb_rw + rgb_ps, :]
        else:
            patch_rgb = rgb_data[rh:rh + self.patch_size, rw:rw + self.patch_size, :]
        new_coords = self._crop_coords(coordinates, h, w, rh, rw, self.patch_size, self.patch_size, rgb_size)
        return _pack(patch_raw, patch_rgb, context_data, new_coords)

    def center_crop(self, raw_data, rgb_data, context_data=None, coordinates=None, rgb_size=None):
        h, w, _ = raw_data.shape
        oy = (h - self.patch_size) // 2
        ox = (w - self.patch_size) // 2
        patch_raw = raw_data[oy:oy + self.patch_size, ox:ox + self.patch_size, :]
        if rgb_size == "2X":
            rgb_oy, rgb_ox = oy * 2, ox * 2
            rgb_ps = self.patch_size * 2
            patch_rgb = rgb_data[rgb_oy:rgb_oy + rgb_ps, rgb_ox:rgb_ox + rgb_ps, :]
        else:
            patch_rgb = rgb_data[oy:oy + self.patch_size, ox:ox + self.patch_size, :]
        new_coords = self._crop_coords(coordinates, h, w, oy, ox, self.patch_size, self.patch_size, rgb_size)
        return _pack(patch_raw, patch_rgb, context_data, new_coords)

    def __call__(self, raw_data, rgb_data, context_data=None, coordinates=None, rgb_size="1X"):
        if rgb_size == "2X":
            if rgb_data.shape[0] != raw_data.shape[0] * 2 or rgb_data.shape[1] != raw_data.shape[1] * 2:
                raise ValueError(
                    f"2X mode requires RGB to be 2x RAW. RGB shape: {rgb_data.shape}, RAW shape: {raw_data.shape}"
                )
        else:
            assert raw_data.shape[:2] == rgb_data.shape[:2]

        if self.is_train:
            result = self.random_crop(raw_data, rgb_data, context_data, coordinates, rgb_size)
            result = self._apply(self.random_rotate, result, rgb_size)
            result = self._apply(self.random_flip, result, rgb_size)
        else:
            result = self.center_crop(raw_data, rgb_data, context_data, coordinates, rgb_size)
        return result

    @staticmethod
    def _apply(fn, packed, rgb_size):
        if len(packed) == 4:
            return fn(packed[0], packed[1], packed[2], packed[3], rgb_size)
        if len(packed) == 3:
            third = packed[2]
            if isinstance(third, dict):
                return fn(packed[0], packed[1], None, third, rgb_size)
            return fn(packed[0], packed[1], third, None, rgb_size)
        return fn(packed[0], packed[1], None, None, rgb_size)
