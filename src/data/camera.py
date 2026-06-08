from dataclasses import dataclass
from typing import Optional

@dataclass
class Camera:
    dataset: str
    name: str
    min_value: int
    black_level: int
    white_level: int

CAMERAS = [
    Camera("FiveK", "NIKON_D700", min_value=0, black_level=0, white_level=16383),
    Camera("FiveK", "Canon_EOS_5D", min_value=0, black_level=0, white_level=4095),
    Camera("PASCALRAW", "PASCALRAW", min_value=0, black_level=0, white_level=4095),
]

def _normalize(name: str) -> str:
    return name.lower().replace("_", "")

def get_camera(dataset: str, name: str) -> Optional[Camera]:
    ds = _normalize(dataset)
    nm = _normalize(name)
    for cam in CAMERAS:
        if _normalize(cam.dataset) == ds and _normalize(cam.name) == nm:
            return cam
    return None

def resolve_raw_range(dataset: str, camera: str, min_mode: str = "black_level"):
    cam = get_camera(dataset, camera)
    if cam is not None:
        if min_mode == "black_level":
            return cam.black_level, cam.white_level
        return cam.min_value, cam.white_level
    if dataset.lower() == "pascalraw":
        return 0, 4095
    raise ValueError(f"unknown camera: dataset={dataset}, camera={camera}")
