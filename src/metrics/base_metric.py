import torch

try:
    from models.utils import rggb_to_rgb, rgb_to_rggb
except ImportError:

    def rggb_to_rgb(data):
        if len(data.shape) == 4:
            r, g1, g2, b = torch.chunk(data, 4, dim=1)
            g = (g1 + g2) / 2
            return torch.cat([r, g, b], dim=1)
        else:
            r, g1, g2, b = torch.chunk(data, 4, dim=0)
            g = (g1 + g2) / 2
            return torch.cat([r, g, b], dim=0)

    def rgb_to_rggb(data):
        if len(data.shape) == 4:
            r, g, b = torch.chunk(data, 3, dim=1)
            return torch.cat([r, g, g, b], dim=1)
        else:
            r, g, b = torch.chunk(data, 3, dim=0)
            return torch.cat([r, g, g, b], dim=0)

class BaseMetric:
    def __init__(
        self, rggb_to_rgb=False, pred_rgb_to_rggb=False, min_value=0, max_value=1
    ):
        self.rggb_to_rgb = rggb_to_rgb
        self.pred_rgb_to_rggb = pred_rgb_to_rggb
        self.min_value = min_value
        self.max_value = max_value

        assert not (rggb_to_rgb and pred_rgb_to_rggb), (
            "rggb_to_rgb and pred_rgb_to_rggb cannot be True at the same time"
        )

    def check_value(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if (pred < self.min_value).any() or (pred > self.max_value).any():
            print(
                f"Predicted value is not in range [{self.min_value}, {self.max_value}], got {pred.min()} and {pred.max()}"
            )
        if (target < self.min_value).any() or (target > self.max_value).any():
            print(
                f"Target value is not in range [{self.min_value}, {self.max_value}], got {target.min()} and {target.max()}"
            )

    def preprocess(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_value(pred, target)

        if self.rggb_to_rgb:
            pred = rggb_to_rgb(pred)
            target = rggb_to_rgb(target)

        if self.pred_rgb_to_rggb:
            pred = rgb_to_rggb(pred)

        return pred, target
