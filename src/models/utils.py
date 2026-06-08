
import numpy as np
import torch
import torch.nn.functional as F

try:
    from metrics import CollectionMetric, PSNRMetric, SSIMMetric
    EVALUATION_MODULE_AVAILABLE = True
except ImportError:
    EVALUATION_MODULE_AVAILABLE = False

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

def gamma_correction(t, gamma=1.0 / 5):
    t = t.clip(0, 1)
    t = t**gamma
    return t

def calculate_metrics(pred, target, mode='rgb', use_evaluation_module=True):

    if mode not in ['rgb', 'rggb']:
        raise ValueError(f"Mode {mode} not supported. Use 'rgb' or 'rggb'.")

    pred_norm = ((pred + 1) / 2).clamp(0, 1)
    target_norm = ((target + 1) / 2).clamp(0, 1)

    if use_evaluation_module and EVALUATION_MODULE_AVAILABLE:
        metrics = CollectionMetric()

        if mode == 'rgb':

            if pred_norm.shape[1] == 4:
                psnr_metric = PSNRMetric(rggb_to_rgb=True)
                ssim_metric = SSIMMetric(rggb_to_rgb=True)
            else:
                psnr_metric = PSNRMetric()
                ssim_metric = SSIMMetric()
        else:

            if pred_norm.shape[1] == 3:
                psnr_metric = PSNRMetric(pred_rgb_to_rggb=True)
                ssim_metric = SSIMMetric(pred_rgb_to_rggb=True)
            else:
                psnr_metric = PSNRMetric()
                ssim_metric = SSIMMetric()

        metrics.add(psnr_metric, 'psnr')
        metrics.add(ssim_metric, 'ssim')

        metrics.update(pred_norm, target_norm)
        result = metrics.compute()

        return result['psnr'], result['ssim']

    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn

        pred_np = pred_norm.detach().cpu().numpy()
        target_np = target_norm.detach().cpu().numpy()

        if mode == 'rgb':

            if pred_np.shape[1] == 4:
                pred_np = rggb_to_rgb(torch.from_numpy(pred_np)).numpy()
                target_np = rggb_to_rgb(torch.from_numpy(target_np)).numpy()
        else:

            if pred_np.shape[1] == 3:
                pred_np = rgb_to_rggb(torch.from_numpy(pred_np)).numpy()
                target_np = rgb_to_rggb(torch.from_numpy(target_np)).numpy()

        batch_size = pred_np.shape[0]
        psnr_values = []
        ssim_values = []

        for i in range(batch_size):
            p = np.transpose(pred_np[i], (1, 2, 0))
            t = np.transpose(target_np[i], (1, 2, 0))

            psnr = psnr_fn(t, p, data_range=1.0)

            ssim = ssim_fn(t, p, data_range=1.0, channel_axis=2)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

        return np.mean(psnr_values), np.mean(ssim_values)

    except ImportError:
        print("Warning: metrics calculation failed. Using dummy values.")
        return 0.0, 0.0

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
