
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from functools import partial
from einops import rearrange
import math

from .utils import calculate_metrics, rgb_to_rggb, rggb_to_rgb

class DeterministicLatentFlowMatching:
    def __init__(self, model, cond_unet, mode='rgb_latent', use_context=False, use_cond_loss=True):
        self.model = model
        self.cond_unet = cond_unet
        self.mode = mode
        self.use_context = use_context
        self.use_cond_loss = use_cond_loss
        if mode not in ['noise', 'rgb_latent']:
            raise ValueError(f"Mode {mode} not supported. Use 'noise' or 'rgb_latent'")

    def compute_loss(self, x_1, t, rgb_latent, context=None, coordinates=None):

        cond_input = context if self.use_context and context is not None else rgb_latent

        cond_output, cond_features = self.cond_unet(cond_input, coordinates)

        if self.mode == 'noise':

            x_0 = torch.randn_like(x_1)
        elif self.mode == 'rgb_latent':

            x_0 = rgb_latent

        t = t[:, None, None, None]
        x_t = (1 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0

        predicted_velocity = self.model(x_t, t.squeeze(-1).squeeze(-1).squeeze(-1), cond_features)

        fm_loss = F.mse_loss(predicted_velocity, target_velocity)

        if self.use_cond_loss and cond_output is not None:
            cond_loss = F.mse_loss(cond_output, x_1)
        else:
            cond_loss = torch.tensor(0.0, device=fm_loss.device)

        return fm_loss, cond_loss

    def sample_ode(self, shape, rgb_latent, num_steps=50, context=None, coordinates=None):
        device = next(self.model.parameters()).device

        cond_input = context if self.use_context and context is not None else rgb_latent

        cond_output, cond_features = self.cond_unet(cond_input, coordinates)

        if self.mode == 'noise':

            x_t = torch.randn(shape, device=device)

        elif self.mode == 'rgb_latent':

            x_t = rgb_latent

        dt = 1.0 / num_steps
        t_values = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]

        for i in range(num_steps):
            t = t_values[i].expand(shape[0]).float()
            velocity = self.model(x_t, t, cond_features)
            x_t = x_t + dt * velocity
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                print(f"Warning: NaN or Inf detected at step {i}")
                break

        return x_t, cond_output

    def set_mode(self, mode):
        if mode not in ['noise', 'rgb_latent']:
            raise ValueError(f"Mode {mode} not supported. Use 'noise' or 'rgb_latent'")
        self.mode = mode
        return self

    def set_use_context(self, use_context):
        self.use_context = use_context
        return self

    def set_use_cond_loss(self, use_cond_loss):
        self.use_cond_loss = use_cond_loss
        return self
