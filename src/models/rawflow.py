
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from .DLAE import DualDomainLatentAutoencoderRAW, DualDomainLatentAutoencoderRGB
from .DLFM import DeterministicLatentFlowMatching
from .fusion import FusionModule
from .utils import AverageMeter, calculate_metrics, rgb_to_rggb, rggb_to_rgb

class hard_log_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = (-1 * torch.log(1 - torch.clamp(torch.abs(x - y), 0, 1) + 1e-6)).mean()
        return loss

class RawFlowModel(nn.Module):
    def __init__(
        self,
        rgb_encoder,
        raw_decoder,
        flow_model,
        cond_unet,
        fusion_module=None,
        flow_steps=50,
        flow_matching_mode='rgb_latent',
        use_fusion=True,
        unfreeze_all=True,
        perceptual_loss=None,
        use_context=False,
        use_cond_loss=True
    ):
        super(RawFlowModel, self).__init__()
        self.rgb_encoder = rgb_encoder
        self.raw_decoder = raw_decoder
        self.flow_model = flow_model
        self.cond_unet = cond_unet
        self.fusion_module = fusion_module
        self.flow_steps = flow_steps
        self.use_fusion = use_fusion and fusion_module is not None
        self.use_context = use_context
        self.use_cond_loss = use_cond_loss
        self.flow_matching = DeterministicLatentFlowMatching(
            flow_model,
            cond_unet,
            mode=flow_matching_mode,
            use_context=use_context,
            use_cond_loss=use_cond_loss
        )

        if unfreeze_all:
            for param in self.parameters():
                param.requires_grad = True

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.hard_log_criterion = hard_log_loss()
        if perceptual_loss is not None:
            self.perceptual_loss = perceptual_loss
        else:
            try:
                from .DLAE import VGGPerceptualLoss
                self.perceptual_loss = VGGPerceptualLoss()
                self.perceptual_loss = self.perceptual_loss.to(next(self.parameters()).device)
            except Exception:
                self.perceptual_loss = nn.MSELoss()

    def forward(self, rgb_input, raw_target=None, context=None, coordinates=None):
        outputs = {}

        rgb_latent, rgb_fea2, rgb_fea4 = self.rgb_encoder(rgb_input)
        outputs['rgb_latent'] = rgb_latent

        pred_raw_latent, cond_output = self.flow_matching.sample_ode(
            shape=rgb_latent.shape,
            rgb_latent=rgb_latent,
            num_steps=self.flow_steps,
            context=context,
            coordinates=coordinates
        )
        outputs['pred_raw_latent'] = pred_raw_latent
        outputs['cond_output'] = cond_output

        if self.use_fusion and self.fusion_module is not None:
            fused_latent = self.fusion_module(pred_raw_latent, cond_output)
            outputs['fused_latent'] = fused_latent
            final_latent = fused_latent
        else:
            final_latent = pred_raw_latent

        pred_raw_img = self.raw_decoder(final_latent, rgb_fea2, rgb_fea4)
        outputs['pred_raw_img'] = pred_raw_img

        if raw_target is not None:

            l1_loss = self.l1_loss(pred_raw_img, raw_target)
            outputs['l1_loss'] = l1_loss

            l2_loss = self.l2_loss(pred_raw_img, raw_target)
            outputs['l2_loss'] = l2_loss

            eps = 1e-8
            log_pred = torch.log(torch.clamp((pred_raw_img + 1) / 2, min=eps))
            log_target = torch.log(torch.clamp((raw_target + 1) / 2, min=eps))
            log_l1_loss = F.l1_loss(log_pred, log_target)
            outputs['log_l1_loss'] = log_l1_loss

            raw_hard_log_loss = self.hard_log_criterion((pred_raw_img + 1) / 2, (raw_target + 1) / 2)

            if pred_raw_img.shape[1] == 4:
                pred_rgb = rggb_to_rgb(pred_raw_img)
                target_rgb = rggb_to_rgb(raw_target)
                perceptual_loss = self.perceptual_loss(pred_rgb, target_rgb)
            else:
                perceptual_loss = self.perceptual_loss(pred_raw_img, raw_target)
            outputs['perceptual_loss'] = perceptual_loss

            total_loss = l1_loss + l2_loss + log_l1_loss + 0.01 * perceptual_loss
            outputs['total_loss'] = total_loss

            with torch.no_grad():

                psnr_rggb, ssim_rggb = calculate_metrics(pred_raw_img, raw_target, mode='rggb')
                outputs['psnr_rggb'] = psnr_rggb
                outputs['ssim_rggb'] = ssim_rggb

                psnr_rgb, ssim_rgb = calculate_metrics(pred_raw_img, raw_target, mode='rgb')
                outputs['psnr'] = psnr_rgb
                outputs['ssim'] = ssim_rgb
                outputs['psnr_rgb'] = psnr_rgb
                outputs['ssim_rgb'] = ssim_rgb

        return outputs

    def set_use_fusion(self, use_fusion):
        self.use_fusion = use_fusion and self.fusion_module is not None
        return self

    def set_use_context(self, use_context):
        self.use_context = use_context
        self.flow_matching.set_use_context(use_context)
        return self

    def set_use_cond_loss(self, use_cond_loss):
        self.use_cond_loss = use_cond_loss
        self.flow_matching.set_use_cond_loss(use_cond_loss)
        return self

    @classmethod
    def from_auto_encoders(cls, rgb_ae, raw_ae, flow_model, cond_unet, fusion_module=None,
                      flow_steps=50, use_fusion=True, unfreeze_all=True, perceptual_loss=None,
                      use_context=False, use_cond_loss=True):

        rgb_encoder = rgb_ae.encoder if hasattr(rgb_ae, 'encoder') else rgb_ae
        raw_decoder = raw_ae.decoder if hasattr(raw_ae, 'decoder') else raw_ae

        device = next(rgb_encoder.parameters()).device
        if perceptual_loss is None:
            try:
                from models.DLAE import VGGPerceptualLoss
                perceptual_loss = VGGPerceptualLoss().to(device)
            except ImportError:
                print("Warning: VGGPerceptualLoss not imported. Using MSE loss.")
                perceptual_loss = nn.MSELoss()

        return cls(
            rgb_encoder=rgb_encoder,
            raw_decoder=raw_decoder,
            flow_model=flow_model,
            cond_unet=cond_unet,
            fusion_module=fusion_module,
            flow_steps=flow_steps,
            use_fusion=use_fusion,
            unfreeze_all=unfreeze_all,
            perceptual_loss=perceptual_loss,
            use_context=use_context,
            use_cond_loss=use_cond_loss
        )

    def rgb_to_raw(self, rgb_input, context=None, coordinates=None):

        rgb_latent, rgb_fea2, rgb_fea4 = self.rgb_encoder(rgb_input)

        pred_raw_latent, cond_output = self.flow_matching.sample_ode(
            shape=rgb_latent.shape,
            rgb_latent=rgb_latent,
            num_steps=self.flow_steps,
            context=context,
            coordinates=coordinates
        )

        if self.use_fusion and self.fusion_module is not None:
            final_latent = self.fusion_module(pred_raw_latent, cond_output)
        else:
            final_latent = pred_raw_latent

        pred_raw_img = self.raw_decoder(final_latent, rgb_fea2, rgb_fea4)

        return pred_raw_img
