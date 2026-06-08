
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from .DLAE import DualDomainLatentAutoencoderRAW, DualDomainLatentAutoencoderRGB
from .DLFM import DeterministicLatentFlowMatching
from .dual_unet import CrossScaleContextGuidance, RawFlowMatchingModel
from .fusion import FusionModule
from .rawflow import RawFlowModel
from .utils import rgb_to_rggb, rggb_to_rgb

class RawFlowWrapper(nn.Module):

    def __init__(self, model_path, camera="NIKON_D700", use_fusion=True,
                 context_size=256, patch_size=256, overlap=32, device=None):
        super(RawFlowWrapper, self).__init__()

        self.model_path = model_path
        self.camera = camera
        self.use_fusion = use_fusion
        self.context_size = context_size
        self.patch_size = patch_size
        self.overlap = overlap

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"RawFlowWrapper: camera={camera}, ckpt={model_path}, device={self.device}, fusion={use_fusion}")

        self.model = self._create_model()
        self.model.eval()

    def _create_model(self):
        model_path = self.model_path

        channels = 96
        down_channels = 6
        rgb_ae = DualDomainLatentAutoencoderRGB(channels=channels, down_channels=down_channels)
        raw_ae = DualDomainLatentAutoencoderRAW(channels=channels, down_channels=down_channels)
        use_context=256

        image_size = 64
        in_channels = down_channels
        model_channels = 96
        out_channels = down_channels
        channel_mult = [1, 2, 4]

        cond_channels = [48, 96, 192, 192, 192, 96, 48]

        flow_model = RawFlowMatchingModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            dropout=0.1,
            channel_mult=channel_mult,
            conv_resample=True,
            dims=2,
            cond_channels=cond_channels,
            use_checkpoint=False,
            norm_num_groups=8
        )

        cond_unet = CrossScaleContextGuidance(
        image_size=64,
        in_channels=6,
        model_channels=48,
        out_channels=6,
        use_context=use_context,
        use_cond_loss=False,

        )

        rgb_ae = rgb_ae.to(self.device)
        raw_ae = raw_ae.to(self.device)
        flow_model = flow_model.to(self.device)
        cond_unet = cond_unet.to(self.device)

        fusion_module = None
        if self.use_fusion:
            fusion_module = FusionModule(
                in_channels=6,
                cond_channels=6,
                n_feat=128,
                num_blocks=4,
            ).to(self.device)
            for _, module in fusion_module.named_modules():
                if hasattr(module, 'to'):
                    module.to(self.device)

        self.model = RawFlowModel.from_auto_encoders(
            rgb_ae=rgb_ae,
            raw_ae=raw_ae,
            flow_model=flow_model,
            cond_unet=cond_unet,
            fusion_module=fusion_module,
            flow_steps=20,
            use_fusion=self.use_fusion,
            unfreeze_all=False,

            use_context=use_context > 0,
            use_cond_loss=False
        )
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:

                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise

        self.model = self.model.to(self.device)
        return self.model

    def _split_image_into_patches(self, image, patch_size, overlap):
        C, H, W = image.shape
        stride = patch_size - overlap

        raw_patch_size = patch_size // 2
        raw_stride = stride // 2

        patches = []
        positions = []

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):

                if y + patch_size > H:
                    y = H - patch_size
                if x + patch_size > W:
                    x = W - patch_size

                patch = image[:, y:y + patch_size, x:x + patch_size].clone()
                patches.append(patch)

                raw_x = x // 2
                raw_y = y // 2
                positions.append((raw_x, raw_y))

        if (W - patch_size) % stride != 0:
            x = W - patch_size
            for y in range(0, H - patch_size + 1, stride):
                if y + patch_size > H:
                    y = H - patch_size
                patch = image[:, y:y + patch_size, x:x + patch_size].clone()
                patches.append(patch)
                raw_x = x // 2
                raw_y = y // 2
                positions.append((raw_x, raw_y))

        if (H - patch_size) % stride != 0:
            y = H - patch_size
            for x in range(0, W - patch_size + 1, stride):
                if x + patch_size > W:
                    x = W - patch_size
                patch = image[:, y:y + patch_size, x:x + patch_size].clone()
                patches.append(patch)
                raw_x = x // 2
                raw_y = y // 2
                positions.append((raw_x, raw_y))

        if (W - patch_size) % stride != 0 and (H - patch_size) % stride != 0:
            x = W - patch_size
            y = H - patch_size
            patch = image[:, y:y + patch_size, x:x + patch_size].clone()
            patches.append(patch)
            raw_x = x // 2
            raw_y = y // 2
            positions.append((raw_x, raw_y))

        return torch.stack(patches) if patches else torch.empty(0), positions

    def _stitch_patches(self, patches, positions, original_shape):
        C, H, W = original_shape
        stitched = torch.zeros((C, H, W), dtype=patches.dtype, device=patches.device)

        for i, ((x, y), patch) in enumerate(zip(positions, patches)):
            patch_h, patch_w = patch.shape[1], patch.shape[2]

            valid_x = min(patch_w, W - x)
            valid_y = min(patch_h, H - y)

            stitched[:, y:y+valid_y, x:x+valid_x] = patch[:, :valid_y, :valid_x]

        return stitched

    def _resize_context_rgb(self, rgb_img):

        if not isinstance(rgb_img, torch.Tensor):
            rgb_img = torch.tensor(rgb_img, dtype=torch.float32)

        if len(rgb_img.shape) == 4 and rgb_img.shape[0] == 1:
            rgb_img = rgb_img.squeeze(0)

        context_size = self.context_size
        context_rgb = F.interpolate(
            rgb_img.unsqueeze(0),
            size=(context_size, context_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return context_rgb

    def _process_patches(self, rgb_patches, context_rgb, batch_size=4):
        n_patches = rgb_patches.size(0)
        raw_patches = []

        if hasattr(self.model.flow_matching, 'cond_unet'):
            self.model.flow_matching.cond_unet = self.model.flow_matching.cond_unet.to(self.device)

        with torch.no_grad():
            for i in range(0, n_patches, batch_size):
                current_batch_size = min(batch_size, n_patches - i)
                batch_rgb = rgb_patches[i:i + current_batch_size].to(self.device)
                batch_context = context_rgb.unsqueeze(0).expand(current_batch_size, -1, -1, -1).to(self.device)
                model_output = self._custom_rgb_to_raw(batch_rgb, batch_context)
                raw_patches.append(model_output)

        if raw_patches:
            raw_patches = torch.cat(raw_patches, dim=0)
        else:
            raw_patches = torch.empty((0, 4, self.patch_size, self.patch_size), device=self.device)

        return raw_patches

    def _custom_rgb_to_raw(self, rgb_input, global_img):

        rgb_input = rgb_input.to(self.device)
        global_img = global_img.to(self.device)

        rgb_latent, rgb_fea2, rgb_fea4 = self.model.rgb_encoder(rgb_input)

        cond_unet = self.model.flow_matching.cond_unet

        if hasattr(self.model.flow_matching, 'sample_ode'):

            pred_raw_latent, cond_output = self.model.flow_matching.sample_ode(
                shape=rgb_latent.shape,
                rgb_latent=rgb_latent,
                context=global_img,
                num_steps=self.model.flow_steps
            )
        else:

            print("Warning: Skipping flow matching, using conditional UNet output only")
            cond_output, _ = cond_unet(global_img)
            pred_raw_latent = cond_output

        if self.model.use_fusion and self.model.fusion_module is not None:
            final_latent = self.model.fusion_module(pred_raw_latent, cond_output)
        else:
            final_latent = pred_raw_latent

        pred_raw_img = self.model.raw_decoder(final_latent, rgb_fea2, rgb_fea4)

        return pred_raw_img

    def normalize_input(self, x):
        return x * 2.0 - 1.0

    def denormalize_output(self, x):
        return (x + 1.0) / 2.0

    def forward(self, rgb_img):
        if rgb_img.dim() == 4 and rgb_img.shape[0] == 1:
            rgb_img = rgb_img.squeeze(0)
        if rgb_img.device != self.device:
            rgb_img = rgb_img.to(self.device)

        context_rgb = self._resize_context_rgb(rgb_img)
        rgb_patches, positions = self._split_image_into_patches(rgb_img, self.patch_size, self.overlap)

        rgb_patches_norm = self.normalize_input(rgb_patches)
        context_rgb_norm = self.normalize_input(context_rgb)
        raw_patches_norm = self._process_patches(rgb_patches_norm, context_rgb_norm)
        raw_patches = self.denormalize_output(raw_patches_norm)

        raw_output_shape = (4, rgb_img.shape[1] // 2, rgb_img.shape[2] // 2)
        return self._stitch_patches(raw_patches, positions, raw_output_shape)
