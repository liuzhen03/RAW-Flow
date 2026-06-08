
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .utils import AverageMeter, calculate_metrics, rgb_to_rggb, rggb_to_rgb

class hard_log_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = (-1 * torch.log(1 - torch.clamp(torch.abs(x - y), 0, 1) + 1e-6)).mean()
        return loss

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()

        sequence = []

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]

        self.model = nn.Sequential(*sequence)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        out = self.model(x) + self.conv(x)
        return out

class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out

class channel_down(nn.Module):
    def __init__(self, channels, down_channels=3):
        super(channel_down, self).__init__()

        self.conv0 = nn.Sequential(Res_block(channels * 4, channels * 2),
                                  Res_block(channels * 2, channels))

        self.conv1 = nn.Conv2d(channels, down_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.conv1(self.conv0(x)))
        return out

class channel_up(nn.Module):
    def __init__(self, channels, down_channels=3):
        super(channel_up, self).__init__()

        self.conv_0 = nn.Conv2d(down_channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Sequential(Res_block(channels, channels * 2),
                                  Res_block(channels * 2, channels * 4))
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv1(self.relu(self.conv_0(x))))
        return out

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        blocks = []
        blocks.append(vgg[:4])
        blocks.append(vgg[4:9])
        blocks.append(vgg[9:18])
        blocks.append(vgg[18:27])
        blocks.append(vgg[27:36])

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] == 4:
            input = rggb_to_rgb(input)
        if target.shape[1] == 4:
            target = rggb_to_rgb(target)

        if input.shape[1] != 3 or target.shape[1] != 3:
            raise ValueError(f"Expected 3 channels after conversion, got {input.shape[1]} and {target.shape[1]}")

        input = (input + 1) / 2
        target = (target + 1) / 2

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        for i, block in enumerate(self.blocks):
            input = block(input)
            target = block(target)
            loss += F.l1_loss(input, target)

        return loss

class RgbEncoder(nn.Module):
    def __init__(self, channels=64):
        super(RgbEncoder, self).__init__()

        self.conv_in = nn.Conv2d(12, channels, kernel_size=(5, 5), stride=(1, 1), padding=2)

        self.block0 = nn.Sequential(Res_block(channels, channels),
                                   Res_block(channels, channels * 2))

        self.down0 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = nn.Sequential(Res_block(channels * 2, channels * 2),
                                   Res_block(channels * 2, channels * 4))

        self.down1 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.channel_down = channel_down(channels, down_channels=6)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        x = F.pixel_unshuffle(x, downscale_factor=2)
        x = self.conv_in(x)
        x0 = self.block0(x)
        level0 = self.down0(x0)
        x1 = self.block1(level0)
        fea4 = self.down1(x1)

        latent = self.channel_down(fea4)

        return latent, level0, fea4

class RgbDecoder(nn.Module):
    def __init__(self, channels=64, down_channels=6):
        super(RgbDecoder, self).__init__()

        self.channel_up = channel_up(channels, down_channels=down_channels)

        self.block_up0 = Res_block(channels * 4, channels * 4)
        self.block_up1 = Res_block(channels * 4, channels * 4)
        self.up_sampling0 = upsampling(channels * 4, channels * 2)
        self.block_up2 = Res_block(channels * 2, channels * 2)
        self.block_up3 = Res_block(channels * 2, channels * 2)
        self.up_sampling1 = upsampling(channels * 2, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.conv_out = nn.Conv2d(channels, 12, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.LeakyReLU()

    def forward(self, latent, fea2, fea4):

        fea_full = self.channel_up(latent)

        recon_fea_up2 = self.up_sampling0(
            self.block_up1(self.block_up0(fea_full) + fea4))
        recon_fea_up4 = self.up_sampling1(
            self.block_up3(self.block_up2(recon_fea_up2) + fea2))

        base_features = self.relu(self.conv2(recon_fea_up4))
        recon_img = torch.tanh(self.conv_out(base_features))
        recon_img = F.pixel_shuffle(recon_img, upscale_factor=2)
        return recon_img

class RawEncoder(nn.Module):
    def __init__(self, channels=64):
        super(RawEncoder, self).__init__()

        self.conv_in = nn.Conv2d(4, channels, kernel_size=(5, 5), stride=(1, 1), padding=2)

        self.block0 = nn.Sequential(Res_block(channels, channels),
                                   Res_block(channels, channels * 2))

        self.down0 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = nn.Sequential(Res_block(channels * 2, channels * 2),
                                   Res_block(channels * 2, channels * 4))

        self.down1 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.channel_down = channel_down(channels, down_channels=6)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        x = self.conv_in(x)
        x0 = self.block0(x)
        level0 = self.down0(x0)
        x1 = self.block1(level0)
        fea4 = self.down1(x1)

        latent = self.channel_down(fea4)

        return latent, level0, fea4

class RawDecoder(nn.Module):
    def __init__(self, channels=64, down_channels=6):
        super(RawDecoder, self).__init__()

        self.channel_up = channel_up(channels, down_channels=down_channels)

        self.block_up0 = Res_block(channels * 4, channels * 4)
        self.block_up1 = Res_block(channels * 4, channels * 4)
        self.up_sampling0 = upsampling(channels * 4, channels * 2)
        self.block_up2 = Res_block(channels * 2, channels * 2)
        self.block_up3 = Res_block(channels * 2, channels * 2)
        self.up_sampling1 = upsampling(channels * 2, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.conv_out = nn.Conv2d(channels, 4, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.LeakyReLU()

    def forward(self, latent, fea2, fea4):

        fea_full = self.channel_up(latent)

        recon_fea_up2 = self.up_sampling0(
            self.block_up1(self.block_up0(fea_full) + fea4))
        recon_fea_up4 = self.up_sampling1(
            self.block_up3(self.block_up2(recon_fea_up2) + fea2))

        base_features = self.relu(self.conv2(recon_fea_up4))
        recon_img = torch.tanh(self.conv_out(base_features))

        return recon_img

class DualDomainLatentAutoencoderRGB(nn.Module):
    def __init__(self, channels=64, down_channels=6):
        super(DualDomainLatentAutoencoderRGB, self).__init__()

        self.encoder = RgbEncoder(channels)
        self.decoder = RgbDecoder(channels, down_channels)

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent, fea2, fea4):
        return self.decoder(latent, fea2, fea4)

    def forward(self, x, target=None):
        output = {}

        latent, fea2, fea4 = self.encode(x)
        output['latent'] = latent
        output['fea2'] = fea2
        output['fea4'] = fea4

        recon_img = self.decode(latent, fea2, fea4)
        output['recon_img'] = recon_img

        if target is not None:
            output['l2_loss'] = self.l2_loss(recon_img, target)
            output['perceptual_loss'] = self.perceptual_loss(recon_img, target)

        return output

class DualDomainLatentAutoencoderRAW(nn.Module):
    def __init__(self, channels=64, down_channels=6):
        super(DualDomainLatentAutoencoderRAW, self).__init__()

        self.encoder = RawEncoder(channels)
        self.decoder = RawDecoder(channels, down_channels)

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.hard_log_criterion = hard_log_loss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent, fea2, fea4):
        return self.decoder(latent, fea2, fea4)

    def forward(self, x, target=None, rgb_features=None, rgb_latent=None):
        output = {}

        latent, raw_fea2, raw_fea4 = self.encode(x)
        output['latent'] = latent
        output['raw_fea2'] = raw_fea2
        output['raw_fea4'] = raw_fea4

        feature_loss = 0.0

        if rgb_features is not None:
            rgb_fea2, rgb_fea4 = rgb_features

            fea2_loss = self.l2_loss(rgb_fea2, raw_fea2)
            fea4_loss = self.l2_loss(rgb_fea4, raw_fea4)

            feature_loss = fea2_loss + fea4_loss

            output['fea2_loss'] = fea2_loss
            output['fea4_loss'] = fea4_loss
            output['feature_loss'] = feature_loss

        if rgb_features is not None:
            rgb_fea2, rgb_fea4 = rgb_features

            recon_img = self.decode(latent, rgb_fea2, rgb_fea4)
            output['use_rgb_features'] = True

        output['recon_img'] = recon_img

        if target is not None:
            output['l2_loss'] = self.l2_loss(recon_img, target)
            output['raw_hard_log_loss'] = self.hard_log_criterion((recon_img + 1) / 2, (target + 1) / 2)

            recon_rgb = rggb_to_rgb(recon_img)
            target_rgb = rggb_to_rgb(target)
            output['perceptual_loss'] = self.perceptual_loss(recon_rgb, target_rgb)

        return output

RgbAE = DualDomainLatentAutoencoderRGB
RawAE = DualDomainLatentAutoencoderRAW
