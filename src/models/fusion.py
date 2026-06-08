
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AverageMeter, calculate_metrics, rgb_to_rggb, rggb_to_rgb

class CrossAttention(nn.Module):
    def __init__(self, in_channels, cond_channels, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5

        self.to_q = nn.Linear(in_channels, in_channels, bias=False)
        self.to_k = nn.Linear(cond_channels, in_channels, bias=False)
        self.to_v = nn.Linear(cond_channels, in_channels, bias=False)
        self.to_out = nn.Linear(in_channels, in_channels)

    def forward(self, x, cond):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)

        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=(h, w), mode='bilinear', align_corners=False)
        cond_flat = cond.view(b, cond.shape[1], h * w).transpose(1, 2)

        q = self.to_q(x_flat)
        k = self.to_k(cond_flat)
        v = self.to_v(cond_flat)

        q = q.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = self.to_out(out)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out

class FusionModule(nn.Module):
    def __init__(self, in_channels, cond_channels, n_feat=64, num_blocks=3):
        super(FusionModule, self).__init__()
        self.conv_in_flow = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1)
        self.conv_in_cond = nn.Conv2d(cond_channels, n_feat, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(Res_block(n_feat, n_feat))
            self.cross_attns.append(CrossAttention(n_feat, n_feat))

        self.conv_out = nn.Conv2d(n_feat, in_channels, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Res_block):
                 for sub_m in m.modules():
                    if isinstance(sub_m, (nn.Conv2d, nn.Linear)):
                        nn.init.zeros_(sub_m.weight)
                        if sub_m.bias is not None:
                            nn.init.zeros_(sub_m.bias)

    def forward(self, flow_latent, cond_latent):

        flow_feat = self.conv_in_flow(flow_latent)
        cond_feat = self.conv_in_cond(cond_latent)

        out = flow_feat
        for i in range(len(self.res_blocks)):
            out_res = self.res_blocks[i](out)
            out_attn = self.cross_attns[i](out_res, cond_feat)
            out = out_res + out_attn

        out = self.conv_out(out)
        return out + flow_latent

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.act(out)
        return out
