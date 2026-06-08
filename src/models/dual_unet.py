
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DLFM import DeterministicLatentFlowMatching
from .utils import AverageMeter, calculate_metrics, rgb_to_rggb, rggb_to_rgb

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding

class SPADEGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, normalization_fn, eps=1e-5):
        super().__init__()
        self.norm = normalization_fn(norm_nc)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.eps = eps

    def forward(self, x, cond):
        x = self.norm(x)
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return x * (1 + gamma) + beta

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, num_head_channels=-1, normalization_fn=None):
        super().__init__()
        self.num_heads = num_heads
        self.norm = normalization_fn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        scale = 1 / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum('bhci,bhcj->bhij', q * scale, k * scale)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        out = out.reshape(B, C, H, W)
        return self.proj(out)

class RGBGuidedResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        normalization_fn,
        c_channels,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_norm = SPADEGroupNorm(channels, c_channels, normalization_fn=normalization_fn)
        self.in_layers = nn.Sequential(
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1, padding_mode="reflect"),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_norm = SPADEGroupNorm(self.out_channels, c_channels, normalization_fn=normalization_fn)
        self.out_layers = nn.Sequential(
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, padding_mode="reflect")),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1, padding_mode="reflect")
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, cond, emb):
        h = self.in_norm(x, cond)
        h = self.in_layers(h)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, scale, shift = self.out_norm, emb_out[:, :self.out_channels], emb_out[:, self.out_channels:]
            h = out_norm(h, cond) * (1 + scale) + shift
        else:
            h = self.out_norm(h, cond) + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, self.out_channels, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, channels, self.out_channels, 3, stride=stride, padding=1, padding_mode="reflect")
        else:
            self.op = nn.AvgPool2d(stride)

    def forward(self, x):
        return self.op(x)

class CoordAwareResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        normalization_fn,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = normalization_fn(self.in_channels)
        self.in_layers = nn.Sequential(
            SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, padding_mode="reflect"),
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )

        self.norm2 = normalization_fn(self.out_channels)
        self.out_layers = nn.Sequential(
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, padding_mode="reflect")),
        )

        if self.in_channels != self.out_channels:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb=None):
        h = self.norm1(x)
        h = self.in_layers(h)

        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]

            if self.use_scale_shift_norm:
                scale, shift = emb_out.chunk(2, dim=1)
                h = self.norm2(h) * (1 + scale) + shift
            else:
                h = self.norm2(h) + emb_out
        else:
            h = self.norm2(h)

        h = self.out_layers(h)
        return self.skip_connection(x) + h

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        dropout,
        normalization_fn,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.norm1 = normalization_fn(self.in_channels)
        self.in_layers = nn.Sequential(
            SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, padding_mode="reflect"),
        )
        self.norm2 = normalization_fn(self.out_channels)
        self.out_layers = nn.Sequential(
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, padding_mode="reflect")),
        )
        if self.in_channels != self.out_channels:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = self.in_layers(h)
        h = self.norm2(h)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class CrossScaleContextGuidance(nn.Module):
    def __init__(
        self,
        image_size=64,
        in_channels=6,
        model_channels=64,
        out_channels=6,
        attention_resolutions=(16, 8),
        dropout=0.1,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        norm_num_groups=8,
        use_context=0,
        use_cond_loss=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.use_context = use_context
        self.use_cond_loss = use_cond_loss

        def get_norm_groups(channels, max_groups=32):
            if channels <= max_groups:

                for i in range(min(max_groups, channels), 0, -1):
                    if channels % i == 0:
                        return i
            else:

                for i in range(max_groups, 0, -1):
                    if channels % i == 0:
                        return i

            return 1

        self.normalization_fn = lambda num_channels, **kwargs: nn.GroupNorm(
            get_norm_groups(num_channels, norm_num_groups),
            num_channels,
            **kwargs
        )

        coord_embed_dim = model_channels * 4
        self.coord_embed = nn.Sequential(
            linear(4, model_channels),
            SiLU(),
            linear(model_channels, coord_embed_dim),
        )

        if self.use_context > 0:

            if self.use_context == 128:
                self.context_processor = nn.Sequential(

                    conv_nd(dims, 3, 32, 3, stride=2, padding=1, padding_mode="reflect"),
                    nn.GroupNorm(8, 32),
                    SiLU(),
                    conv_nd(dims, 32, in_channels, 3, padding=1, padding_mode="reflect"),
                    nn.GroupNorm(get_norm_groups(in_channels), in_channels),
                    SiLU(),
                )
            elif self.use_context == 256:
                self.context_processor = nn.Sequential(

                    conv_nd(dims, 3, 32, 3, stride=2, padding=1, padding_mode="reflect"),
                    nn.GroupNorm(8, 32),
                    SiLU(),

                    conv_nd(dims, 32, in_channels, 3, stride=2, padding=1, padding_mode="reflect"),
                    nn.GroupNorm(get_norm_groups(in_channels), in_channels),
                    SiLU(),
                )
            else:
                raise ValueError(f"Unsupported context size: {self.use_context}. Use 0, 128, or 256.")

        self.conv_in = conv_nd(dims, in_channels, model_channels, 3, padding=1, padding_mode="reflect")

        self.down1_res1 = CoordAwareResidualBlock(
            in_channels=model_channels,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.down1_attn = AttentionBlock(
            model_channels,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.down1_res2 = CoordAwareResidualBlock(
            in_channels=model_channels,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.downsample1 = Downsample(
            model_channels,
            use_conv=conv_resample,
            dims=dims,
            out_channels=model_channels * 2,
        )

        self.down2_res1 = CoordAwareResidualBlock(
            in_channels=model_channels * 2,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 2,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.down2_attn = AttentionBlock(
            model_channels * 2,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.down2_res2 = CoordAwareResidualBlock(
            in_channels=model_channels * 2,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 2,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.downsample2 = Downsample(
            model_channels * 2,
            use_conv=conv_resample,
            dims=dims,
            out_channels=model_channels * 4,
        )

        self.down3_res1 = CoordAwareResidualBlock(
            in_channels=model_channels * 4,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.down3_attn = AttentionBlock(
            model_channels * 4,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.down3_res2 = CoordAwareResidualBlock(
            in_channels=model_channels * 4,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )

        self.mid_res1 = CoordAwareResidualBlock(
            in_channels=model_channels * 4,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.mid_attn = AttentionBlock(
            model_channels * 4,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.mid_res2 = CoordAwareResidualBlock(
            in_channels=model_channels * 4,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )

        self.up1_res1 = CoordAwareResidualBlock(
            in_channels=model_channels * 4 + model_channels * 4,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.up1_attn = AttentionBlock(
            model_channels * 4,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.up1_res2 = CoordAwareResidualBlock(
            in_channels=model_channels * 4,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.upsample1 = Upsample(
            model_channels * 4,
            use_conv=conv_resample,
            dims=dims,
            out_channels=model_channels * 2,
        )

        self.up2_res1 = CoordAwareResidualBlock(
            in_channels=model_channels * 2 + model_channels * 2,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 2,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.up2_attn = AttentionBlock(
            model_channels * 2,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.up2_res2 = CoordAwareResidualBlock(
            in_channels=model_channels * 2,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels * 2,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.upsample2 = Upsample(
            model_channels * 2,
            use_conv=conv_resample,
            dims=dims,
            out_channels=model_channels,
        )

        self.up3_res1 = CoordAwareResidualBlock(
            in_channels=model_channels + model_channels,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.up3_attn = AttentionBlock(
            model_channels,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.up3_res2 = CoordAwareResidualBlock(
            in_channels=model_channels,
            emb_channels=coord_embed_dim,
            dropout=dropout,
            normalization_fn=self.normalization_fn,
            out_channels=model_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )

        if self.use_cond_loss:
            self.out = nn.Sequential(
                self.normalization_fn(model_channels),
                SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, padding_mode="reflect")),
            )
        else:
            self.out = None

    def forward(self, x, coordinates=None):

        if self.use_context > 0:

            x = self.context_processor(x)

        coord_emb = None
        if coordinates is not None:
            coord_emb = self.coord_embed(coordinates)

        h = self.conv_in(x)

        h = self.down1_res1(h, coord_emb)
        h = self.down1_attn(h)
        h = self.down1_res2(h, coord_emb)
        cond_feature1 = h
        skip_down1 = h

        h = self.downsample1(h)

        h = self.down2_res1(h, coord_emb)
        h = self.down2_attn(h)
        h = self.down2_res2(h, coord_emb)
        cond_feature2 = h
        skip_down2 = h

        h = self.downsample2(h)

        h = self.down3_res1(h, coord_emb)
        h = self.down3_attn(h)
        h = self.down3_res2(h, coord_emb)
        cond_feature3 = h
        skip_down3 = h

        h = self.mid_res1(h, coord_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, coord_emb)
        cond_feature4 = h

        h = torch.cat([h, skip_down3], dim=1)
        h = self.up1_res1(h, coord_emb)
        h = self.up1_attn(h)
        h = self.up1_res2(h, coord_emb)
        cond_feature5 = h

        h = self.upsample1(h)

        h = torch.cat([h, skip_down2], dim=1)
        h = self.up2_res1(h, coord_emb)
        h = self.up2_attn(h)
        h = self.up2_res2(h, coord_emb)
        cond_feature6 = h

        h = self.upsample2(h)

        h = torch.cat([h, skip_down1], dim=1)
        h = self.up3_res1(h, coord_emb)
        h = self.up3_attn(h)
        h = self.up3_res2(h, coord_emb)
        cond_feature7 = h

        if self.use_cond_loss and self.out is not None:
            output = self.out(h)
        else:

            if x.shape[1] == self.out_channels:
                output = torch.zeros_like(x)
            else:
                output = torch.zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3]), device=x.device)

        return output, (
            cond_feature1,
            cond_feature2,
            cond_feature3,
            cond_feature4,
            cond_feature5,
            cond_feature6,
            cond_feature7,
        )

class RawFlowMatchingModel(nn.Module):
    def __init__(
        self,
        image_size=64,
        in_channels=6,
        model_channels=96,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dims=2,
        cond_channels=None,
        use_checkpoint=False,
        norm_num_groups=8,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.normalization_fn = lambda num_channels, **kwargs: nn.GroupNorm(
            min(norm_num_groups, num_channels), num_channels, **kwargs
        )

        self.cond_channels = cond_channels or [
            model_channels,
            model_channels * 2,
            model_channels * 4,
            model_channels * 4,
            model_channels * 4,
            model_channels * 2,
            model_channels,
        ]
        self.num_cond_features = 7

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_conv = conv_nd(dims, in_channels, model_channels, 3, padding=1, padding_mode="reflect")

        self.input1_res1 = RGBGuidedResidualBlock(
            model_channels,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[0],
            out_channels=model_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.input1_attn = AttentionBlock(
            model_channels,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.input1_res2 = RGBGuidedResidualBlock(
            model_channels,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[0],
            out_channels=model_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.downsample1 = Downsample(
            model_channels,
            conv_resample,
            dims=dims,
            out_channels=model_channels * 2,
        )

        self.input2_res1 = RGBGuidedResidualBlock(
            model_channels * 2,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[1],
            out_channels=model_channels * 2,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.input2_attn = AttentionBlock(
            model_channels * 2,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.input2_res2 = RGBGuidedResidualBlock(
            model_channels * 2,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[1],
            out_channels=model_channels * 2,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.downsample2 = Downsample(
            model_channels * 2,
            conv_resample,
            dims=dims,
            out_channels=model_channels * 4,
        )

        self.input3_res1 = RGBGuidedResidualBlock(
            model_channels * 4,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[2],
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.input3_attn = AttentionBlock(
            model_channels * 4,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.input3_res2 = RGBGuidedResidualBlock(
            model_channels * 4,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[2],
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )

        self.middle_res1 = RGBGuidedResidualBlock(
            model_channels * 4,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[3],
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.middle_attn = AttentionBlock(
            model_channels * 4,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.middle_res2 = RGBGuidedResidualBlock(
            model_channels * 4,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[3],
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )

        self.output1_res1 = RGBGuidedResidualBlock(
            model_channels * 4 + model_channels * 4,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[4],
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.output1_attn = AttentionBlock(
            model_channels * 4,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.output1_res2 = RGBGuidedResidualBlock(
            model_channels * 4,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[4],
            out_channels=model_channels * 4,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.upsample1 = Upsample(
            model_channels * 4,
            conv_resample,
            dims=dims,
            out_channels=model_channels * 2,
        )

        self.output2_res1 = RGBGuidedResidualBlock(
            model_channels * 2 + model_channels * 2,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[5],
            out_channels=model_channels * 2,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.output2_attn = AttentionBlock(
            model_channels * 2,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.output2_res2 = RGBGuidedResidualBlock(
            model_channels * 2,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[5],
            out_channels=model_channels * 2,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.upsample2 = Upsample(
            model_channels * 2,
            conv_resample,
            dims=dims,
            out_channels=model_channels,
        )

        self.output3_res1 = RGBGuidedResidualBlock(
            model_channels + model_channels,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[6],
            out_channels=model_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )
        self.output3_attn = AttentionBlock(
            model_channels,
            num_heads=4,
            num_head_channels=-1,
            normalization_fn=self.normalization_fn,
        )
        self.output3_res2 = RGBGuidedResidualBlock(
            model_channels,
            time_embed_dim,
            dropout,
            self.normalization_fn,
            c_channels=self.cond_channels[6],
            out_channels=model_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
        )

        self.out = nn.Sequential(
            self.normalization_fn(model_channels),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, padding_mode="reflect")),
        )

    def forward(self, x, timesteps, cond_features):
        if len(cond_features) != self.num_cond_features:
            print(f"Warning: Expected {self.num_cond_features} condition features, received {len(cond_features)}")
            cond_features = cond_features + (cond_features[-1],) * (self.num_cond_features - len(cond_features))

        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = self.input_conv(x)
        hs = []

        h = self.input1_res1(h, cond_features[0], time_emb)
        h = self.input1_attn(h)
        h = self.input1_res2(h, cond_features[0], time_emb)
        hs.append(h)

        h = self.downsample1(h)

        h = self.input2_res1(h, cond_features[1], time_emb)
        h = self.input2_attn(h)
        h = self.input2_res2(h, cond_features[1], time_emb)
        hs.append(h)

        h = self.downsample2(h)

        h = self.input3_res1(h, cond_features[2], time_emb)
        h = self.input3_attn(h)
        h = self.input3_res2(h, cond_features[2], time_emb)
        hs.append(h)

        h = self.middle_res1(h, cond_features[3], time_emb)
        h = self.middle_attn(h)
        h = self.middle_res2(h, cond_features[3], time_emb)

        h = torch.cat([h, hs.pop()], dim=1)
        h = self.output1_res1(h, cond_features[4], time_emb)
        h = self.output1_attn(h)
        h = self.output1_res2(h, cond_features[4], time_emb)

        h = self.upsample1(h)

        h = torch.cat([h, hs.pop()], dim=1)
        h = self.output2_res1(h, cond_features[5], time_emb)
        h = self.output2_attn(h)
        h = self.output2_res2(h, cond_features[5], time_emb)

        h = self.upsample2(h)

        h = torch.cat([h, hs.pop()], dim=1)
        h = self.output3_res1(h, cond_features[6], time_emb)
        h = self.output3_attn(h)
        h = self.output3_res2(h, cond_features[6], time_emb)

        return self.out(h)
