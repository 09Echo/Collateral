import math
from functools import partial

import numpy as np
import torch
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import einsum, nn


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a.

    MLP) class.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """foward function"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):
    """Depthwise Separable Convolution with BN module."""

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        """
        foward function
        """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, embed_dim, num_path=4, isPool=False):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=2 if isPool and idx == 0 else 1,
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        """foward function"""
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.

    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        """init function"""
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""

    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MHCABlock(nn.Module):
    """Multi-Head Convolutional self-Attention block."""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            drop_path=0.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        """foward function"""
        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x


class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""

    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            drop_path_list=[],
            qk_scale=None,
            crpe_window={
                3: 2,
                5: 3,
                7: 3
            },
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads,
                                  h=num_heads,
                                  window=crpe_window)
        self.MHCA_layers = nn.ModuleList([
            MHCABlock(
                dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_list[idx],
                qk_scale=qk_scale,
                shared_cpe=self.cpe,
                shared_crpe=self.crpe,
            ) for idx in range(self.num_layers)
        ])

    def forward(self, x, size):
        """foward function"""
        H, W = size
        B = x.shape[0]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(in_features,
                               hidden_features,
                               act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        """foward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class PRM(nn.Module):
    def __init__(self, img_size=512, kernel_size=4, downsample_ratio=1, dilations=[1, 3, 7], in_chans=1, embed_dim=64,
                 share_weights=False, op='cat'):
        super().__init__()
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.share_weights = share_weights
        self.outSize = img_size // downsample_ratio
        self.aggregate = nn.Sequential(
            nn.Conv2d(embed_dim * len(self.dilations), embed_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(embed_dim)
        )

        if share_weights:
            self.convolution = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                                         stride=self.stride, padding=3 * dilations[0] // 2, dilation=dilations[0])

        else:
            self.convs = nn.ModuleList()
            for dilation in self.dilations:
                padding = math.ceil(((self.kernel_size - 1) * dilation + 1 - self.stride) / 2)
                self.convs.append(nn.Sequential(
                    *[nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                                stride=self.stride, padding=padding, dilation=dilation),
                      nn.GELU()]))

        if self.op == 'sum':
            self.out_chans = embed_dim
        elif op == 'cat':
            self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        B, C, W, H = x.shape
        if self.share_weights:
            padding = math.ceil(((self.kernel_size - 1) * self.dilations[0] + 1 - self.stride) / 2)
            y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                                     stride=self.downsample_ratio, padding=padding,
                                     dilation=self.dilations[0]).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                padding = math.ceil(((self.kernel_size - 1) * self.dilations[i] + 1 - self.stride) / 2)
                _y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                                          stride=self.downsample_ratio, padding=padding,
                                          dilation=self.dilations[i]).unsqueeze(dim=-1)
                y = torch.add(y, _y)
        else:
            y = []
            for i in range(len(self.dilations)):
                y.append(self.convs[i](x))
            y = torch.cat(y, dim=1)
            y = self.aggregate(y)
        return y


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Gudiance_forward(nn.Module):
    def __init__(self, embed_dim):
        super(Gudiance_forward, self).__init__()
        self.concat1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, attn, conv):
        x = torch.cat([attn, conv], dim=1)
        x = self.concat1(x)
        return x


class Global_Attention(nn.Module):
    def __init__(self, embed_dim, len, r=4):
        super(Global_Attention, self).__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.hconv = nn.Conv2d(embed_dim, embed_dim, kernel_size=(len, 1), bias=False)
        self.hconv1 = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.wconv = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, len), bias=False)
        self.wconv1 = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.hsoftmax = nn.Softmax(dim=1)
        self.wsoftmax = nn.Softmax(dim=1)
        self.sigmoid1 = nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.hconv(x1)
        x2 = self.hconv1(x2)
        x2 = self.hsoftmax(x2)
        x3 = self.wconv(x1)
        x3 = self.wconv1(x3)
        x3 = self.wsoftmax(x3)
        x4 = torch.matmul(x3, x2)
        x4 = self.sigmoid1(x4)
        x1 = x1 * x4
        x2 = self.mlp(x1)
        x = x + x2
        x = self.sigmoid(x)

        return x


class Local_Attention(nn.Module):
    def __init__(self, embed_dim, nums=3):
        super(Local_Attention, self).__init__()
        layers = [nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=False),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim, bias=False),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim, bias=False),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)]) for i in range(nums)]
        self.layers = nn.ModuleList(layers)
        self.conv = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        for layer in self.layers:
            x1 = layer[0](x1)
            x2 = layer[1](x1) + layer[2](x1) + layer[3](x1) + x1
            x2 = layer[4](x2)
        x1 = self.sigmoid1(x2)
        x1 = torch.mul(x1, x)
        # x = x + x1
        x = self.sigmoid2(x1)

        return x


class Multiscale_Stage(nn.Module):
    def __init__(self, embed_dim, out_embed_dim, num_layers=1, num_heads=8, mlp_ratio=3,
                 drop_path_list=[], len=128):
        super(Multiscale_Stage, self).__init__()
        self.mhca_blks = MHCAEncoder(
            embed_dim,
            num_layers,
            num_heads,
            mlp_ratio,
            drop_path_list=drop_path_list,
        )
        self.InvRes = ResBlock(in_features=embed_dim, out_features=embed_dim)
        self.guide_f = Gudiance_forward(embed_dim)
        # self.ga = Global_Attention(embed_dim, len)
        self.la = Local_Attention(embed_dim)
        self.prm = PRM(in_chans=embed_dim, embed_dim=out_embed_dim, kernel_size=3, downsample_ratio=2)
        self.concat1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, guide):
        _, _, H, W = guide.shape
        x = guide.flatten(2).transpose(1, 2)
        attn = self.mhca_blks(x, (H, W))
        conv = self.InvRes(guide)
        fuse = torch.cat([attn, conv], dim=1)
        fuse = self.concat1(fuse)
        loc = self.la(fuse)
        # glb = self.ga(fuse)
        conv = conv * loc
        # attn = attn * glb
        guide = self.guide_f(attn, conv)
        guide = self.prm(guide)

        return guide


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """Generate drop path rate list following linear decay rule."""
    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


class Cls_head(nn.Module):
    """a linear layer for classification."""

    def __init__(self, embed_dim, num_classes):
        """initialization"""
        super().__init__()

        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """foward function"""
        # (B, C, H, W) -> (B, C, 1)

        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        # Shape : [B, C]
        out = self.cls(x)
        return out


class MPViT(nn.Module):
    """Multi-Path ViT class."""

    def __init__(
            self,
            img_size=128,
            num_stages=4,
            num_layers=[1, 1, 1, 1],
            embed_dims=[64, 128, 256, 512],
            in_dims=[64, 128, 216, 288],
            mlp_ratios=[8, 8, 4, 4],
            num_heads=[8, 8, 8, 8],
            drop_path_rate=0.0,
            in_chans=3,
            num_classes=3,
            downsample_ratio=[2, 2, 2, 2],
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                1,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        # Multi_scale.

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList([
            Multiscale_Stage(
                embed_dims[idx],
                embed_dims[idx + 1]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                num_layers[idx],
                num_heads[idx],
                mlp_ratios[idx],
                drop_path_list=dpr[idx],
                len=128 // (2 ** idx),
            ) for idx in range(self.num_stages)
        ])

        # Classification head.
        self.cls_head = Cls_head(embed_dims[-1], num_classes)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dims[0] * 2, embed_dims[0], 1, 1),
            nn.BatchNorm2d(embed_dims[0])
        )
        self.prm = PRM(kernel_size=3, downsample_ratio=1, in_chans=64, embed_dim=64)
        self.apply(self._init_weights)
        self.origin_conv1 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0], 1, 1, 0),
            nn.BatchNorm2d(embed_dims[0])
        )

        self.flip_conv1 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0], 1, 1, 0),
            nn.BatchNorm2d(embed_dims[0])
        )
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(embed_dims[0] * 2, embed_dims[0], 3, 1, 1),
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.Sigmoid()
        )

        # self.fusion = nn.Conv2d(2, 1, 1, 1, 0)

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        """get classifier function"""
        return self.head

    def forward_features(self, x):
        """forward feature function"""

        # x's shape : [B, C, H, W]
        x = self.stem(x)  # Shape : [B, C, H/4, W/4]
        x_flip = self.flip(x, -1)
        d = x - x_flip
        x1 = self.origin_conv1(x)
        d1 = self.flip_conv1(d)
        f = torch.cat([x1, d1], dim=1)
        d = d + d * self.fusion(f)
        x_new = torch.cat([d, x], dim=1)
        x_new = self.conv(x_new)
        x = self.prm(x_new)
        guide = x
        for idx in range(self.num_stages):
            guide = self.mhca_stages[idx](guide)

        # x = torch.cat([attn, conv, guide], dim=1)
        # x = self.conv(x)
        # x_new_g = self.gl(x_new)
        # x_new = x_new + x_new * x_new_g

        return guide

    def flip(self, x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                     -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)

    def forward(self, x):
        """foward function"""
        x = self.forward_features(x)
        # cls head
        out = self.cls_head(x)
        return out


def _cfg_mpvit(url="", **kwargs):
    """configuration of mpvit."""
    return {
        "url": url,
        "num_classes": 2,
        "input_size": (1, 512, 512),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": 0,
        "std": 1,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


def mpvit_small(**kwargs):
    """mpvit_small :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 6, 3]
    - #channels : [64, 128, 216, 288]
    - MLP_ratio : 4
    Number of params : 22892400
    FLOPs : 4799650824
    Activations : 30601880
    """

    model = MPViT(
        in_chans=1,
        img_size=512,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


if __name__ == "__main__":
    import torch
    from thop import profile

    model = mpvit_small()
    input = torch.randn(1, 1, 512, 512)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)
