import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torch.nn import TransformerEncoderLayer
import math


def bchw2bcl(x):
    b, c, h, w = x.shape
    return x.view(b, c, h * w).contiguous()


def bchw2blc(x):
    b, c, h, w = x.shape
    return x.view(b, c, h * w).permute(0, 2, 1).contiguous()


def bcl2bchw(x):
    b, c, l = x.shape
    h = int(math.sqrt(l))
    w = h
    return x.view(b, c, h, w).contiguous()


def blc2bchw(x):
    b, l, c = x.shape
    h = int(math.sqrt(l))
    w = h
    return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''

    def __init__(self, in_c, init_value=1e-2):
        super().__init__()
        self.inChannels = in_c
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((in_c)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1)
            return scale * x




class DWConv3x3(nn.Module):
    def __init__(self, in_c):
        super(DWConv3x3, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True, groups=in_c)

    def forward(self, x):
        x = self.conv(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_c, out_c, hid_c, ls=1e-2, drop=0.0):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_c)

        self.fc1 = nn.Conv2d(in_c, hid_c, 1)
        self.dwconv = DWConv3x3(hid_c)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_c, out_c, 1)

        self.layer_scale = LayerScale(in_c, init_value=ls)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        shortcut = x.clone()

        # ffn
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)

        # layer scale
        x = self.layer_scale(x)
        x = self.drop(x)

        out = shortcut + x
        return out

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class MSM_AMM(nn.Module):
    """
    LAP
    """

    def __init__(self, dim):
        super(MSM_AMM, self).__init__()

        # aggression local info
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           Conv(dim, dim // 8, kernel_size=1),
                                           nn.ReLU(),
                                           Conv(dim // 8, dim, kernel_size=1),
                                           nn.Sigmoid())


    def forward(self, x):
        shortcut = x.clone()

        x_33 = self.conv0(x)

        x_15 = self.conv0_1(x)
        x_15 = self.conv0_2(x_15)

        x_111 = self.conv1_1(x)
        x_111 = self.conv1_2(x_111)

        add = x_33 + x_15 + x_111

        mixer = self.conv3(add)
        out1 = mixer * shortcut

        channel = self.channel(x)

        out2 = channel * shortcut

        out = out1 + out2

        return out




# class LocalConvAttention(nn.Module):
#     """
#     LAP
#     """
#
#     def __init__(self, dim):
#         super(LocalConvAttention, self).__init__()
#
#         # aggression local info
#         self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
#
#         self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
#         self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
#
#         self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#
#
#     def forward(self, x):
#         shortcut = x.clone()
#
#         x_33 = self.conv0(x)
#
#         x_15 = self.conv0_1(x)
#         x_15 = self.conv0_2(x_15)
#
#         x_111 = self.conv1_1(x)
#         x_111 = self.conv1_2(x_111)
#
#         add = x_33 + x_15 + x_111
#
#         mixer = self.conv3(add)
#         out = mixer * shortcut
#
#
#         return out




class BHAttention(nn.Module):
    """
    GAP
    """

    def __init__(self, dim, h, drop=0.0, sr_ratio=4):
        super(BHAttention, self).__init__()

        # aggression local info
        self.local_embed = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 4, 4, groups=dim // 4),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU6())

        self.maxBatchRelu = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(dim,dim//4,1,1,groups=dim//4),
            nn.BatchNorm2d(dim//4),
            nn.ReLU6())

        self.avgBatchRelu = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Conv2d(dim,dim//4,1,1,groups=dim//4),
            nn.BatchNorm2d(dim//4),
            nn.ReLU6())

        self.dim = dim // 4
        self.real_h = int(h // 4)
        self.window_size = [self.real_h, self.real_h]

        if self.dim <= 64:
            self.num_heads = 2
        else:
            self.num_heads = 4


        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.kv = nn.Linear(self.dim, self.dim * 2, bias=False)
        # self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(drop)

        self.conv_out = nn.Conv2d(self.dim, dim, 1, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):

        # local embed
        #x: torch.Size([1, 384, 32, 32])
        x_q = self.local_embed(x)  # b c h w
        #x: torch.Size([1, 96, 8, 8]
        B, C, H, W = x_q.shape

        x_q = x_q.view(B, C, H * W).permute(0, 2, 1).contiguous() #1 64 96
        B, N, C = x_q.shape
        q = self.q(x_q).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)  #B head N C

        # x = x.permute(0,2,1).reshape(B,C,H,W)


        x_max = self.maxBatchRelu(x)
        x_avg = self.avgBatchRelu(x)
        #x_max shape: torch.Size([1, 384, 8, 8])
        b1, c1, h1, w1 = x_max.shape
        b2, c2, h2, w2 = x_avg.shape
        x_max = x_max.view(b1, c1, h1 * w1).permute(0, 2, 1).contiguous()
        x_avg = x_avg.view(b2, c2, h2 * w2).permute(0, 2, 1).contiguous()
        #x: torch.Size([1, 64, 96])
        # self-attn
        # B_1, N1, C1 = x_max.shape
        B_1, N1, C1 = x_max.shape
        B_2, N2, C2 = x_avg.shape
        kv1 = self.kv(x_max)
        kv2 = self.kv(x_avg)
        kv1 = kv1.reshape(B_1, N1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        kv2 = kv2.reshape(B_2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]
        # q1 = q1 * self.scale
        # q2 = q2 * self.scale
        q = q * self.scale
        attn1 = (q @ k1.transpose(-2, -1))
        attn2 = (q @ k2.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn1 = attn1 + relative_position_bias.unsqueeze(0)
        attn2 = attn2 + relative_position_bias.unsqueeze(0)
        attn1 = self.softmax(attn1)
        attn2 = self.softmax(attn2)
        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B_1, N1, C1).contiguous()
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B_2, N2, C2).contiguous()
        x1 = self.proj(x1)
        x2 = self.proj(x2)
        x1 = self.proj_drop(x1)
        x2 = self.proj_drop(x2)

        x1 = blc2bchw(x1)
        x2 = blc2bchw(x2)
        out = x1 + x2
        out = self.up(self.conv_out(out))

        return out


class BHAFormer(nn.Module):
    """
    DSAFormer block
    """

    # def __init__(self, dim, ls=1e-2, drop=0.0,num_heads=8, sr_ratio=1):
    def __init__(self, dim, h, ls=1e-2, drop=0.0):
        super(BHAFormer, self).__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        # self.detail_attn = LocalConvAttention(dim)
        # self.channel_attn = ChannelAttnBlock(dim, down_ratio, h, heads)
        # self.global_attn = GlobalSelfAttentionV3(dim, num_heads, sr_ratio)
        self.global_attn = BHAttention(dim, h, drop)

        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim, init_value=ls)
        self.drop = nn.Dropout(p=drop)

        hidden_dim = 4 * dim
        self.ffn = FFN(in_c=dim, out_c=dim, hid_c=hidden_dim, ls=ls, drop=drop)

    def forward(self, x):

        shortcut = x.clone()

        # proj1
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)

        # attn
        # xd = self.detail_attn(x)
        xg = self.global_attn(x)
        # attn = xd + xg
        attn = xg

        # proj2
        attn = self.proj2(attn)
        attn = self.layer_scale(attn)
        attn = self.drop(attn)

        attn_out = attn + shortcut

        # ffn
        out = self.ffn(attn_out)
        return out


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        # self.sfs = nn.ModuleList(
        #     [SpatialFormer(dims[0], 128),
        #      SpatialFormer(dims[1], 64),
        #      SpatialFormer(dims[2], 32),
        #      SpatialFormer(dims[3], 16)]
        # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # x =  x + self.sfs[i](x)
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        """
        torch.Size([1, 96, 128, 128])
        torch.Size([1, 192, 64, 64])
        torch.Size([1, 384, 32, 32])
        torch.Size([1, 768, 16, 16])
        """
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth"
}


@register_model  # tiny
def convnext_encoder(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


class LRDU(nn.Module):
    def __init__(self, in_c, factor):
        super(LRDU, self).__init__()

        self.up_factor = factor
        self.factor1 = factor * factor // 2
        self.factor2 = factor * factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1 * in_c, (1, 7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1 * in_c, self.factor2 * in_c, (7, 1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            LRDU(ch_in, 2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x




class Model(nn.Module):
    def __init__(self, n_class=2, pretrained=True):
        super(Model, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        self.Up5 = up_conv(ch_in=768, ch_out=384)
        self.Up_conv5 = conv_block(ch_in=384 * 2, ch_out=384)

        self.Up4 = up_conv(ch_in=384, ch_out=192)
        self.Up_conv4 = conv_block(ch_in=384, ch_out=192)

        self.Up3 = up_conv(ch_in=192, ch_out=96)
        self.Up_conv3 = conv_block(ch_in=192, ch_out=96)

        self.Up4x = LRDU(96, 4)
        self.convout = nn.Conv2d(96, n_class, kernel_size=1, stride=1, padding=0)

        self.decoder = True

        self.ma32= MSM_AMM(384)
        self.ma64= MSM_AMM(192)
        self.ma128= MSM_AMM(96)


        # self.ce1 = SpatialFormer(dim=96,num_heads=2,sr_ratio=8)
        # self.ce2 = SpatialFormer(dim=192,num_heads=4,sr_ratio=4)
        # self.ce3 = SpatialFormer(dim=384,num_heads=8,sr_ratio=2)
        self.bhaformer1 = BHAFormer(96, 128)
        self.bhaformer2 = BHAFormer(192, 64)
        self.bhaformer3 = BHAFormer(384, 32)
        # self.channel_mixer = ChannelAttnBlock(192, down_ratio=16, h=128, heads=2)
        # self.channel_mixer2 = ChannelAttnBlock(384, down_ratio=8, h=64, heads=2)
        # self.channel_mixer3 = ChannelAttnBlock(768, down_ratio=4, h=32, heads=2)


        self.backbone = convnext_encoder(pretrained, True)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)



    def forward(self, x):

        x128, x64, x32, x16 = self.backbone(x)  # x16: torch.Size([1, 768, 16, 16])

        x32 = self.ma32(x32)
        x64 = self.ma64(x64)
        x128 = self.ma128(x128)



        d32 = self.Up5(x16)  # d32_1: torch.Size([1, 384, 32, 32])
        d32 = torch.cat([x32, d32], dim=1)  # d32_2: torch.Size([1, 768, 32, 32])
        d32 = self.Up_conv5(d32)  # d32_3: torch.Size([1, 384, 32, 32])
        d32 = self.bhaformer3(d32) + d32  # d32_4: torch.Size([1, 384, 32, 32])

        d64 = self.Up4(d32)
        d64 = torch.cat([x64, d64], dim=1)
        d64 = self.Up_conv4(d64)
        d64 = self.bhaformer2(d64) + d64

        d128 = self.Up3(d64)
        d128 = torch.cat([x128, d128], dim=1)
        d128 = self.Up_conv3(d128)
        d128 = self.bhaformer1(d128) + d128

        d512 = self.Up4x(d128)
        d512 = self.convout(d512)
        return d512


if __name__ == "__main__":

    model = Model()
    x = torch.ones([1, 3, 512, 512])
    print(model(x).shape)

    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table

        flops = FlopCountAnalysis(model, x)
        print("FLOPs: %.4f G" % (flops.total() / 1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6))

        # FLOPs: 50.7615 G
        # Params: 48.4985 M
