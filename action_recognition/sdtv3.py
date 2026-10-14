"""SDTv3 / E-SpikeFormer 19M architecture used by the HARDVS experiments."""
import torch
from torch import nn
from timm.models.layers import trunc_normal_, DropPath
from oilif import OILIF

class SepConv_Spike(nn.Module):
    """
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2, act2_layer=nn.Identity, bias=False, kernel_size=7, padding=3, D_Norm=8):
        super().__init__()
        self.D_Norm = D_Norm
        med_channels = int(expansion_ratio * dim)
        self.spike1 = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.pwconv1 = nn.Sequential(nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias), nn.BatchNorm2d(med_channels))
        self.spike2 = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.dwconv = nn.Sequential(nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias), nn.BatchNorm2d(med_channels))
        self.spike3 = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.pwconv2 = nn.Sequential(nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias), nn.BatchNorm2d(dim))

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.spike1(x) / self.D_Norm
        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.spike2(x) / self.D_Norm
        x = self.dwconv(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.spike3(x) / self.D_Norm
        x = self.pwconv2(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x

class MS_ConvBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, D_Norm=8):
        super().__init__()
        self.Conv = SepConv_Spike(dim=dim, D_Norm=D_Norm)
        self.mlp_ratio = mlp_ratio
        self.D_Norm = D_Norm
        self.spike1 = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.conv1 = nn.Conv2d(dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.spike2 = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.conv2 = nn.Conv2d(dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x) / self.D_Norm
        x = self.bn1(self.conv1(x.flatten(0, 1))).reshape(T, B, self.mlp_ratio * C, H, W)
        x = self.spike2(x) / self.D_Norm
        x = self.bn2(self.conv2(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x
        return x

class MS_MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0, D_Norm=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.D_Norm = D_Norm
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_spike(x) / self.D_Norm
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x) / self.D_Norm
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        return x

class MS_Attention_linear(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1, D_Norm=8, lamda_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.D_Norm = D_Norm
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** (-0.5)
        self.lamda_ratio = lamda_ratio
        self.head_spike = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))
        self.q_spike = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))
        self.k_spike = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False), nn.BatchNorm2d(int(dim * lamda_ratio)))
        self.v_spike = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.attn_spike = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.proj_conv = nn.Sequential(nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)
        x = self.head_spike(x) / self.D_Norm
        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C_v, H, W)
        q = self.q_spike(q)
        q = q.flatten(3)
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        k = self.k_spike(k)
        k = k.flatten(3)
        k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        v = self.v_spike(v)
        v = v.flatten(3)
        v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, C_v // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        x = q @ k.transpose(-2, -1)
        x = x @ v * (self.scale * 2)
        x = x.transpose(3, 4).reshape(T, B, C_v, N).contiguous()
        x = self.attn_spike(x) / self.D_Norm
        x = x.reshape(T, B, C_v, H, W)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x

class MS_Block_Spike_SepConv(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, sr_ratio=1, init_values=1e-06, D_Norm=8):
        super().__init__()
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1, D_Norm=D_Norm)
        self.attn = MS_Attention_linear(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, lamda_ratio=4, D_Norm=D_Norm)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, D_Norm=D_Norm)
        self.layer_scale1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.layer_scale2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.layer_scale3 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.conv(x) * self.layer_scale1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.attn(x) * self.layer_scale2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.mlp(x) * self.layer_scale3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x

class MS_DownSampling(nn.Module):

    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True, D_Norm=8):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.D_Norm = D_Norm
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = OILIF(tau=None, step_mode="single", detach_reset=False)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, 'encode_spike'):
            x = self.encode_spike(x) / self.D_Norm
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)
        return x

class SDTv3Backbone(nn.Module):

    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11, embed_dim=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[6, 8, 6], sr_ratios=[8, 4, 2], D_Norm=8):
        super().__init__()
        D_Norm = 1
        self.num_classes = num_classes
        self.depths = depths
        self.D_Norm = D_Norm
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.downsample1_1 = MS_DownSampling(in_channels=in_channels, embed_dims=embed_dim[0] // 2, kernel_size=7, stride=2, padding=3, first_layer=True, D_Norm=self.D_Norm)
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios, D_Norm=self.D_Norm)])
        self.downsample1_2 = MS_DownSampling(in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0], kernel_size=3, stride=2, padding=1, first_layer=False, D_Norm=self.D_Norm)
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios, D_Norm=self.D_Norm)])
        self.downsample2 = MS_DownSampling(in_channels=embed_dim[0], embed_dims=embed_dim[1], kernel_size=3, stride=2, padding=1, first_layer=False, D_Norm=self.D_Norm)
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios, D_Norm=self.D_Norm)])
        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios, D_Norm=self.D_Norm)])
        self.downsample3 = MS_DownSampling(in_channels=embed_dim[1], embed_dims=embed_dim[2], kernel_size=3, stride=2, padding=1, first_layer=False, D_Norm=self.D_Norm)
        self.block3 = nn.ModuleList([MS_Block_Spike_SepConv(dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer, sr_ratio=sr_ratios, D_Norm=self.D_Norm) for j in range(int(depths * 0.75))])
        self.downsample4 = MS_DownSampling(in_channels=embed_dim[2], embed_dims=embed_dim[3], kernel_size=3, stride=1, padding=1, first_layer=False, D_Norm=self.D_Norm)
        self.block4 = nn.ModuleList([MS_Block_Spike_SepConv(dim=embed_dim[3], num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer, sr_ratio=sr_ratios, D_Norm=self.D_Norm) for j in range(int(depths * 0.25))])
        self.head = nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        self.spike = OILIF(tau=None, step_mode="single", detach_reset=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)
        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)
        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        return x

    def forward(self, x):
        x = x.transpose(0, 1).contiguous()
        x = self.forward_features(x)
        x = x.flatten(3).mean(3)
        x = self.spike(x)
        x = self.head(x).mean(0)
        return x

def SDTv3(num_classes=300, drop_path=0.1):
    return SDTv3Backbone(embed_dim=[64,128,256,360], num_heads=8, mlp_ratios=4, in_channels=2, num_classes=num_classes, depths=8, sr_ratios=1, D_Norm=1, drop_path_rate=drop_path)
