import torch
import torch.nn as nn

class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True)
        )

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg + mx)

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        avgc = torch.mean(x, dim=1, keepdim=True)
        maxc,_ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avgc, maxc], dim=1)
        return torch.sigmoid(self.conv(y))

class LIAM(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate(kernel_size)

    def forward(self, x, return_map=False):
        ch_attn = self.channel_gate(x)
        x_ch = x * ch_attn
        spat = self.spatial_gate(x_ch)
        out = x_ch * spat
        if return_map:
            # return spatial map also (useful for visualization)
            return out, spat
        return out
