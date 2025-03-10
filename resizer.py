import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)


class Resizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.interpolate_mode = 'bilinear'
        self.scale_factor = 2   # cfg.data.image_size / cfg.data.resizer_image_size

        n = 16 # cfg.resizer.num_kernels
        r = 2 # cfg.resizer.num_resblocks
        slope = 0.2 # cfg.resizer.negative_slope

        self.module1 = nn.Sequential(
            nn.Conv2d(3, n, kernel_size=7, padding=3),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(n, n, kernel_size=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.BatchNorm2d(n)
        )

        resblocks = []
        for i in range(r):
            resblocks.append(ResBlock(n, slope))
        self.resblocks = nn.Sequential(*resblocks)

        self.module3 = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n)
        )

        self.module4 = nn.Conv2d(n, 3, kernel_size=7,
                                 padding=3)

        self.interpolate = partial(F.interpolate,
                                   size= (480, 640),
                                   # scale_factor=self.scale_factor,
                                   mode=self.interpolate_mode,
                                   align_corners=False,
                                   recompute_scale_factor=False)

    def forward(self, x):
        residual = self.interpolate(x)

        out = self.module1(x)
        out_residual = self.interpolate(out)

        out = self.resblocks(out_residual)
        out = self.module3(out)
        out = out + out_residual

        out = self.module4(out)

        out = out + residual

        return out

