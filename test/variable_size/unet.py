import torch
import torch.nn as nn
import torch.nn.functional as F
import nestedtensor


class UNet(nn.Module):

    def __init__(self, h=64, mid_block_num=4):
        """
        nestedtensor doesn't support ConvTranspose2d
        """
        super(UNet, self).__init__()
        self.encoder1 = nn.Conv2d(3, h, 5, 2, 2)
        self.encoder2 = nn.Conv2d(h, h, 3, 1, 1)
        self.encoder3 = nn.Conv2d(h, h, 3, 2, 1)
        self.encoder4 = nn.Conv2d(h, h, 3, 1, 1)
        self.encoder5 = nn.Conv2d(h, h, 3, 2, 1)

        self.mid = nn.Sequential(*[nn.Conv2d(h, h, 3, 1, 1) for _ in range(mid_block_num)])

        self.decoder1 = nn.Conv2d(h + h, h, 3, 1, 1)
        self.decoder2 = nn.Conv2d(h + h, h, 3, 1, 1)
        self.decoder3 = nn.Conv2d(h + 3, 1, 3, 1, 1)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            cat = torch.cat
        else:
            cat = nestedtensor.cat
        x1 = x
        x = self.encoder1(x)
        x = self.encoder2(x)
        x2 = x
        x = self.encoder3(x)
        x = self.encoder4(x)
        x3 = x
        x = self.encoder5(x)

        x = self.mid(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder1(cat([x, x3], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder2(cat([x, x2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder3(cat([x, x1], dim=1))

        return x
