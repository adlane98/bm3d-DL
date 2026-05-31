import torch
import torch.nn as nn
import torchvision

class ConvDenoiser(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enc1 = self._block(1, 32)
        self.enc2 = self._block(32, 64)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.bottleneck = self._block(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.dec2 = self._block(128, 64) # skip connection de up2 avec enc2

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False)
        self.dec1 = self._block(64, 32) # skip connection de up1 avec enc1

        self.out = nn.Conv2d(32, 1, kernel_size=1)


    @staticmethod
    def _block(in_chan, out_chan):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))

        b = self.bottleneck(self.maxpool(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))
    