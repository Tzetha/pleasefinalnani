from torch import nn
import torch
from resnet import BottleNeck1D_IR, BottleNeck_IR

class Audio_front(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels

        self.frontend = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.PReLU(256)
        )

        self.res_block = nn.Sequential(
            BottleNeck_IR(256, 256, stride=1, dim_match=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.4),
        )

        self.linear = nn.Sequential(
            nn.Linear(256 * 20, 512),
            nn.LayerNorm(512)
        )

    def forward(self, x):
        """
        x shape: (B, 1, 80, T) - Spectrogram
        """
        B, C, F, T = x.shape  # (Batch, 1, 80, T)

        # Ensure T is divisible by 4 for downsampling
        pad_T = (4 - (T % 4)) % 4  # Compute required padding
        x = nn.functional.pad(x, (0, pad_T))  # Pad only on the time dimension
        T_padded = x.shape[-1]  # Updated T

        x = self.frontend(x)  # (B, 256, F/4, T/4)
        x = self.Res_block(x)  # (B, 256, F/4, T/4)
        B, C, F, T = x.size()

        x = x.view(B, C * F, T).transpose(1, 2).contiguous()  # (B, T, 512 * F/4)
        x = x.view(B * T, -1)
        x = self.Linear(x)  # (B, T, 512)
        x = x.view(B, T, -1)

        return x
