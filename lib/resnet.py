import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BottleNeckBase(nn.Module):
    """Base class for IR bottlenecks (1D and 2D)."""
    def __init__(self, shortcut_layer=None):
        super().__init__()
        self.shortcut_layer = shortcut_layer

    def forward(self, x):
        shortcut = x if self.shortcut_layer is None else self.shortcut_layer(x)
        return shortcut + self.res_layer(x)

class BottleNeck1D_IR(BottleNeckBase):
    """Improved Residual Bottleneck for 1D convolutions."""
    def __init__(self, in_ch, out_ch, stride, dim_match):
        shortcut = None if dim_match else nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        super().__init__(shortcut_layer=shortcut)
        self.res_layer = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.Conv1d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv1d(out_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        )

class BottleNeck_IR(BottleNeckBase):
    """Improved Residual Bottleneck for 2D convolutions."""
    def __init__(self, in_ch, out_ch, stride, dim_match):
        shortcut = None if dim_match else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        super().__init__(shortcut_layer=shortcut)
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

class ResNet(nn.Module):
    """Custom ResNet using IR Bottlenecks."""
    def __init__(self, block, layers):
        super().__init__()
        self.layer1 = self._make_layer(block, 64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.out_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.LayerNorm(512)
        )

        self._init_weights()

    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride):
        layers = [block(in_ch, out_ch, stride, dim_match=False)]
        layers += [block(out_ch, out_ch, stride=1, dim_match=True) for _ in range(1, num_blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.out_layer(x)  # Output shape: (B, 512)
