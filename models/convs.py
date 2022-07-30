import torch.nn as nn
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Conv(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 32,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            use_bn: bool = True,
            use_ac: bool = True
    ):
        ...
        super().__init__()
        self.use_bn = use_bn
        self.use_ac = use_ac
        self.c_in = in_channels
        self.c_out = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                              padding=padding).to(DEVICE)
        self.activation = nn.LeakyReLU().to(DEVICE)
        self.batch_norm = nn.BatchNorm2d(out_channels).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.batch_norm(x)
        if self.use_ac:
            x = self.activation(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 c_inside: int = 3,
                 times: int = 1,
                 use_residual: bool = True
                 ):
        ...
        super().__init__()

        self.layer = nn.ModuleList()

        self.use_residual = use_residual

        for i in range(times):
            self.layer.append(
                nn.Sequential(
                    Conv(in_channels=c_inside, out_channels=c_inside // 2, kernel_size=1, stride=1, use_bn=True,
                         use_ac=True),
                    Conv(in_channels=c_inside // 2, out_channels=c_inside, kernel_size=3, stride=1, padding=1,
                         use_bn=True, use_ac=False)
                )
            )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x) + x if self.use_residual else layer(x)
        return x


class ConvPool(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.cv1 = Conv(in_c, in_c, kernel_size=3, stride=1)
        self.cv2 = Conv(in_c, out_c, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return self.pool(self.cv2(self.cv1(x)))


class ConvPoolCM(nn.Module):
    def __init__(self, in_c, out_c, in_c2, out_c2):
        super().__init__()
        self.cv1 = Conv(in_c, out_c, kernel_size=3, stride=1)
        self.cv2 = Conv(in_c2, out_c2, kernel_size=3, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return self.max_pool(self.cv2(self.cv1(x)))


class MiddleFlow(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.cv1 = Conv(in_c, in_c, kernel_size=3, stride=1)
        self.cv2 = Conv(in_c, in_c, kernel_size=3, stride=1)
        self.cv3 = Conv(in_c, in_c, kernel_size=3, stride=1)

    def forward(self, x):
        return self.cv3(self.cv2(self.cv1(x)))


class ConvRoute(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.cv = Conv(in_c, out_c, kernel_size=kernel_size, stride=2)

    def forward(self, x):
        return self.cv(x)
