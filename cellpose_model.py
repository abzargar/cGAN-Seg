import torch
import torch.nn as nn
import torch.nn.functional as F


def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d((2, 2))
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.add_module('res_down_%d' % n, resdown(nbase[n], nbase[n + 1], sz))
            else:
                self.down.add_module('conv_down_%d' % n, convdown(nbase[n], nbase[n + 1], sz))

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        self.conv = batchconv(in_channels, out_channels, sz)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x):
        feat = self.full(style)
        y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
                                                      concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y)
        x = x + self.conv[3](style, self.conv[2](style, x))
        return x


class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
                                                      concatenation=concatenation))

    def forward(self, x, y, style):
        x = self.conv[1](style, self.conv[0](x) + y)
        return x


class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** .5

        return style


class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d' % (n - 1),
                                   resup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))
            else:
                self.up.add_module('conv_up_%d' % (n - 1),
                                   convup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))

    def forward(self, style, xd):
        x = self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up) - 2, -1, -1):
            x = self.upsampling(x)
            x = self.up[n](x, xd[n], style)
        return x

class Cellpose_CPnet(nn.Module):
    def __init__(self, n_channels=3,n_classes=2, kernel_size=3, residual_on=True,
                 style_on=True, concatenation=False):
        super(Cellpose_CPnet, self).__init__()
        self.nbase = [n_channels, 64, 128, 256]
        self.n_classes=n_classes
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.downsample = downsample(self.nbase, kernel_size, residual_on=residual_on)
        nbaseup = self.nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, kernel_size, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], n_classes, 1)
        self.style_on = style_on

    def forward(self, data):
        T0 = self.downsample(data)
        style = self.make_style(T0[-1])
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0)
        logits = self.output(T0)
        return logits

