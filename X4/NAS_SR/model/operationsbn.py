import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

sys.path.append("..")
from option import args


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


OPS = {
    'rcab': lambda C, stride, affine: RCAB(C, C, 3, 16, bias=True, bn=True, act=nn.ReLU(True)),
    'w1': lambda C, stride, affine: W1(C, C * 6, C, 3, None, act=nn.ReLU(True)),
    'w2': lambda C, stride, affine: W2(C, C, C * 6, 3, None, act=nn.ReLU(True)),
    'w3': lambda C, stride, affine: W3(C, C * 6, C, 3, None, act=nn.ReLU(True)),
    'w4': lambda C, stride, affine: W4(C, C, C * 6, 3, None, act=nn.ReLU(True)),
    'resb': lambda C, stride, affine: Resb(C, C, C, 3, None, act=nn.ReLU(True)),
    'SE_Block': lambda C, stride, affine: SE_Block(C),
    'CALayer': lambda C, stride, affine: CALayer(C),
    'ESA': lambda C, stride, affine: ESA(C),
    'eca_layer': lambda C, stride, affine: eca_layer(C),
    'skip_connect': lambda C, stride, affine: Identity(),
}


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class W1(nn.Module):
    def __init__(self, n_feats, c1, c2, kernel_size, wn, act):
        super(W1, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, c1, 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(c1, c2, 1, padding=1 // 2))
        body.append(nn.Conv2d(c2, n_feats, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class W2(nn.Module):
    def __init__(self, n_feats, c1, c2, kernel_size, wn, act):
        super(W2, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, c1, kernel_size, padding=kernel_size // 2))
        body.append(nn.Conv2d(c1, c2, 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(c2, n_feats, 1, padding=1 // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class W3(nn.Module):
    def __init__(self, n_feats, c1, c2, kernel_size, wn, act):
        super(W3, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, c1, 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(c1, c2, 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(c2, n_feats, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class W4(nn.Module):
    def __init__(self, n_feats, c1, c2, kernel_size, wn, act):
        super(W4, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, c1, kernel_size, padding=kernel_size // 2))
        body.append(act)
        body.append(nn.Conv2d(c1, c2, 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(c2, n_feats, 1, padding=1 // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y.squeeze()


# no att
class RCAB(nn.Module):
    def __init__(
            self, n_feat, c1, kernel_size, reduction, conv=default_conv,
            bias=True, bn=False, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, c1, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(c1, n_feat, kernel_size, bias=bias))
        # modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Resb(nn.Module):
    def __init__(self, n_feats, c1, c2, kernel_size, wn, act):
        super(Resb, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, c1, 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(c1, c2, kernel_size, padding=kernel_size // 2))
        body.append(act)
        body.append(nn.Conv2d(c2, n_feats, 1, padding=1 // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# attention op
# attention op
# attention op
# attention op
# attention op
# attention op
# attention op
# attention op
# attention op
# attention op
# attention op
# attention op
# attention op
# attention op


# SE_Block
class SE_Block(nn.Module):
    def __init__(self, n_feats, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(n_feats, 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4, n_feats, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x), y.squeeze()


class ESA(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        # print(m)
        # print(m.shape)
        # print(x.shape)
        # exit()
        # print()
        # print(nn.AdaptiveAvgPool2d(1)(m).shape)
        # exit()
        return x * m, nn.AdaptiveAvgPool2d(1)(m).squeeze()
        # return x * m


# ECA模块
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()  # super类的作用是继承的时候，调用含super的哥哥的基类__init__函数。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()  # b代表b个样本，c为通道数，h为高度，w为宽度

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)
        # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算
        return x * y.expand_as(x), y.squeeze()









