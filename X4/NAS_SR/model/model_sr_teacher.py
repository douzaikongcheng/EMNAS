import torch
from torch import nn
import torch.nn.functional as F
from model.genotypes import COMPACT_PRIMITIVES as PRIMITIVES
from model.genotypes import Genotype
from model import common
from model import genotypes
from model.common import default_conv as conv
from model.utils import arch_to_genotype
from collections import namedtuple
from torch.autograd import Variable
from option_main import args
import math


def make_model(args, parent=False):
    return Network(args)


OPS = {
    'rcab': lambda exp, C, stride, affine: RCAB(C, exp, 3, 16, bias=True, bn=True, act=nn.ReLU(True)),
    'w1': lambda exp, C, stride, affine: W1(C, exp, C, 3, None, act=nn.ReLU(True)),
    'w2': lambda exp, C, stride, affine: W2(C, C, exp, 3, None, act=nn.ReLU(True)),
    'w3': lambda exp, C, stride, affine: W3(C, exp, C, 3, None, act=nn.ReLU(True)),
    'w4': lambda exp, C, stride, affine: W4(C, C, exp, 3, None, act=nn.ReLU(True)),
    'resb': lambda exp, C, stride, affine: Resb(C, exp, C, 3, None, act=nn.ReLU(True)),
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
    def __init__(self, n_feats, exp, c2, kernel_size, wn, act):
        super(W1, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, exp[0], 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(exp[0], exp[1], 1, padding=1 // 2))
        body.append(nn.Conv2d(exp[1], n_feats, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res


class W2(nn.Module):
    def __init__(self, n_feats, c1, exp, kernel_size, wn, act):
        super(W2, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, exp[0], kernel_size, padding=kernel_size // 2))
        body.append(nn.Conv2d(exp[0], exp[1], 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(exp[1], n_feats, 1, padding=1 // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res


class W3(nn.Module):
    def __init__(self, n_feats, exp, c2, kernel_size, wn, act):
        super(W3, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, exp[0], 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(exp[0], exp[1], 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(exp[1], n_feats, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res


class W4(nn.Module):
    def __init__(self, n_feats, c1, exp, kernel_size, wn, act):
        super(W4, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, exp[0], kernel_size, padding=kernel_size // 2))
        body.append(act)
        body.append(nn.Conv2d(exp[0], exp[1], 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(exp[1], n_feats, 1, padding=1 // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        # res += x
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
        return x * y


# no att
class RCAB(nn.Module):
    def __init__(
            self, n_feat, exp, kernel_size, reduction, conv=default_conv,
            bias=True, bn=False, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, exp[0], kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(exp[0], n_feat, kernel_size, bias=bias))
        # modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res


class Resb(nn.Module):
    def __init__(self, n_feats, exp, c2, kernel_size, wn, act):
        super(Resb, self).__init__()
        body = []
        body.append(nn.Conv2d(n_feats, exp[0], 1, padding=1 // 2))
        body.append(act)
        body.append(nn.Conv2d(exp[0], exp[1], kernel_size, padding=kernel_size // 2))
        body.append(act)
        body.append(nn.Conv2d(exp[1], n_feats, 1, padding=1 // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        # res += x
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
        return x * y.expand_as(x)


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
        return x * m
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
        return x * y.expand_as(x)






class Cell(nn.Module):

    def __init__(self, genotype, exp, C):
        # exp   : expand channel
        super(Cell, self).__init__()
        print(exp, C)
        op_names, indices = zip(*genotype.normal)
        op_names_att, indices_att = zip(*genotype.reduce)
        concat = genotype.normal_concat
        self._compile(exp, C, op_names, indices, concat, op_names_att)

    def _compile(self, exp, C, op_names, indices, concat, op_names_att):
        name = op_names[0]
        stride = 1
        self.op = OPS[name](exp, C, stride, False)
        name_att = op_names_att[0]
        stride = 1
        self.op_att = OPS[name_att](C, stride, False)

    def forward(self, s0, drop_prob):
        x = self.op(s0)
        x = self.op_att(x)
        return x + s0


class Network(nn.Module):
    """
    stack number:layer of cells and then flatten to fed a linear layer
    """

    def __init__(self, args, conv=common.default_conv):
        super(Network, self).__init__()

        # n_resgroups = args.n_resgroups
        # n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        # reduction = args.reduction
        scale = 2
        act = nn.ReLU(True)

        args.n_resgroups = 8
        args.n_resblocks = 2

        self.total_cell_num = args.n_resgroups * args.n_resblocks
        # self.c = args.init_channels
        self.layers = args.layers
        self.criterion = nn.L1Loss().cuda()
        # self.steps = args.nodes
        self.args = args
        # genotype = eval("genotypes.%s" % args.genotype)
        # geno_channel = eval("genotypes.%s" % args.geno_channel)

        genotype = [Genotype(normal=[('w1', 0)], normal_concat=range(1, 2), reduce=[('ESA', 0)], reduce_concat=range(1, 2))] * self.total_cell_num
        geno_channel = [[32, 8] for _ in range(self.total_cell_num)]

        # for i, chan in enumerate(geno_channel):
        #     for j, ch in enumerate(chan):
        #         t = (args.n_feats + args.offset) / args.search_n_feats
        #         geno_channel[i][j] = int(t * geno_channel[i][j])

        exp = []
        self.G1 = 8
        self.G2 = 4
        for i in range(self.total_cell_num):
            op_names, indices = zip(*genotype[i].normal)
            if op_names[0] == 'rcab':
                ch = geno_channel[i][0]
                exp.append([ch * self.G2])
            elif op_names[0] == 'w1':
                ch1, ch2 = geno_channel[i]
                exp.append([ch1 * self.G1, ch2 * self.G2])
            elif op_names[0] == 'w3':
                ch1, ch2 = geno_channel[i]
                exp.append([ch1 * self.G1, ch2 * self.G2])
            elif op_names[0] == 'wd2':
                ch1, ch2 = geno_channel[i]
                exp.append([ch1 * self.G2, ch2 * self.G1])
            elif op_names[0] == 'w4':
                ch1, ch2 = geno_channel[i]
                exp.append([ch1 * self.G2, ch2 * self.G1])
            elif op_names[0] == 'resb':
                ch1, ch2 = geno_channel[i]
                exp.append([ch1 * self.G2, ch2 * self.G2])

        # print(exp)
        # exit()
        # self.multiplier = args.nodes
        self.total_cell_num = args.n_resblocks * args.n_resgroups
        # c_curr = args.init_channels
        self.cells = nn.ModuleList()

        # cpp, cp, c_curr = c_curr, c_curr, self.c  # 48, 48, 16
        self.cells = nn.ModuleList()
        # print(args.rgb_range)
        # args.rgb_range = 1
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        for i in range(self.total_cell_num):
            self.cells += [Cell(genotype[i], exp[i], args.n_feats)]

        modules_body = []
        # for _ in range(args.n_resgroups):
        #     modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_body.append(conv(n_feats, args.tail_fea, kernel_size))
        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, args.tail_fea, act=False),
            conv(args.tail_fea, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def new(self):
        """
        create a new model and initialize it with current alpha parameters.
        However, its weights are left untouched.
        :return:
        """
        model_new = Network(self.args).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        # x = x.cuda()
        feature_maps = []
        # print(x.shape)
        # exit()
        # print(x) 同
        x = self.sub_mean(x)
        x = self.head(x)
        # print(x)   # 同
        res = x
        for i, cell in enumerate(self.cells):
            # if i % args.n_resgroups == 0:
            if i % 8 == 0:
                tmp = res
                # print(tmp)  # 不同
                # print(args.n_resgroups)  # teacher是 8
            res = cell(res, 0)
            # if (i + 1) % args.n_resgroups == 0:
            if (i + 1) % 8 == 0:
                # res = self.body[(i + 1) // args.n_resgroups - 1](res)
                res += tmp
            if i == 7 or i == 15:
                feature_maps.append(res)
        # print(res)   # 不同
        res += x
        res = self.body[-1](res)

        x = self.tail(res)
        x = self.add_mean(x)
        # print(x[0][0][0:5][0:5])

        # return x
        return [x, feature_maps]
        # return [x, feature_maps]

    def loss(self, x, target):
        """
        :param x:
        :param target:
        :return:
        """
        logits = self(x)
        return self.criterion(logits, target)

    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') == -1:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))
