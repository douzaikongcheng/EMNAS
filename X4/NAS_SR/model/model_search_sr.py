import torch
from torch import nn
import torch.nn.functional as F
from model.operationsbn_search import OPS
from model.genotypes import COMPACT_PRIMITIVES as PRIMITIVES
from model.genotypes import Genotype, ATT_PRIMITIVES
from model import common
from model.common import default_conv as conv
from model.utils import arch_to_genotype
from collections import namedtuple
from torch.autograd import Variable
from option import args
from model.operationsbn_search import Conv2dWp8, Conv2dWp4
import copy
import numpy as np


def make_model(args, parent=False):
    return Network(args)


class MixedLayer(nn.Module):
    def __init__(self, c, stride):
        super(MixedLayer, self).__init__()
        self.layers = nn.ModuleList()
        for primitive in PRIMITIVES:
            layer = OPS[primitive](c, stride, False)
            self.layers.append(layer)
        self.sample = False

    def forward(self, x, weights):
        if self.sample:
            theta = []
            max_idx = torch.argmax(weights).item()
            for i, layer in enumerate(self.layers):
                if i == max_idx:
                    for m in layer.modules():
                        if isinstance(m, Conv2dWp4) or isinstance(m, Conv2dWp8):
                            theta.append(m.theta)
            weights_ = copy.deepcopy(weights[0].data)
            k = np.random.randint(1, len(self.layers))
            values, indices = weights_.topk(k, dim=0, largest=True, sorted=True)
            return sum(weights[0][idx] * self.layers[idx](x) for idx in indices), theta
        else:
            theta = []
            max_idx = torch.argmax(weights).item()
            for i, layer in enumerate(self.layers):
                if i == max_idx:
                    for m in layer.modules():
                        if isinstance(m, Conv2dWp4) or isinstance(m, Conv2dWp8):
                            theta.append(m.theta)
            return sum([w * layer(x) for w, layer in zip(weights[0], self.layers)]), theta


class MixedAtt(nn.Module):
    def __init__(self, c, stride):
        super(MixedAtt, self).__init__()
        self.layers = nn.ModuleList()
        for primitive in ATT_PRIMITIVES:
            layer = OPS[primitive](c, stride, False)
            self.layers.append(layer)
        self.sample = False

    def forward(self, x, weights):
        if self.sample:
            weights_ = copy.deepcopy(weights[0].data)
            k = np.random.randint(1, len(self.layers))
            values, indices = weights_.topk(k, dim=0, largest=True, sorted=True)
            return sum(weights[0][idx] * self.layers[idx](x) for idx in indices)
        else:
            y = sum([w * layer(x)[0] for w, layer in zip(weights[0], self.layers)])
            return y


class Cell(nn.Module):
    def __init__(self, steps, multiplier, cpp, cp, c, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.steps = steps
        self.multiplier = multiplier
        stride = 1
        self.layer = MixedLayer(c, stride)
        self.att = MixedAtt(c, stride)

    def forward(self, s0, weights, weights1):
        x, theta = self.layer(s0, weights)
        x = self.att(x, weights1)
        return x + s0, theta


class Network(nn.Module):
    def __init__(self, args, multiplier=1, conv=common.default_conv):
        super(Network, self).__init__()
        self.multiplier = args.nodes
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.c = args.init_channels
        self.layers = args.layers
        self.criterion = nn.L1Loss().cuda()
        self.steps = args.nodes
        self.args = args

        self.cells = nn.ModuleList()

        k = 1
        self.num_ops = len(PRIMITIVES)
        self.num_att_ops = len(ATT_PRIMITIVES)
        self.cell_idx = 0
        self.total_cell_num = args.n_resblocks * args.n_resgroups

        self.arch_alpha_normal = [nn.Parameter(torch.randn(k, self.num_ops), requires_grad=True) for _ in range(self.total_cell_num)]
        self.arch_beta_normal = [nn.Parameter(torch.randn(k, self.num_att_ops), requires_grad=True) for _ in range(self.total_cell_num)]
        # self.arch_gamma_normal = [nn.Parameter(torch.zeros(n_feats), requires_grad=False) for _ in range(self.total_cell_num)]
        # self.arch_gamma = [nn.Parameter(torch.randn(n_feats), requires_grad=True) for _ in range(self.total_cell_num)]
        # self.arch_alpha_normal = [nn.Parameter(torch.zeros(k, num_ops), requires_grad=True) for _ in range(self.total_cell_num)]
        self.arch_theta = []
        with torch.no_grad():
            for i in range(self.total_cell_num):
                self.arch_alpha_normal[i].mul_(1e-2)
                self.arch_beta_normal[i].mul_(1e-2)
                # self.arch_gamma[i].mul_(1e-2)

        self.para = self.arch_alpha_normal + self.arch_beta_normal

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        for _ in range(args.n_resblocks * args.n_resgroups):
            self.cells += [Cell(args.nodes, args.nodes, args.n_feats, args.n_feats, args.n_feats, False, False)]

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

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        thetas = []
        for i, cell in enumerate(self.cells):
            if i % args.n_resgroups == 0:
                tmp = res
            weights = F.softmax(self.arch_alpha_normal[i], dim=-1)
            weights1 = F.softmax(self.arch_beta_normal[i], dim=-1)
            # weights2 = F.softmax(self.arch_gamma[i], dim=-1)
            res, theta = cell(res, weights, weights1)
            thetas.append(theta)
            if (i + 1) % args.n_resgroups == 0:
                # res = self.body[(i + 1) // args.n_resgroups - 1](res)
                res += tmp
        # print(ops_att_weight_list)
        # print(self.arch_gamma_normal)
        self.arch_theta = thetas
        # print(self.arch_gamma_normal)
        # exit()
        res += x
        res = self.body[-1](res)
        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def loss(self, x, target):
        """
        :param x:
        :param target:
        :return:
        """
        logits = self(x)
        return self.criterion(logits, target)

    def arch_parameters(self):
        arch_ch_theta = []
        for name, param in self.named_parameters():
            if 'theta' in name:
                arch_ch_theta.append(param)
            else:
                pass
        return self.para + arch_ch_theta

    def model_parameters(self):
        for k, v in self.named_parameters():
            if 'arch_p' not in k:
                yield v

    def save_arch_to_pdf(self, suffix):
        genotype = arch_to_genotype(self.cur_normal_arch, self._steps, "COMPACT")
        return genotype

    def genotype(self):
        def _parse(weights, PRIMITIVES):
            gene = []
            n = 1
            start = 0
            for i in range(self.steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i, i+1), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:1]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], i))
                start = end
            return gene

        genotypes = []
        for i in range(args.n_resblocks * args.n_resgroups):
            gene_op = _parse(F.softmax(self.arch_alpha_normal[i], dim=-1).data.cpu().numpy(), PRIMITIVES)
            gene_att = _parse(F.softmax(self.arch_beta_normal[i], dim=-1).data.cpu().numpy(), ATT_PRIMITIVES)
            concat = range(self.steps, self.steps + 1)
            genotype = Genotype(
                normal=gene_op, normal_concat=concat,
                reduce=gene_att, reduce_concat=concat
            )
            genotypes.append(genotype)

        return genotypes

    def reset_sample(self):
        for m in self.modules():
            if isinstance(m, MixedLayer) or isinstance(m, MixedAtt):
                m.sample = True

    def reset_sample_false(self):
        for m in self.modules():
            if isinstance(m, MixedLayer) or isinstance(m, MixedAtt):
                m.sample = False

    def get_channel(self):
        weight = self.arch_theta
        weight_total = []
        for w in weight:
            weight_total.extend(w)

        total = 0
        for w in weight:
            for ww in w:
                total += ww.data.shape[0]
        bn = torch.zeros(total)
        index = 0
        for w in weight:
            for ww in w:
                size = ww.data.shape[0]
                bn[index:(index + size)] = ww.data.abs().clone()
                index += size

        y, i = torch.sort(bn)  # small ---> big
        # thre_index = int(total * args.percent)
        thre_index = int(total * args.percent)
        thre = y[thre_index]

        # print(bn)
        # print(thre)
        # print(y)
        # exit()

        pruned = 0
        cfg = []
        cfg_mask = []
        for w in weight:
            layer_cfg = []
            layer_cfg_mask = []
            for ww in w:
                weight_copy = ww.data.clone()
                mask = weight_copy.abs().gt(thre).float().cuda()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                layer_cfg.append(max(int(torch.sum(mask)), 4))  # avoid occur zero channel
                layer_cfg_mask.append(mask.clone())
            cfg.append(layer_cfg)
            cfg_mask.append(layer_cfg_mask)

        return cfg

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))



