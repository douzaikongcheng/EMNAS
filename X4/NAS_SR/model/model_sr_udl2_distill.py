from model import common
from model.model_sr_udl1_distill import Network as EDSR
from model.model_sr_udl1_distill import EDSR_U
import torch.nn as nn
import torch
import random


def make_model(args, parent=False):
    return EDSR_two(args)


class EDSR_two(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        self.args = args
        super(EDSR_two, self).__init__()
        self.EDSR_var = EDSR(args=args)
        # self.EDSR_U = EDSR(args=args)
        self.EDSR_U = EDSR_U(args=args)
        if args.pre_train_step1 != '.':
            self.EDSR_var.load_state_dict(torch.load(args.pre_train_step1), strict=True)
            # self.EDSR_U.load_state_dict(torch.load(args.pre_train_step1), strict=True)
            self.EDSR_U.load_state_dict(torch.load(args.pre_train_step1), strict=False)

    def forward(self, x):
        if not self.args.test_only:
            with torch.no_grad():
                var = self.EDSR_var(x)
            x = self.EDSR_U(x)
            # x = x[0]
            # return [x[0], var[1]]
            return [x, var[1]]
        else:
            x = self.EDSR_U(x)
            x = x[0]
            return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
