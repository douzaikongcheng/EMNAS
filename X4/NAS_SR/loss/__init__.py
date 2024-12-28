import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F




import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models

import math


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ContrastLoss(nn.Module):
    def __init__(self, weights, d_func, t_detach = False, is_one=False):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = weights
        self.d_func = d_func
        self.is_one = is_one
        self.t_detach = t_detach

    def forward(self, teacher, student, neg, hr, contra_lambda):
        teacher_vgg, student_vgg, neg_vgg, = self.vgg(teacher), self.vgg(student), self.vgg(neg)
        if self.d_func == "L1":
            self.forward_func = self.L1_forward

        return self.forward_func(teacher_vgg, student_vgg, neg_vgg, hr, student, contra_lambda)

    def L1_forward(self, teacher, student, neg, hr, stu, contra_lambda):
        """
        :param teacher: 5*batchsize*color*patchsize*patchsize
        :param student: 5*batchsize*color*patchsize*patchsize
        :param neg: 5*negnum*color*patchsize*patchsize
        :return:
        """
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4)### batchsize*negnum*color*patchsize*patchsize

            if self.t_detach:
                d_ts = self.l1(teacher[i].detach(), student[i])
            else:
                d_ts = self.l1(teacher[i], student[i])
            d_sn = torch.mean(torch.abs(neg_i.detach() - student[i]).sum(0))

            contrastive = d_ts / (d_sn + 1e-7)
            loss += self.weights[i] * contrastive

        loss += self.l1(hr, stu) + contra_lambda * loss

        return loss

    def calc_cos_stu_neg(self, stu, neg):
        n = stu.shape[0]
        m = neg.shape[0]

        stu = stu.view(n, -1)
        neg = neg.view(m, n, -1)
        # normalize
        stu = F.normalize(stu, p=2, dim=1)
        neg = F.normalize(neg, p=2, dim=2)
        # multiply
        d_sn = torch.mean((stu * neg).sum(0))
        return d_sn


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.t_detach = args.contrast_t_detach

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'CSDLoss':
                loss_function = ContrastLoss(args.vgg_weight, args.d_func, self.t_detach)
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    # def forward(self, sr, hr):
    def forward(self, sr_teacher_, sr_, bic_sample_, hr_, contra_lambda):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                # loss = l['function'](sr, hr)
                loss = l['function'](sr_teacher_, sr_, bic_sample_, hr_, contra_lambda)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            # print('self.log[:, i].numpy():',self.log[:, i].numpy())
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

