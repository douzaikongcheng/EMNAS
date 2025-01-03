import  torch
import  numpy as np
from    torch import optim, autograd
import torch.nn.functional as F
import torch.nn as nn
from dis_loss import prp_2_oh_array, DisLoss


def concat(xs):
    """
    flatten all tensor from [d1,d2,...dn] to [d]
    and then concat all [d_1] to [d_1+d_2+d_3+...]
    :param xs:
    :return:
    """
    return torch.cat([x.view(-1) for x in xs])


class Arch:

    def __init__(self, model, args):
        """

        :param model: network
        :param args:
        """
        self.momentum = args.momentum # momentum for optimizer of theta
        self.wd = args.wd # weight decay for optimizer of theta
        self.model = model # main model with respect to theta and alpha
        # this is the optimizer to optimize alpha parameter
        self.args = args  # main model with respect to theta and alpha
        self.optimizer = optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_lr,
                                          betas=(0.5, 0.999),
                                          weight_decay=args.arch_wd)

    def comp_unrolled_model(self, x, target, eta, optimizer):
        """
        loss on train set and then update w_pi, not-in-place
        :param x:
        :param target:
        :param eta:
        :param optimizer: optimizer of theta, not optimizer of alpha
        :return:
        """
        # forward to get loss
        loss = self.model.loss(x, target)
        # flatten current weights
        theta = concat(self.model.parameters()).detach()
        # theta: torch.Size([1930618])
        # print('theta:', theta.shape)
        try:
            # fetch momentum data from theta optimizer
            moment = concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)
        except:
            moment = torch.zeros_like(theta)

        # flatten all gradients
        #print(loss, self.model.parameters())
        #self.model = self.model.cuda()
        #for param in self.model.parameters():
            #print(param.device)
        dtheta = concat(autograd.grad(loss, self.model.parameters())).data
        # indeed, here we implement a simple SGD with momentum and weight decay
        # theta = theta - eta * (moment + weight decay + dtheta)
        theta = theta.sub(eta, moment + dtheta + self.wd * theta)
        # construct a new model
        unrolled_model = self.construct_model_from_theta(theta)

        return unrolled_model

    def step(self, x_train, target_train, x_valid, target_valid, eta, loss, optimizer, unrolled, epoch):
        """
        update alpha parameter by manually computing the gradients
        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta:
        :param optimizer: theta optimizer
        :param unrolled:
        :return:
        """
        # alpha optimizer
        self.optimizer.zero_grad()

        # compute the gradient and write it into tensor.grad
        # instead of generated by loss.backward()
        if unrolled:
            self.backward_step_unrolled(x_train, target_train, x_valid, target_valid, eta, optimizer)
        else:
            # directly optimize alpha on w, instead of w_pi
            self.backward_step(x_valid, target_valid, epoch)

        self.optimizer.step()

    def backward_step(self, x_valid, target_valid, epoch):
        """
        simply train on validate set and backward
        :param x_valid:
        :param target_valid:
        :return:
        """
        # loss = self.model.loss(x_valid, target_valid)
        # # both alpha and theta require grad but only alpha optimizer will
        # # step in current phase.
        # loss.backward()

        if epoch > self.args.epoch_dis_start:
            loss_psnr = self.model.loss(x_valid, target_valid)
            loss_dis = DisLoss()(self.model.model.arch_alpha_normal) + DisLoss()(self.model.model.arch_beta_normal)
            loss = self.args.loss_weight * loss_dis + loss_psnr
            loss.backward()
        else:
            loss = self.model.loss(x_valid, target_valid)
            loss.backward()

    def backward_step_unrolled(self, x_train, target_train, x_valid, target_valid, eta, optimizer):
        """
        train on validate set based on update w_pi
        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta: 0.01, according to author's comments
        :param optimizer: theta optimizer
        :return:
        """

        # theta_pi = theta - lr * grad
        unrolled_model = self.comp_unrolled_model(x_train, target_train, eta, optimizer)
        # calculate loss on theta_pi
        unrolled_loss = unrolled_model.loss(x_valid, target_valid)

        # this will update theta_pi model, but NOT theta model
        unrolled_loss.backward()
        # grad(L(w', a), a), part of Eq. 6
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self.hessian_vector_product(vector, x_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            # g = g - eta * ig, from Eq. 6
            g.data.sub_(eta, ig.data)

        # write updated alpha into original model
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def construct_model_from_theta(self, theta):
        """
        construct a new model with initialized weight from theta
        it use .state_dict() and load_state_dict() instead of
        .parameters() + fill_()
        :param theta: flatten weights, need to reshape to original shape
        :return:
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = v.numel()
            # restore theta[] value to original shape
            # k = k[6:]
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def hessian_vector_product(self, vector, x, target, r=1e-2):
        """
        slightly touch vector value to estimate the gradient with respect to alpha
        refer to Eq. 7 for more details.
        :param vector: gradient.data of parameters theta
        :param x:
        :param target:
        :param r:
        :return:
        """
        R = r / concat(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            # w+ = w + R * v
            p.data.add_(R, v)
        loss = self.model.loss(x, target)
        # gradient with respect to alpha
        grads_p = autograd.grad(loss, self.model.arch_parameters())


        for p, v in zip(self.model.parameters(), vector):
            # w- = (w+R*v) - 2R*v
            p.data.sub_(2 * R, v)
        loss = self.model.loss(x, target)
        grads_n = autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # w = (w+R*v) - 2R*v + R*v
            p.data.add_(R, v)

        h= [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
        # h len: 2 h0 torch.Size([14, 8])
        # print('h len:', len(h), 'h0', h[0].shape)
        return h
