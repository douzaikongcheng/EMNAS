import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        # print(self.model, file=ckp.log_file)

    def arch_parameters(self):
        return self.model.arch_parameters()

    def loss(self, x, target):
        return self.model.loss(x, target)

    def new(self):
        return self.model.new()

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        # print(self.idx_scale)
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    # ori forward_chop
    # def forward_chop(self, *args, shave=10, min_size=160000):
    #     # print(self.scale)
    #     # print(self.idx_scale)
    #     # print(self.input_large)
    #     # scale = 1 if self.input_large else self.scale[self.idx_scale]
    #     # if len(*args[0]) > 1:
    #     #     args = [*args[0]]
    #     # print(len(*args[0]))  # 3
    #     # print(args[0].size())
    #     scale = 1 if self.input_large else self.scale[0]
    #     n_GPUs = min(self.n_GPUs, 4)
    #     # height, width
    #     h, w = args[0].size()[-2:]
    #     # print(args[0].size())
    #
    #     top = slice(0, h // 2 + shave)
    #     bottom = slice(h - h // 2 - shave, h)
    #     left = slice(0, w // 2 + shave)
    #     right = slice(w - w // 2 - shave, w)
    #     x_chops = [torch.cat([
    #         a[..., top, left],
    #         a[..., top, right],
    #         a[..., bottom, left],
    #         a[..., bottom, right]
    #     ]) for a in args]
    #
    #     y_chops = []
    #     if h * w < 4 * min_size:
    #         for i in range(0, 4, n_GPUs):
    #             x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
    #             y = P.data_parallel(self.model, *x, range(n_GPUs))
    #             if not isinstance(y, list): y = [y]
    #             if not y_chops:
    #                 # print(len(y))
    #                 # print(y[0].shape)
    #                 # print(y[1].shape)
    #                 # print(len(y[2]))
    #                 # print(y[2][1].shape)
    #                 y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
    #             else:
    #                 for y_chop, _y in zip(y_chops, y):
    #                     y_chop.extend(_y.chunk(n_GPUs, dim=0))
    #     else:
    #         for p in zip(*x_chops):
    #             y = self.forward_chop(*p, shave=shave, min_size=min_size)
    #             if not isinstance(y, list): y = [y]
    #             if not y_chops:
    #                 y_chops = [[_y] for _y in y]
    #             else:
    #                 for y_chop, _y in zip(y_chops, y): y_chop.append(_y)
    #
    #     h *= scale
    #     w *= scale
    #     top = slice(0, h // 2)
    #     bottom = slice(h - h // 2, h)
    #     bottom_r = slice(h // 2 - h, None)
    #     left = slice(0, w // 2)
    #     right = slice(w - w // 2, w)
    #     right_r = slice(w // 2 - w, None)
    #
    #     # batch size, number of color channels
    #     b, c = y_chops[0][0].size()[:-2]
    #     y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
    #     for y_chop, _y in zip(y_chops, y):
    #         _y[..., top, left] = y_chop[0][..., top, left]
    #         _y[..., top, right] = y_chop[1][..., top, right_r]
    #         _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
    #         _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]
    #
    #     if len(y) == 1: y = y[0]
    #
    #     return y


    def forward_chop(self, *args, shave=10, min_size=160000):
        scale = 1 if self.input_large else self.scale[0]
        n_GPUs = min(self.n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]
        # print(args[0].size())

        top = slice(0, h // 2 + shave)
        bottom = slice(h - h // 2 - shave, h)
        left = slice(0, w // 2 + shave)
        right = slice(w - w // 2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                y = P.data_parallel(self.model, *x, range(n_GPUs))
                if isinstance(y, list) and len(y) == 2:
                    new = []
                    new.append(y[0])
                    new.append(y[1][0])
                    new.append(y[1][1])
                    # y : [sr, fms] = [tensor, [tensor, tensor]] = [hw, [hw / 2, hw / 2]]
                    y = new
                    if not isinstance(y, list): y = [y]
                    if not y_chops:
                        y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                    else:
                        for y_chop, _y in zip(y_chops, y):
                            y_chop.extend(_y.chunk(n_GPUs, dim=0))
                elif isinstance(y, list) and len(y) == 3:
                    new = []
                    new.append(y[0])
                    new.append(y[1])
                    new.append(y[2][0])
                    new.append(y[2][1])
                    # y : [sr, var, fms] = [tensor, tensor, [tensor, tensor]] = [hw, hw, [hw / 2, hw / 2]]
                    y = new
                    if not isinstance(y, list): y = [y]
                    if not y_chops:
                        y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                    else:
                        for y_chop, _y in zip(y_chops, y):
                            y_chop.extend(_y.chunk(n_GPUs, dim=0))
                else:
                    if not isinstance(y, list): y = [y]
                    if not y_chops:
                        y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                    else:
                        for y_chop, _y in zip(y_chops, y):
                            y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)
        # print(y_chops[0][0].shape)
        # print(y_chops[0][1].shape)
        # print(y_chops[0][2].shape)
        # print(y_chops[0][3].shape)
        # print(len(y_chops))
        # print(len(y_chops[0]))
        # torch.Size([1, 3, 116, 116])
        # torch.Size([1, 3, 116, 116])
        # torch.Size([1, 3, 116, 116])
        # torch.Size([1, 3, 116, 116])
        # 3
        # 4
        ori_h, ori_w = h, w
        if len(y_chops) == 3:
            # print("3" * 20)
            res = []
            for i in range(3):
                if i == 0:
                    h = scale * ori_h
                    w = scale * ori_w
                else:
                    h = scale * ori_h
                    w = scale * ori_w
                    h = h // 2
                    w = w // 2
                top = slice(0, h // 2)
                bottom = slice(h - h // 2, h)
                bottom_r = slice(h // 2 - h, None)
                left = slice(0, w // 2)
                right = slice(w - w // 2, w)
                right_r = slice(w // 2 - w, None)

                # batch size, number of color channels
                b, c = y_chops[i][0].size()[:-2]

                y_chop = y_chops[i]
                _y = y_chop[0].new(b, c, h, w)
                # print(y_chop[0].shape)
                # print(_y.shape)
                _y[..., top, left] = y_chop[0][..., top, left]
                _y[..., top, right] = y_chop[1][..., top, right_r]
                _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
                _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]
                res.append(_y)
            # for i in res:
            #     print(i.shape)
            return [res[0], [res[1], res[2]]]

        elif len(y_chops) == 4:
            # print("4"*20)
            res = []
            for i in range(4):
                if i < 2:
                    h = scale * ori_h
                    w = scale * ori_w
                else:
                    h = scale * ori_h
                    w = scale * ori_w
                    h = h // 2
                    w = w // 2
                top = slice(0, h // 2)
                bottom = slice(h - h // 2, h)
                bottom_r = slice(h // 2 - h, None)
                left = slice(0, w // 2)
                right = slice(w - w // 2, w)
                right_r = slice(w // 2 - w, None)

                # batch size, number of color channels
                b, c = y_chops[i][0].size()[:-2]

                y_chop = y_chops[i]
                _y = y_chop[0].new(b, c, h, w)
                _y[..., top, left] = y_chop[0][..., top, left]
                _y[..., top, right] = y_chop[1][..., top, right_r]
                _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
                _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]
                res.append(_y)
            return [res[0], res[1], [res[2], res[3]]]
        else:
            h *= scale
            w *= scale
            top = slice(0, h // 2)
            bottom = slice(h - h // 2, h)
            bottom_r = slice(h // 2 - h, None)
            left = slice(0, w // 2)
            right = slice(w - w // 2, w)
            right_r = slice(w // 2 - w, None)

            # batch size, number of color channels
            b, c = y_chops[0][0].size()[:-2]
            y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
            for y_chop, _y in zip(y_chops, y):
                _y[..., top, left] = y_chop[0][..., top, left]
                _y[..., top, right] = y_chop[1][..., top, right_r]
                _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
                _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

            if len(y) == 1: y = y[0]

            return y

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y


class Controller_Model(object):
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.n_GPUs = args.n_GPUs
        # self.optimizer = torch.optim.Adam(self.model.pruner_parameters(),
        #                                   lr=args.pruner_learning_rate, betas=(0.5, 0.999),
        #                                   weight_decay=args.pruner_weight_decay)
        # self.baseline = 0
        # self.gamma = args.gamma

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.n_GPUs > 1:
            return P.data_parallel(self.model, x, idx_scale, range(self.n_GPUs))
        else:
            return self.model(x, idx_scale)

    # def forward(self, input_valid, idx_scale):
    #     # self.optimizer.zero_grad()
    #     # loss, reward, pruned_reward, normal_ent, reduce_ent = self.model._loss_pruner(input_valid, target_valid, self.baseline)
    #     # loss, accuracy, pruned_accuracy, pruner_normal_entropy, reward = self.model._loss_pruner(input_valid)
    #     logits, pruner_normal_logP, pruner_normal_entropy, pruner_reduce_logP, pruner_reduce_entropy, position, log_p_position, entropy_position = self.model.model._loss_pruner(input_valid)
    #     # loss.backward()
    #     # nn.utils.clip_grad_norm(self.model.pruner_parameters(), self.args.grad_clip)
    #     # self.optimizer.step()
    #     # self.update_baseline(reward)
    #     # return loss, reward, accuracy, pruned_accuracy
    #     return logits, pruner_normal_logP, pruner_normal_entropy, pruner_reduce_logP, pruner_reduce_entropy, position, log_p_position, entropy_position
