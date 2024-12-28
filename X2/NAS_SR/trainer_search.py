import os
import math
from decimal import Decimal
import torch.nn.functional as F
import utility
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
from model.utils import plot_genotype
from model.operationsbn_search import Conv2dWp8, Conv2dWp4
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


from dis_loss import prp_2_oh_array, DisLoss


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, arch):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.arch = arch
        self.loader_train = loader.loader_train
        self.loader_val_train = loader.loader_val_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.torch_version = float(torch.__version__[0:3])

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def compute_flops(self, module: nn.Module, size, skip_pattern):
        def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            *_, h, w = output.shape
            module.output_size = (h, w)

        hooks = []
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(size_hook))
        with torch.no_grad():
            training = module.training
            module.eval()
            module(torch.rand(size).cuda(), 2)
            module.train(mode=training)
        for hook in hooks:
            hook.remove()

        flops = 0
        for name, m in module.named_modules():
            if skip_pattern in name:
                continue
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'output_size'):
                    h, w = m.output_size
                    kh, kw = m.kernel_size
                    flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
            if isinstance(module, nn.Linear):
                flops += m.in_features * m.out_features
        return flops

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def updateBN(self):
        for m in self.model.model.modules():
            if isinstance(m, Conv2dWp8) or isinstance(m, Conv2dWp4):
                m.theta.grad.data.add_(self.args.s * torch.sign(m.theta.data))  # L1

    def train(self):
        self.optimizer.schedule()
        if self.torch_version < 1.1:
            self.loss.step()
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
        lr_ = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr_))
        )
        self.loss.start_log()
        self.model.train()

        # note when size 48 x 48 and flops * 1e-8  is equal to 480 x 480 and flops * 1e-9
        flops = self.compute_flops(self.model, size=(1, self.args.n_colors, 480, 480), skip_pattern='skip')
        # flops = self.compute_flops(self.model, size=(1, self.args.n_colors, 320, 180), skip_pattern='skip')

        used_flops = flops * 1e-8
        self.ckp.write_log(
            '[Model flops(* 1e-8): {}]'.format(used_flops)
        )

        num_parameters = sum([param.nelement() for param in self.model.parameters()])
        print('[Model param: {}]'.format(num_parameters))

        # param = self.count_parameters(self.model)
        # self.ckp.write_log(
        #     '[Model param: {}]'.format(param)
        # )
        # print(self.loader_val_train)
        timer_data, timer_model = utility.timer(), utility.timer()
        valid_iter = iter(self.loader_val_train)

        self.ckp.write_log('op weight')
        for i in range(self.args.n_resblocks * self.args.n_resgroups):
            self.ckp.write_log(str(F.softmax(self.model.model.arch_alpha_normal[i], dim=-1).data.cpu().numpy()))

        self.ckp.write_log('att weight')
        for i in range(self.args.n_resblocks * self.args.n_resgroups):
            self.ckp.write_log(str(F.softmax(self.model.model.arch_beta_normal[i], dim=-1).data.cpu().numpy()))

        if len(self.model.model.arch_theta[0]):
            self.ckp.write_log('ch weight')
            for i in range(len(self.model.model.arch_theta)):
                p = []
                for j in range(len(self.model.model.arch_theta[i])):
                    p.append(self.model.model.arch_theta[i][j].data.cpu().numpy())
                self.ckp.write_log(str(p))

        for batch, (lr, hr, _) in enumerate(self.loader_train):
            idx_scale = None
            lr, hr = self.prepare(lr, hr)
            lr_search, hr_search, _ = next(valid_iter)  # [b, 3, 32, 32], [b]
            lr_search, hr_search = self.prepare(lr_search, hr_search)
            timer_data.hold()
            timer_model.tic()
            ## update alpha
            if epoch > self.args.pretrain_epoch:
                self.model.model.reset_sample()
                self.arch.step(lr, hr, lr_search, hr_search, lr_, self.loss, self.optimizer, unrolled=False, epoch=epoch)
                # print(len(self.model.model.para))
                # print(self.model.model.para)

            ## update network
            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)

            if epoch > self.args.epoch_dis_start:
                loss_psnr = self.loss(sr, hr)
                loss_dis = DisLoss()(self.model.model.arch_alpha_normal) + DisLoss()(self.model.model.arch_beta_normal)
                loss = self.args.loss_weight * loss_dis + loss_psnr
                loss.backward()
            else:
                loss = self.loss(sr, hr)
                loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )

            # sparsity on theta
            # self.updateBN()

            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
                if epoch > self.args.epoch_dis_start:
                    # print("loss:", loss.item(), "   loss_dis:", loss_dis.item())
                    self.ckp.write_log('loss psnr:{:.2f}\tloss dis:{:.2f}\tloss total:{:.2f}'.format(loss_psnr.item(), loss_dis.item(), loss.item()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        if self.torch_version >= 1.1:
            self.loss.step()

    def test(self):
        torch.set_grad_enabled(False)
        if self.torch_version < 1.1:
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        self.model.model.reset_sample_false()
        timer_test = utility.timer()
        best_psnr = 0
        best_epoch = False
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                self.ckp.log[-1, idx_data, idx_scale] = 0
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    # print(lr.shape, hr.shape, sr.shape)
                    save_list = [sr]
                    psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, idx_scale] += psnr
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                if self.ckp.log[-1, idx_data, idx_scale] >= best_psnr:
                    best_psnr = float(self.ckp.log[-1, idx_data, idx_scale])
                genotype = self.model.model.genotype()
                channel = self.model.model.get_channel()

                best = self.ckp.log.max(0)
                if best[1][idx_data, idx_scale] + 1 == epoch:
                    best_epoch = True
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                if epoch % self.args.sampling_epoch_margin == 0:
                    folder = os.path.join('..', 'experiment', self.args.save)
                    self.ckp.write_log('epoch:{}'.format(epoch))
                    # self.ckp.write_log('upsampling_position:{}'.format(upsampling_position))
                    self.ckp.write_log('genotype = {}'.format(genotype))
                    self.ckp.write_log('channel = {}'.format(channel))
                    # plot_genotype(genotype.normal,
                    #               os.path.join(folder, "normal_{}".format(epoch)))
                #     draw_genotype(genotype.upsampling, 4,
                #                   os.path.join(folder, "upsampling_{}_upsamplingPos_{}".format(epoch, upsampling_position)))
                # self.ckp.write_log('upsampling_position:{}'.format(upsampling_position))
                # self.ckp.write_log('genotype:{}'.format(genotype))
                self.ckp.log[-1, idx_data, idx_scale] = best_psnr

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best_epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
        if epoch > self.args.pretrain_epoch:
            self.model.model.reset_sample()

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.final_test()
            return True
        else:
            if self.torch_version < 1.1:
                epoch = self.optimizer.get_last_epoch() + 1
            else:
                epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs

    def final_test(self):
        torch.set_grad_enabled(False)

        if self.torch_version < 1.1:
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                # for lr, hr, filename, _ in tqdm(d, ncols=80):
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if self.args.save_results:
            self.ckp.end_background()
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)

    def derive(self):
        torch.set_grad_enabled(False)

        if self.torch_version < 1.1:
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                result = {}
                best_derive_psnr = 0
                for i in range(10):
                    result_psnr = []
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        save_list = [sr]
                        psnr = utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                        result_psnr.append(psnr)
                        if self.args.save_gt:
                            save_list.extend([lr, hr])
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} '.format(
                            d.dataset.name,
                            scale,
                            sum(result_psnr) / len(result_psnr)
                        )
                    )
                    if sum(result_psnr) / len(result_psnr) >= best_derive_psnr:
                        best_derive_psnr = sum(result_psnr) / len(result_psnr)
                    genotype, upsampling_position = self.model.model.save_arch_to_pdf(epoch)
                    self.ckp.write_log('genotype:{}'.format(genotype))
                    result[sum(result_psnr) / len(result_psnr)] = (genotype, upsampling_position)
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        self.ckp.write_log('Best psnr:{:.3f}'.format(best_derive_psnr))
        self.ckp.write_log('Best Position:{}'.format(result[best_derive_psnr][1]))
        self.ckp.write_log('Best genotype:{}'.format(result[best_derive_psnr][0]))

        if self.args.save_results:
            self.ckp.end_background()
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
