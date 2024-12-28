import os
import math
from decimal import Decimal
import  torch.nn.functional as F
import utility

import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
from model.utils import  plot_genotype
from model import vis_fea_map
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def spatial_similarity(fm):
    # print(fm.shape)  # torch.Size([8, 32, 96, 96]) LR patch
    fm = fm.view(fm.size(0), fm.size(1), -1)
    # print(fm.shape)  # torch.Size([8, 32, 9216]) 9216=96*96
    # print(torch.pow(fm, 2).shape)  # torch.Size([8, 32, 9216])
    # print(torch.sum(torch.pow(fm,2), 1).shape)  # torch.Size([8, 9216])
    # print(torch.sqrt(torch.sum(torch.pow(fm,2), 1)).shape)  # torch.Size([8, 9216])
    # print(torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).shape)  # torch.Size([8, 1, 9216])
    # print(torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape).shape)  # torch.Size([8, 32, 9216])
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
    # print(norm_fm.shape)  # torch.Size([8, 32, 9216])
    # print(norm_fm.transpose(1,2).shape)  # torch.Size([8, 9216, 32])
    # bbm [8, 9216, 32] * [8, 32, 9216] = [8, 9216, 9216]
    s = norm_fm.transpose(1, 2).bmm(norm_fm)
    # print(s.shape)  # torch.Size([8, 9216, 9216])
    s = s.unsqueeze(1)
    # print(s.shape)  # torch.Size([8, 1, 9216, 9216])
    return s


class FeatureLoss(nn.Module):
    def __init__(self, loss=nn.L1Loss()):
        super(FeatureLoss, self).__init__()
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        length = len(outputs)
        tmp = [self.loss(outputs[i], targets[i]) for i in range(length)]
        loss = sum(tmp)
        return loss


class Trainer():
    def __init__(self, args, loader, my_model, my_model_teacher, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_teacher = my_model_teacher
        self.model_teacher.model.load_state_dict(torch.load('../experiment/train_teacher/model/model_best.pt'), strict=True)
        self.loss = my_loss
        # print(self.model_teacher.model.tail[1].weight.data[0][0])
        self.optimizer = utility.make_optimizer(args, self.model)
        self.torch_version = float(torch.__version__[0:3])

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        # self.final_test()

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
            # module(torch.rand(size), 2)
            # module(torch.rand(size), 1)
            module(torch.rand(size), 0)
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

    def train(self):
        self.optimizer.schedule()
        if self.torch_version < 1.1:
            self.loss.step()
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        # # note when size 48 x 48 and flops * 1e-8  is equal to 480 x 480 and flops * 1e-9
        # flops = self.compute_flops(self.model, size=(1, self.args.n_colors, 480, 480), skip_pattern='skip')
        # # flops = self.compute_flops(self.model, size=(1, self.args.n_colors, 320, 180), skip_pattern='skip')
        # used_flops = flops * 1e-9
        # self.ckp.write_log(
        #     '[Model flops(* 1e-9): {}]'.format(used_flops)
        # )
        params = self.count_parameters(self.model)
        self.ckp.write_log(
            '[Model params: {}]'.format(params)
        )

        # param = self.count_parameters(self.model)
        # self.ckp.write_log(
        #     '[Model param: {}]'.format(param)
        # )

        self.model_teacher.eval()
        # 冻结参数更省显存
        for p in self.model_teacher.model.parameters():
            p.requires_grad = False

        # # flops_teacher = self.compute_flops(self.model_teacher, size=(1, self.args.n_colors, 720, 1300), skip_pattern='skip')
        # flops_teacher = self.compute_flops(self.model_teacher, size=(1, self.args.n_colors, 480, 480),
        #                                    skip_pattern='skip')
        # used_flops_teacher = flops_teacher * 1e-9
        # self.ckp.write_log(
        #     '[teacher Model flops(* 1e-9): {}]'.format(used_flops_teacher)
        # )
        params_teacher = self.count_parameters(self.model_teacher)
        self.ckp.write_log(
            '[teacher Model params: {}]'.format(params_teacher)
        )

        timer_data, timer_model = utility.timer(), utility.timer()

        alpha = 0.5

        loss_fms = FeatureLoss(loss=nn.L1Loss())

        for batch, (lr, hr, _) in enumerate(self.loader_train):
            idx_scale=None
            lr, hr = self.prepare(lr, hr)
            # noise = torch.randn_like(hr) * self.args.noise_std
            # lr = hr + noise
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)  # train : [sr, var, fms], test : [sr]
            sr_teacher = self.model_teacher(lr, idx_scale)  # [sr, fms]

            # print(self.model.chop)
            # print(self.model_teacher.chop)
            # exit()

            # print(self.model_teacher.model.tail[1].weight.requires_grad)
            # print(self.model_teacher.model.tail[1].weight.data.requires_grad)
            # print(self.model_teacher.model.tail[1].weight.data[0][0])

            # print(self.model.model.EDSR_U.tail[1].weight.data[0][0])

            student_fms = sr[2]
            teacher_fms = sr_teacher[1]
            # print(len(student_fms))
            # print(len(teacher_fms))
            aggregated_student_fms = []
            aggregated_teacher_fms = []

            # print(len(spatial_similarity(student_fms[0])))

            aggregated_student_fms.append([spatial_similarity(fm) for fm in student_fms])
            aggregated_teacher_fms.append([spatial_similarity(fm) for fm in teacher_fms])

            # print(len(aggregated_student_fms))

            if isinstance(sr, list):

                # step1
                if self.args.stage == 'step1':
                    u = 255 * abs(hr - sr[0])
                    var = torch.exp(sr[1])
                    # b, c, h, w = var.shape
                    # var = var.view(b,c,-1)
                    # print(torch.max(var, dim=-1))
                    loss = self.loss(255 * sr[0], 255 * hr) + torch.mean(
                        torch.mul(var, torch.exp(-torch.div(u, var))) - sr[1] - 1)

                # step2
                elif self.args.stage == 'step2':
                    b, c, h, w = sr[1].shape
                    s1 = sr[1].view(b, c, -1)
                    pmin = torch.min(s1, dim=-1)
                    pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                    s = sr[1]
                    s = s - pmin + 1
                    sr_ = torch.mul(sr[0], s)
                    hr_ = torch.mul(hr, s)
                    # loss = self.loss(sr_, hr_)
                    sr_teacher_ = torch.mul(sr_teacher[0], s)

                    # 不是udl loss的原因
                    # l1 = (1 - alpha) * self.loss(sr[0], hr)
                    # l2 = alpha * self.loss(sr[0], sr_teacher[0])
                    # l3 = 10 * loss_fms(aggregated_student_fms[0], aggregated_teacher_fms[0])

                    l1 = (1 - alpha) * self.loss(sr_, hr_)
                    l2 = alpha * self.loss(sr_, sr_teacher_)
                    # print(len(aggregated_student_fms))
                    l3 = 10 * loss_fms(aggregated_student_fms[0], aggregated_teacher_fms[0])

                    # l1 = (1 - alpha) * self.loss(sr_, hr_)
                    # # l1 = self.loss(sr_, hr_)
                    # # print((1 - alpha) * self.loss(sr_, hr_))
                    # print(self.loss(sr_, hr_))
                    #
                    # l2 = alpha * self.loss(sr_, sr_teacher_)
                    #
                    # print(self.loss(hr_, sr_teacher_))
                    # #
                    # #
                    # print(l2)
                    # l3 = 10 * loss_fms(aggregated_student_fms[0], aggregated_teacher_fms[0])
                    # # print(l1.item(), l2.item(), l3.item())
                    # # 9.880887031555176 19.965251922607422 0.9185634851455688
                    loss = l1 + l2 + l3
                    # loss = l1 + l2
                    # loss = l1


                else:
                    loss = self.loss(sr[0], hr)
            else:
                loss = self.loss(sr, hr)

            # loss = self.loss(sr, hr)

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                if self.args.gclip > 0:
                    utils.clip_grad_value_(
                        self.model.parameters(),
                        self.args.gclip
                    )
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            # loss.backward()
            # if self.args.gclip > 0:
            #     utils.clip_grad_value_(
            #         self.model.parameters(),
            #         self.args.gclip
            #     )
            # self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

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
        # self.model_teacher.eval()
        timer_test = utility.timer()
        best_psnr = 0
        best_epoch = False
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                # self.ckp.log[-1, idx_data, idx_scale]=0
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    # noise = torch.randn_like(hr) * self.args.noise_std
                    # lr = hr + noise
                    sr = self.model(lr, idx_scale)
                    # sr = self.model_teacher(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    psnr=utility.calc_psnr(
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

                best = self.ckp.log.max(0)
                if best[1][idx_data, idx_scale]+1 == epoch:
                    best_epoch=True
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale]+1
                    )
                )
                if epoch % self.args.sampling_epoch_margin == 0:

                    folder = os.path.join( '..', 'experiment', self.args.save)
                    self.ckp.write_log('epoch:{}'.format(epoch))
                    self.ckp.write_log('upsampling_position:{}'.format(upsampling_position))
                    self.ckp.write_log('genotype:{}'.format(genotype))
                    plot_genotype(genotype.normal,
                                  os.path.join(folder, "normal_{}".format(epoch)))
                #     draw_genotype(genotype.upsampling, 4,
                #                   os.path.join(folder, "upsampling_{}_upsamplingPos_{}".format(epoch, upsampling_position)))
                # self.ckp.write_log('upsampling_position:{}'.format(upsampling_position))
                self.ckp.write_log('genotype:{}'.format(genotype))
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
        # self.model.eval()
        self.model_teacher.eval()


        # self.model.eval()
        best_psnr = 0
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                # for lr, hr, filename, _ in tqdm(d, ncols=80):
                idx_img = 0
                for lr, hr, filename in tqdm(d, ncols=80):
                    idx_img += 1
                    lr, hr = self.prepare(lr, hr)
                    # noise = torch.randn_like(hr) * self.args.noise_std
                    # lr = hr + noise


                    # udl
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        # lr, hr = self.prepare([lr, hr])
                        lr, hr = self.prepare(lr, hr)
                    else:
                        # lr = self.prepare([lr])[0]
                        lr = self.prepare(lr)[0]


                    sr = self.model(lr,idx_scale)
                    # sr = self.model_teacher(lr,idx_scale)
                    # sr = sr[0]
                    # var = None

                    if isinstance(sr, list):
                        var = sr[1]
                        sr = sr[0]
                        if var.size(1) > 1:
                            convert = var.new(1, 3, 1, 1)
                            convert[0, 0, 0, 0] = 65.738
                            convert[0, 1, 0, 0] = 129.057
                            convert[0, 2, 0, 0] = 25.064
                            var.mul_(convert).div_(256)
                            var = var.sum(dim=1, keepdim=True)
                    else:
                        var = None
                    # print(self.args.rgb_range)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]

                    # udl
                    # if not no_eval:
                    #     eval_acc += utility.calc_psnr(
                    #         sr, hr, scale, self.args.rgb_range,
                    #         benchmark=self.loader_test.dataset.benchmark
                    #     )
                    #     save_list.extend([lr, hr])
                    if not no_eval:
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                    # udl
                    if var is not None:
                        path = "../experiment/" + self.args.save + "/model/" + "{:03d}_var.png".format(idx_img)
                        vis_fea_map.draw_features(var.cpu().numpy(), path)


                    # if var is not None:
                    #     vis_fea_map.draw_features(var.cpu().numpy(),
                    #                               "{}/results/SR/{}/X{}/{:03d}_var.png".format(self.args.save,
                    #                                                                            # self.args.testset,
                    #                                                                            self.args.data_test,
                    #                                                                            self.args.scale[0],
                    #                                                                            idx_img))

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
                best_psnr = best[0][idx_data, idx_scale]
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
        return best_psnr.item()

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
                result={}
                best_derive_psnr=0
                for i in range(10):
                    result_psnr = []
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        save_list = [sr]
                        psnr=utility.calc_psnr(
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
                    if sum(result_psnr) / len(result_psnr)>=best_derive_psnr:
                        best_derive_psnr=sum(result_psnr) / len(result_psnr)
                    genotype,upsampling_position = self.model.model.save_arch_to_pdf(epoch)
                    self.ckp.write_log('genotype:{}'.format(genotype))
                    result[sum(result_psnr) / len(result_psnr)] =(genotype,upsampling_position)
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