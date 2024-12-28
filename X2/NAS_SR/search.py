import torch
import  torch.nn.functional as F
import utility
import data
import model
import loss
from option import args
from trainer_search import Trainer
from arch import Arch
from arch_trainer import Arch_Trainer
import time
import os
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    global loss
    if args.data_test == 'video':
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            share_model = model.Model(args, checkpoint)
            print(share_model)
            arch = Arch(share_model,args)
            loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, share_model, loss, checkpoint,arch)

            cur_epoch=0
            while not t.terminate():
                epoch_time = time.time()
                # for i in range(args.n_resblocks * args.n_resgroups):
                #     print(F.softmax(share_model.model.arch_alpha_normal[i], dim=-1).data.cpu().numpy())
                t.train()
                # if cur_epoch >= args.arch_start_training:
                #     arch_trainer_t.arch_train()
                t.test()
                cur_epoch+=1
                epoch_time = time.time() - epoch_time
                print('epochs time: {} h {} min '.format(int(epoch_time / 3600), epoch_time / 60),
                      ' total time: {} h {} min'.format(int(epoch_time * args.epochs / 3600),
                                                        epoch_time * args.epochs / 60 % 60))
            checkpoint.done()

if __name__ == '__main__':
    main()
