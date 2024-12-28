import torch
import torch.nn as nn
import utility
import data
import model
import loss
from option_main import args
from trainer import Trainer
from controller_trainer import Controller_Trainer
import time
import os
from snip import SNIP

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: (isinstance(layer, nn.Conv2d) and layer.weight.data.shape != torch.Size([3, 3, 1, 1])) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        # print(keep_mask)
        layer.weight.register_hook(hook_factory(keep_mask))


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
            # print(share_model.model.body[0].weight)
            # print(share_model.model.tail[1].weight)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # keep_masks = SNIP(share_model.model, 0.8, loader.loader_train, device)  # TODO: shuffle?
            keep_masks = SNIP(share_model.model, args.percent_ir, loader.loader_train, device)  # TODO: shuffle?
            # for p in keep_masks:
            #     print(p.shape)
            apply_prune_mask(share_model.model, keep_masks)
            # print(share_model.model.body[0].weight)
            # print(share_model.model.head[0].weight)
            print(share_model.model.tail[1].weight)
            utility.get_parameter_number(share_model)
            loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, share_model, loss, checkpoint)
            while not t.terminate():
                epoch_time = time.time()
                t.train()
                t.final_test()
                epoch_time = time.time() - epoch_time
                print('epochs time: {} h {} min '.format(int(epoch_time / 3600), epoch_time / 60),
                      ' total time: {} h {} min'.format(int(epoch_time * args.epochs / 3600),
                                                        epoch_time * args.epochs / 60 % 60))

            checkpoint.done()


if __name__ == '__main__':
    main()
