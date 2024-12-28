import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device):
    # TODO: shuffle?

    # Grab a single batch from the training dataset

    # inputs, targets, _ = next(iter(train_dataloader))
    # inputs = inputs.to(device)
    # targets = targets.to(device)
    # print(inputs.shape)
    # print(targets.shape)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if (isinstance(layer, nn.Conv2d) and layer.weight.data.shape != torch.Size([3, 3, 1, 1]) and layer.weight.data.shape != torch.Size([32, 3, 3, 3]) and layer.weight.data.shape != torch.Size([48, 12, 3, 3]) and layer.weight.data.shape != torch.Size([3, 32, 3, 3])) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            # nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d) and layer.weight.data.shape != torch.Size([3, 3, 1, 1]) and layer.weight.data.shape != torch.Size([32, 3, 3, 3]) and layer.weight.data.shape != torch.Size([48, 12, 3, 3]) and layer.weight.data.shape != torch.Size([3, 32, 3, 3]):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    # net.zero_grad()
    # outputs = net.forward(inputs)
    # loss = F.l1_loss(outputs, targets)
    # loss.backward()

    grads_abs = []
    for layer in net.modules():
        if (isinstance(layer, nn.Conv2d) and layer.weight.data.shape != torch.Size([3, 3, 1, 1]) and layer.weight.data.shape != torch.Size([32, 3, 3, 3]) and layer.weight.data.shape != torch.Size([48, 12, 3, 3]) and layer.weight.data.shape != torch.Size([3, 32, 3, 3])) or isinstance(layer, nn.Linear):
            # grads_abs.append(torch.abs(layer.weight_mask.grad))
            grads_abs.append(torch.abs(layer.weight))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    # norm_factor = torch.sum(all_scores)
    # all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]
    print(acceptable_score)

    keep_masks = []
    for g in grads_abs:
        # keep_masks.append(((g / norm_factor) >= acceptable_score).float())
        keep_masks.append((g >= acceptable_score).float())
        # print((g >= acceptable_score).float())

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return(keep_masks)
