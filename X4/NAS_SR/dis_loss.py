import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


def prp_2_oh_array(arr):
    arr_size = arr.shape[1]
    arr_max = np.argmax(arr, axis=1)
    oh_arr = np.eye(arr_size)[arr_max]
    return oh_arr


class DisLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L2"""

    def __init__(self, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(DisLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input):
        # print(input)
        # print(input[0])
        # print(input[0].squeeze())

        # i = input[0]
        # print(F.softmax(i, dim=-1).cpu().detach().squeeze())
        # print(torch.tensor(prp_2_oh_array(F.softmax(i, dim=-1).cpu().detach())))

        # loss = sum([F.mse_loss(F.softmax(i, dim=-1).cpu().detach().squeeze(), torch.tensor(prp_2_oh_array(F.softmax(i, dim=-1).cpu().detach()), requires_grad=False, dtype=i.dtype)) for i in input])
        # loss = sum([F.l1_loss(F.softmax(i, dim=-1).cpu().detach().squeeze(), torch.tensor(prp_2_oh_array(F.softmax(i, dim=-1).cpu().detach()), requires_grad=False, dtype=i.dtype)) for i in input])
        # loss = sum([F.cross_entropy(F.softmax(i, dim=-1).cpu().detach().squeeze(), torch.tensor(prp_2_oh_array(F.softmax(i, dim=-1).cpu().detach()), requires_grad=False, dtype=i.dtype)) for i in input])
        # loss = sum([F.kl_div(F.softmax(i, dim=-1).cpu().detach().squeeze(), torch.tensor(prp_2_oh_array(F.softmax(i, dim=-1).cpu().detach()), requires_grad=False, dtype=i.dtype)) for i in input])

        # loss = sum([torch.sum(torch.mul(F.softmax(i, dim=-1).cpu().detach().squeeze(), torch.log(F.softmax(i, dim=-1).cpu().detach().squeeze()))) for i in input])
        # print(loss)
        # loss = sum([F.kl_div(F.softmax(i, dim=-1).cpu().detach().squeeze(), torch.tensor(prp_2_oh_array(F.softmax(i, dim=-1).cpu().detach()), requires_grad=False, dtype=i.dtype)) for i in input])
        # print(loss)

        # loss = sum([F.cross_entropy(F.softmax(i, dim=-1).cpu().detach().squeeze(), torch.tensor(prp_2_oh_array(F.softmax(i, dim=-1).cpu().detach()), requires_grad=False, dtype=i.dtype)) for i in input])
        # loss = F.mse_loss(input, torch.tensor(prp_2_oh_array(input.cpu().detach()), requires_grad=False, dtype=input.dtype).cuda())
        # i = input[0]
        # out = F.softmax(i, dim=-1).cpu().detach().squeeze()
        # one_hot_out = torch.tensor(prp_2_oh_array(F.softmax(i, dim=-1).cpu().detach()), requires_grad=False, dtype=i.dtype)
        # print(out)
        # print(one_hot_out)
        # print(torch.max(out))
        # loss = F.cross_entropy(out, one_hot_out)

        loss = sum([F.cross_entropy(F.softmax(i, dim=-1).cpu().detach(), torch.tensor([torch.argmax(i)])) for i in input])
        # loss = torch.sum(torch.tensor([F.cross_entropy(F.softmax(i, dim=-1), torch.tensor([torch.argmax(i)])) for i in input]))
        return loss