# coding=utf-8

import torch
from numba import jit

#
@jit
def CELoss(pred, target, mask, eps=1e-8, ):

    loss = -(target * torch.log(pred + eps) + (1. - target) * torch.log(1. - pred + eps))
    loss = torch.sum(loss * mask)

    return loss

@jit
def alpha_CELoss(pred, target, mask, alpha = 0.3, eps=1e-8, ):

    loss = -(alpha*target * torch.log(pred + eps) + (1-alpha) * (1. - target) * torch.log(1. - pred + eps))
    loss = torch.sum(loss * mask)

    return loss

