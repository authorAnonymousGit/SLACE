import warnings
import pandas as pd
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from collections import Counter

from utils import *


def change_inputs(targets, softmax_vals):
    targets = targets + 1
    labels_num = int(np.max(targets))
    d = Counter(targets)
    dist_dict = {int(k): int(v) for k, v in d.items()}
    prox_mat = create_prox_mat(dist_dict, inv=False)
    norm_prox_mat = f.normalize(prox_mat, p=1, dim=0)
    prox_dom = create_prox_dom(prox_mat).double()

    targets = targets - 1
    softmax_vals = torch.tensor(softmax_vals, requires_grad=True)
    targets = torch.tensor(targets)
    targets.requires_grad_(False)
    num_examples = targets.shape[0]
    # print(norm_prox_mat)
    # print(prox_mat)
    return labels_num, prox_mat.T, norm_prox_mat.T, prox_dom, softmax_vals, targets.long(), num_examples


def WOCEL_alpha(alpha, return_loss=False):
    def Loss(targets, softmax_vals):
        labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
                                                                                                           softmax_vals)

        one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
        one_hot_target[range(num_examples), targets] = 1
        one_hot_target_comp = 1 - one_hot_target
        elem_weight = torch.tensor(norm_prox_mat[:, targets]).t()
        wrong_mass_weight = torch.sum(one_hot_target_comp * softmax_vals, dim=1).unsqueeze(dim=1) * one_hot_target_comp
        mass_weights = alpha * one_hot_target + (1 - alpha) * wrong_mass_weight * elem_weight
        sum_coeff = torch.matmul(softmax_vals.unsqueeze(dim=1).double(), prox_dom[targets]).squeeze(dim=1)

        mass_weights = mass_weights.detach()
        sum_coeff = sum_coeff.detach()

        loss_val = -1 * torch.sum(mass_weights * torch.log(softmax_vals / sum_coeff))

        if return_loss == True:
            return loss_val

        loss_val.backward()
        grads = softmax_vals.grad.numpy()

        del softmax_vals, targets, sum_coeff, one_hot_target, one_hot_target_comp, wrong_mass_weight, \
            mass_weights, elem_weight

        return grads.flatten(), np.ones(grads.shape).flatten()

    return Loss


def WOCEL_accumulating(alpha, return_loss=False):
    def Loss(targets, softmax_vals):
        labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
                                                                                                           softmax_vals)

        one_hot_target = torch.tensor(np.zeros([num_examples, labels_num]))
        one_hot_target[range(num_examples), targets.long()] = 1
        one_hot_target_comp = 1 - one_hot_target
        mass_weights = alpha * one_hot_target + (1 - alpha) * one_hot_target_comp

        accumulating_softmax = torch.matmul(prox_dom[targets.long()].double(),
                                            torch.unsqueeze(softmax_vals, 2).double()).double().squeeze(dim=2)

        loss_val = -1 * torch.sum(mass_weights * torch.log(accumulating_softmax))

        if return_loss == True:
            return loss_val

        loss_val.backward()
        grads = softmax_vals.grad.numpy()

        del softmax_vals, targets, accumulating_softmax, one_hot_target, one_hot_target_comp, \
            mass_weights

        return grads.flatten(), np.ones(grads.shape).flatten()

    return Loss


def accumulating_with_sord_prox(alpha, return_loss=False, type="max"):
    def Loss(targets, softmax_vals):
        labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
                                                                                                           softmax_vals)

        if type == "max":
            phi = torch.max(prox_mat) - prox_mat[targets].to(device)
        if type == "norm_max":
            phi = torch.max(norm_prox_mat) - norm_prox_mat[targets].to(device)
        if type == "norm_log":
            phi = - torch.log(norm_prox_mat[targets].to(device))
        if type == "log":
            phi = - torch.log(prox_mat[targets].to(device))
        if type == "norm_division":
            phi = 1/(norm_prox_mat[targets].to(device))
        if type == "division":
            phi = 1/(prox_mat[targets].to(device))

        softmax_targets = f.softmax(-alpha * phi, dim=1).to(device)

        one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
        one_hot_target[range(num_examples), targets] = 1
        one_hot_target_comp = 1 - one_hot_target
        mass_weights = one_hot_target * softmax_targets + one_hot_target_comp * softmax_targets

        accumulating_softmax = torch.matmul(prox_dom[targets.long()].double(),
                                            torch.unsqueeze(softmax_vals, 2).double()).double().squeeze(dim=2)

        loss_val = -1 * torch.sum(mass_weights * torch.log(accumulating_softmax))

        if return_loss == True:
            return loss_val

        loss_val.backward()
        grads = softmax_vals.grad.numpy()

        del softmax_vals, targets, accumulating_softmax, one_hot_target, one_hot_target_comp, \
            mass_weights

        return grads.flatten(), np.ones(grads.shape).flatten()

    return Loss


#
#
# def accumulating_with_sord_prox(alpha, return_loss=False):
#     def Loss(targets, softmax_vals):
#         labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
#                                                                                                            softmax_vals)
#
#         phi = torch.max(prox_mat) - prox_mat[targets].to(device)
#         softmax_targets = f.softmax(-alpha * phi, dim=1).to(device)
#
#         one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
#         one_hot_target[range(num_examples), targets] = 1
#         one_hot_target_comp = 1 - one_hot_target
#         mass_weights = one_hot_target * softmax_targets + one_hot_target_comp * softmax_targets
#
#         accumulating_softmax = torch.matmul(prox_dom[targets.long()].double(),
#                                             torch.unsqueeze(softmax_vals, 2).double()).double().squeeze(dim=2)
#
#         loss_val = -1 * torch.sum(mass_weights * torch.log(accumulating_softmax))
#
#         if return_loss == True:
#             return loss_val
#
#         loss_val.backward()
#         grads = softmax_vals.grad.numpy()
#
#         del softmax_vals, targets, accumulating_softmax, one_hot_target, one_hot_target_comp, \
#             mass_weights
#
#         return grads.flatten(), np.ones(grads.shape).flatten()
#
#     return Loss
#
#
# def accumulating_with_sord_norm_prox(alpha, return_loss=False):
#     def Loss(targets, softmax_vals):
#         labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
#                                                                                                            softmax_vals)
#
#         phi = torch.max(norm_prox_mat) - norm_prox_mat[targets].to(device)
#         softmax_targets = f.softmax(-alpha * phi, dim=1).to(device)
#
#         one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
#         one_hot_target[range(num_examples), targets] = 1
#         one_hot_target_comp = 1 - one_hot_target
#         mass_weights = one_hot_target * softmax_targets + one_hot_target_comp * softmax_targets
#
#         accumulating_softmax = torch.matmul(prox_dom[targets.long()].double(),
#                                             torch.unsqueeze(softmax_vals, 2).double()).double().squeeze(dim=2)
#
#         loss_val = -1 * torch.sum(mass_weights * torch.log(accumulating_softmax))
#
#         if return_loss == True:
#             return loss_val
#
#         loss_val.backward()
#         grads = softmax_vals.grad.numpy()
#
#         del softmax_vals, targets, accumulating_softmax, one_hot_target, one_hot_target_comp, \
#             mass_weights
#
#         return grads.flatten(), np.ones(grads.shape).flatten()
#
#     return Loss

def accumulating_with_sord(alpha, return_loss=False):
    def Loss(targets, softmax_vals):
        labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
                                                                                                           softmax_vals)

        phi = torch.abs(torch.arange(labels_num, device=device).view(1, -1) - targets.double().view(-1, 1))

        softmax_targets = f.softmax(-alpha * phi, dim=1).to(device)

        one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
        one_hot_target[range(num_examples), targets] = 1
        one_hot_target_comp = 1 - one_hot_target
        mass_weights = one_hot_target * softmax_targets + one_hot_target_comp * softmax_targets

        matrices = []
        labels = list(range(labels_num))
        for h in range(labels_num):
            matrix = torch.zeros((labels_num, labels_num), dtype=torch.float32, device=device)

            for i in range(labels_num):
                for j in range(labels_num):
                    distance_i = abs(labels[i] - labels[h])
                    distance_j = abs(labels[j] - labels[h])

                    if distance_j <= distance_i:
                        matrix[i, j] = 1

            matrices.append(matrix)
        prox_dom = torch.stack(matrices)

        accumulating_softmax = torch.matmul(prox_dom[targets.long()].double(),
                                            torch.unsqueeze(softmax_vals, 2).double()).double().squeeze(dim=2)

        loss_val = -1 * torch.sum(mass_weights * torch.log(accumulating_softmax))

        if return_loss == True:
            return loss_val

        loss_val.backward()
        grads = softmax_vals.grad.numpy()

        del softmax_vals, targets, accumulating_softmax, one_hot_target, one_hot_target_comp, \
            mass_weights

        return grads.flatten(), np.ones(grads.shape).flatten()

    return Loss


def Cross_entropy(alpha, return_loss=False):
    def Loss(targets, softmax_vals):
        labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
                                                                                                           softmax_vals)

        batch_size = softmax_vals.shape[0]
        outputs = softmax_vals[range(batch_size), targets]
        torch.cuda.empty_cache()
        loss_val = -torch.sum(torch.log(outputs))

        if return_loss == True:
            return loss_val

        loss_val.backward()
        grads = softmax_vals.grad.numpy()

        del softmax_vals, targets, batch_size

        return grads.flatten(), np.ones(grads.shape).flatten()

    return Loss


def Focal_loss(alpha=None, return_loss=False):
    def Loss(targets, softmax_vals):
        labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
                                                                                                           softmax_vals)

        batch_size = softmax_vals.shape[0]
        targets = torch.tensor(targets, dtype=torch.long).to(softmax_vals.device)

        # Select the probabilities corresponding to the target labels
        pt = softmax_vals[range(batch_size), targets]

        loss_val = -torch.sum(((1 - pt) ** alpha) * torch.log(pt))

        if return_loss:
            return loss_val

        loss_val.backward()
        grads = softmax_vals.grad.numpy()

        del softmax_vals, targets, batch_size

        return grads.flatten(), np.ones(grads.shape).flatten()

    return Loss


def SORD(alpha, return_loss=False, prox=False, type="max"):
    def Loss(targets, softmax_vals):
        labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
                                                                                                           softmax_vals)

        if not prox:
            phi = torch.abs(torch.arange(labels_num, device=device).view(1, -1) - targets.double().view(-1, 1))
        else:
            if type == "max":
                phi = torch.max(prox_mat) - prox_mat[targets].to(device)
            if type == "norm_max":
                phi = torch.max(norm_prox_mat) - norm_prox_mat[targets].to(device)
            if type == "norm_log":
                phi = - torch.log(norm_prox_mat[targets].to(device))
            if type == "log":
                phi = - torch.log(prox_mat[targets].to(device))
            if type == "norm_division":
                phi = 1 / (norm_prox_mat[targets].to(device))
            if type == "division":
                phi = 1 / (prox_mat[targets].to(device))

        softmax_targets = f.softmax(-alpha * phi, dim=1).to(device)

        loss_val = -1 * torch.sum(softmax_targets * torch.log(softmax_vals))

        if return_loss == True:
            return loss_val

        loss_val.backward()
        grads = softmax_vals.grad.numpy()

        del softmax_targets, softmax_vals, targets

        return grads.flatten(), np.ones(grads.shape).flatten()

    return Loss


def OLL(alpha, return_loss=False, prox=False, type="max"):
    def Loss(targets, softmax_vals):
        labels_num, prox_mat, norm_prox_mat, prox_dom, softmax_vals, targets, num_examples = change_inputs(targets,
                                                                                                           softmax_vals)

        if not prox:
            dis = torch.abs(torch.arange(labels_num, device=device).view(1, -1) - targets.double().view(-1, 1))
        else:
            if type == "max":
                dis = torch.max(prox_mat) - prox_mat[targets].to(device)
            if type == "norm_max":
                dis = torch.max(norm_prox_mat) - norm_prox_mat[targets].to(device)
            if type == "norm_log":
                dis = - torch.log(norm_prox_mat[targets].to(device))
            if type == "log":
                dis = - torch.log(prox_mat[targets].to(device))
            if type == "norm_division":
                dis = 1 / (norm_prox_mat[targets].to(device))
            if type == "division":
                dis = 1 / (prox_mat[targets].to(device))

        loss_val = -1 * torch.sum(torch.log(1 - softmax_vals) * (dis ** alpha))

        if return_loss == True:
            return loss_val

        loss_val.backward()
        grads = softmax_vals.grad.numpy()

        del dis, softmax_vals, targets

        return grads.flatten(), np.ones(grads.shape).flatten()

    return Loss


def call_loss(name, alpha=1, return_loss=False):
    if name == "Cross_Entropy":
        return Cross_entropy(alpha, return_loss=return_loss)
    if name == "Focal_loss":
        return Focal_loss(alpha, return_loss=return_loss)
    if name == "SORD":
        return SORD(alpha, return_loss=return_loss)
    if name == "OLL":
        return OLL(alpha, return_loss=return_loss)
    if name == "SORD_prox":
        return SORD(alpha, return_loss=return_loss, prox=True)
    if name == "OLL_prox":
        return OLL(alpha, return_loss=return_loss, prox=True)
    if name == "Wocel":
        return WOCEL_alpha(alpha, return_loss=return_loss)
    if name == "Accumulating":
        return WOCEL_accumulating(alpha, return_loss=return_loss)
    if name == "Accumulating_SORD_prox":
        return accumulating_with_sord_prox(alpha, return_loss=return_loss)
    if name == "Accumulating_SORD_norm_prox":
        return accumulating_with_sord_prox(alpha, return_loss=return_loss, type="norm_max")
    if name == "Accumulating_SORD":
        return accumulating_with_sord(alpha, return_loss=return_loss)
    types = ["max","norm_max", "norm_log", "log", "norm_division", "division"]
    for type in types:
        if name == "SORD_"+type:
            return SORD(alpha, return_loss=return_loss, type=type, prox=True)
        if name == "OLL_"+type:
            return OLL(alpha, return_loss=return_loss, type=type, prox=True)
        if name == "Accumulating_SORD_prox_"+type:
            return accumulating_with_sord_prox(alpha, return_loss=return_loss, type=type)