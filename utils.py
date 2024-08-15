
import warnings
import pandas as pd
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from collections import Counter


# def find_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# device = find_device()
device = "cpu"

def create_prox_mat(dist_dict, inv=True):
    labels = list(dist_dict.keys())
    labels.sort()
    denominator = sum(dist_dict.values())
    prox_mat = np.zeros([len(labels), len(labels)])
    for label1 in labels:
        for label2 in labels:
            label1 = int(label1)
            label2 = int(label2)
            minlabel, maxlabel = min(label1, label2), max(label1, label2)
            numerator = dist_dict[label1] / 2
            if minlabel == label1:  # Above the diagonal
                for tmp_label in range(minlabel + 1, maxlabel + 1):
                    numerator += dist_dict[tmp_label]
            else:  # Under the diagonal
                for tmp_label in range(maxlabel - 1, minlabel - 1, -1):
                    numerator += dist_dict[tmp_label]
            if inv:
                prox_mat[label1 - 1][label2 - 1] = (-np.log(numerator / denominator)) ** -1
            else:
                prox_mat[label1 - 1][label2 - 1] = -np.log(numerator / denominator)
    return torch.tensor(prox_mat)


def create_prox_dom(prox_mat):
    labels_num = prox_mat.shape[0]
    prox_dom_mat = []
    for label in range(prox_mat.shape[0]):
        label_prox_ordered = reversed(prox_mat[:, label].argsort()).tolist()
        prox_dom_mat.append(torch.tensor([[0.0 if label_prox_ordered.index(ind_axe) <
                                                  label_prox_ordered.index(ind_inner) else 1.0
                                           for ind_inner in range(labels_num)]
                                          for ind_axe in range(labels_num)]))
    return torch.stack(prox_dom_mat)


def CrossEntropyLoss(outputs, targets):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = outputs[range(batch_size), targets.detach().numpy()]
    del targets, batch_size
    torch.cuda.empty_cache()
    return -torch.sum(torch.log(outputs)) / num_examples


def WOCEL(targets, softmax_vals):
    alpha = 0.75
    t=targets.detach().numpy()
    labels_num = int(np.max(t)) + 1
    d = Counter(t)
    dist_dict = {int(k): int(v) for k, v in d.items()}
    prox_mat = create_prox_mat(dist_dict, inv=False)
    norm_prox_mat = f.normalize(prox_mat, p=1, dim=0)
    prox_dom = create_prox_dom(prox_mat).double()
    # The following two lines were added for converting the numpy arrays to torch tensors
    # softmax_vals = torch.tensor(softmax_vals, requires_grad=True)
    # targets = torch.tensor(targets)
    targets.requires_grad_(False)

    num_examples = targets.shape[0]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num]))
    one_hot_target[range(num_examples), targets.long()] = 1
    one_hot_target_comp = 1 - one_hot_target
    elem_weight = norm_prox_mat[:, targets.long()].clone().t()
    wrong_mass_weight = torch.sum(one_hot_target_comp * softmax_vals, dim=1).unsqueeze(dim=1) * one_hot_target_comp
    mass_weights = alpha * one_hot_target + (1 - alpha) * wrong_mass_weight * elem_weight
    sum_coeff = torch.matmul(softmax_vals.unsqueeze(dim=1).double(),
                             prox_dom[targets.long()].double()).double().squeeze(dim=1)

    # mass_weights = mass_weights.detach()
    # sum_coeff = sum_coeff.detach()

    loss_val = -1 * torch.sum(mass_weights * torch.log(softmax_vals / sum_coeff)) / num_examples

    return loss_val


def cem2(y_pred, y_true):
    dist_dict = dict(pd.DataFrame(y_true)[0].value_counts())
    prox_mat = create_prox_mat(dist_dict, inv=False)
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['pred_prox'] = df.apply(lambda row:
                               prox_mat[int(row["y_pred"])][int(row["y_true"])],
                               axis=1)
    df['truth_prox'] = df.apply(lambda row:
                                prox_mat[int(row["y_true"])][int(row["y_true"])],
                                axis=1)
    return df['pred_prox'].sum() / df['truth_prox'].sum()

    # custom eval matrix


def cem(y_pred, dtrain):
    y_true = dtrain.get_label().astype(int)
    y_pred = np.argmax(y_pred, axis=1)
    dist_dict = dict(pd.DataFrame(y_true)[0].value_counts())
    prox_mat = create_prox_mat(dist_dict, inv=False)
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['pred_prox'] = df.apply(lambda row:
                               prox_mat[int(row["y_pred"])][int(row["y_true"])],
                               axis=1)
    df['truth_prox'] = df.apply(lambda row:
                                prox_mat[int(row["y_true"])][int(row["y_true"])],
                                axis=1)
    return "cem", df['pred_prox'].sum() / df['truth_prox'].sum()

