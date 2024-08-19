import pandas as pd
import torch
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

