import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(labels_true, labels_pred):
    labels_true = np.asarray(labels_true, dtype=np.int64)
    labels_pred = np.asarray(labels_pred, dtype=np.int64)
    size = max(labels_true.max(initial=0), labels_pred.max(initial=0)) + 1
    confusion = np.zeros((size, size), dtype=np.int64)
    np.add.at(confusion, (labels_pred, labels_true), 1)
    row, col = linear_sum_assignment(confusion.max() - confusion)
    return float(confusion[row, col].sum() / len(labels_true))


def evaluate_clustering(labels_true, labels_pred):
    return {
        "acc": clustering_accuracy(labels_true, labels_pred),
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
    }

