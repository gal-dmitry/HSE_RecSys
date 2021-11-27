import numpy as np


"""NDCG"""
def DCG(y_pred, y_true):
    assert len(y_pred) == len(y_true), "Size error"
    n = len(y_pred)
    idx = np.argsort(-y_pred)
    return sum(y_true[idx] / np.log2(1 + np.arange(1, n + 1)))


def IDCG(y_pred, y_true):
    assert len(y_pred) == len(y_true), "Size error"
    n = len(y_pred)
    rev_y_true = -np.sort(-y_true)
    return sum(rev_y_true / np.log2(1 + np.arange(1, n + 1)))


def NDCG(X, Y_pred, Y_true):
    assert len(X) == len(Y_pred) == len(Y_true), "Size error"
    X_unique = np.unique(X)
    _NDCG = []

    for X_idx in X_unique:
        idx = X == X_idx 
        _IDCG = IDCG(Y_pred[idx], Y_true[idx])
        if _IDCG > 0:
            _DCG= DCG(Y_pred[idx], Y_true[idx])
            _NDCG.append(_DCG/_IDCG)

    return np.array(_NDCG).mean()


"""ROC_AUC"""
def AUC(X, Y_pred, Y_true):
    assert len(X) == len(Y_pred) == len(Y_true), "Size error"
    X_unique = np.unique(X)
    _AUC = []

    for X_idx in X_unique:
        try:
            idx = X == X_idx
            _AUC.append(roc_auc_score(Y_true[idx], Y_pred[idx]))
        except:
            pass

    return np.array(_AUC).mean()

