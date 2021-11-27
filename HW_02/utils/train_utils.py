import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Generator
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold



"""
DATASET
"""
class TrainDataset():
    
    def __init__(self, df):
        self._df = df
        
    def __len__(self):
        return len(self._df)

    def split(self, n_splits):
        
        group_kfold = GroupKFold(n_splits=n_splits)

        df_sorted = self._df.sort_values(by="msno")
        data = df_sorted.drop("target", axis=1)
        groups = data.msno.cat.codes.to_numpy()

        for train_index, test_index in group_kfold.split(data, groups=groups):
            train_dataset = TrainDataset(self._df.iloc[train_index])
            test_dataset = TrainDataset(self._df.iloc[test_index])
            yield train_dataset, test_dataset

    @property
    def queries(self):
        return self._df.msno.cat.codes.to_numpy()
    
    @property
    def labels(self):
        return self._df.target.to_numpy()

    @property
    def pandas_df(self, indices=None):
        if indices is not None:
            return self._df.iloc[indices]
        return self._df

    

"""
METRCIS
"""

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



"""
MODEL
"""    
class CatBoostModel():
    
    def __init__(self, 
                 loss_function, 
                 iterations, 
                 task_type, 
                 random_state):
        
        self._model = CatBoostRanker(loss_function=loss_function, 
                                     iterations=iterations,
                                     task_type=task_type, 
                                     random_state=random_state)

        
    def fit(self, dataset):
        pool = CatBoostModel.to_pool(dataset)
        self._model.fit(pool)
        return self

    
    def predict(self, dataset):
        pool = CatBoostModel.to_pool(dataset)
        pred = self._model.predict(pool)
        return pred

    
    @staticmethod
    def to_pool(dataset):
        cat_features = dataset.pandas_df.select_dtypes(include=["category"]).columns.to_numpy()
        data = dataset.pandas_df.drop("target", axis=1)
        label = dataset.pandas_df.target.to_numpy()
        group_id = dataset.pandas_df.msno.cat.codes.to_numpy()
        pool = Pool(data=data, 
                    label=label, 
                    group_id=group_id,
                    cat_features=cat_features, 
                    has_header=True)

        return pool
    
    
    def cv_scores(self, dataset, n_splits):
        
        metrics = {"NDCG": [], "ROC_AUC": []}

#         for train_dataset, test_dataset in tqdm(dataset.split(n_splits)):
        for train_dataset, test_dataset in dataset.split(n_splits):
            self.fit(train_dataset)
            scores = self.predict(test_dataset)
            metrics["NDCG"].append(NDCG(test_dataset.queries, scores, test_dataset.labels))
            metrics["ROC_AUC"].append(AUC(test_dataset.queries, scores, test_dataset.labels))

        return metrics
    
    
    
    