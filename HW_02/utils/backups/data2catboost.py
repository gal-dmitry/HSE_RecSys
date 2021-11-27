import numpy as np
import pandas as pd
from typing import Generator
from sklearn.model_selection import GroupKFold



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
