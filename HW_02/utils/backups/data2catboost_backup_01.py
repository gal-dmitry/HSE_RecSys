from typing import Generator
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold





class Dataset(ABC):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def merge(self, dataset: "Dataset", on: str, how: str) -> "Dataset":
        self._df = self._df.merge(dataset.pandas_df, on=on, how=how)
        return self

    def to_category(self, columns: List[str]):
        for column in columns:
            self._df[column] = self._df[column].astype("category")

    def fill_na_category(self, columns: List[str]):
        for column in columns:
            self._df[column] = self._df[column].cat.add_categories("<UNK>").fillna(value="<UNK>")

    @property
    def pandas_df(self, indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        if indices is not None:
            return self._df.iloc[indices]

        return self._df

    def __len__(self) -> int:
        return len(self._df)





class TrainDataset(Dataset):
    
    def reduce_by_members(self, size: int, inplace: bool = False) -> "TrainDataset":
        if not inplace:
            dataset = TrainDataset(self._df).reduce_by_members(size, inplace=True)
            return dataset

        self._df = self._df.groupby("msno").head(size).reset_index(drop=True)
        return self

    def remove_by_mask(self, mask, inplace: bool = False) -> "TrainDataset":
        if not inplace:
            dataset = TrainDataset(self._df).remove_by_mask(mask, inplace=True)
            return dataset

        self._df = self._df[~mask]
        return self

    def sort_by(self, column: str, inplace: bool = False) -> "TrainDataset":
        if not inplace:
            dataset = TrainDataset(self._df).sort_by(column, inplace=True)
            return dataset

        self._df = self._df.sort_values(by="msno")
        return self

    def split(self, n_splits: int) -> Generator:
        group_kfold = GroupKFold(n_splits=n_splits)

        df_sorted = self._df.sort_values(by="msno")
        data = df_sorted.drop("target", axis=1)
        groups = data.msno.cat.codes.to_numpy()

        for train_index, test_index in group_kfold.split(data, groups=groups):
            train_dataset = TrainDataset(self._df.iloc[train_index])
            test_dataset = TrainDataset(self._df.iloc[test_index])
            yield train_dataset, test_dataset

    @property
    def queries(self) -> np.ndarray:
        return self._df.msno.cat.codes.to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self._df.target.to_numpy()

