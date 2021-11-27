from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from data2catboost import TrainDataset
from metrics import NDCG as ndcg
from metrics import AUC as auc_per_query

    
import numpy as np
from catboost import CatBoostRanker, Pool



class Model(ABC):
    @abstractmethod
    def fit(self, dataset: TrainDataset) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: TrainDataset) -> np.ndarray:
        raise NotImplementedError

    def cv_scores(self, dataset: TrainDataset, n_splits: int) -> Dict[str, List[float]]:
        metrics = {"NDCG": [], "ROC_AUC": []}

        for train_dataset, test_dataset in dataset.split(n_splits):
            self.fit(train_dataset)
            scores = self.predict(test_dataset)

            metrics["NDCG"].append(ndcg(test_dataset.queries, scores, test_dataset.labels))
            metrics["ROC_AUC"].append(auc_per_query(test_dataset.queries, scores, test_dataset.labels))

        return metrics
    

    
class CatBoostModel(Model):
    def __init__(self, loss_function: str, iterations: int, task_type: str, random_state: int):
        self._model = CatBoostRanker(loss_function=loss_function, iterations=iterations,
                                     task_type=task_type, random_state=random_state)

    def fit(self, dataset: TrainDataset) -> "CatBoostModel":
        pool = CatBoostModel.to_pool(dataset)
        self._model.fit(pool)
        return self

    def predict(self, dataset: TrainDataset) -> np.ndarray:
        pool = CatBoostModel.to_pool(dataset)
        pred = self._model.predict(pool)
        return pred

    @staticmethod
    def to_pool(dataset: TrainDataset) -> Pool:
        cat_features = dataset.pandas_df.select_dtypes(include=["category"]).columns.to_numpy()

        data = dataset.pandas_df.drop("target", axis=1)
        label = dataset.pandas_df.target.to_numpy()
        group_id = dataset.pandas_df.msno.cat.codes.to_numpy()

        pool = Pool(data=data, label=label, group_id=group_id,
                    cat_features=cat_features, has_header=True)

        return pool