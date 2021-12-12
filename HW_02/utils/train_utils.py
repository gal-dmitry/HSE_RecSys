import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Generator

import shap
from catboost import CatBoostRanker, Pool
from gensim.models import Word2Vec

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder



"""
DATASET
"""
class TrainDataset:
    
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
    
    def _add(self, name, values):
        self._df[name] = values

    def _drop(self, name):
        self._df = self._df.drop(columns=name)
    
    def reduce_by_members(self, size):
        self._df = self._df.groupby("msno").head(size).reset_index(drop=True)
        return self
    
    def sort_by(self, column, inplace=False):
        if not inplace:
            dataset = TrainDataset(self._df).sort_by(column, inplace=True)
            return dataset

        self._df = self._df.sort_values(by="msno")
        return self
    
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
METRICS
"""
def AUC_per_query(queries, scores, relevance):
    
    assert len(queries) == len(scores) == len(relevance)
    query_labels = np.unique(queries)
    aucs = []

    for query_label in query_labels:
        try:
            query_mask = queries == query_label
            query_auc = roc_auc_score(relevance[query_mask], scores[query_mask])
            aucs.append(query_auc)
        except:
            pass

    return np.array(aucs).mean()


def get_metric(model, dataset, n_splits):

    metrics = pd.DataFrame(columns=["AUC_per_query"], 
                           index=[f"split: {i+1}" for i in range(n_splits)])
                     
    for i, (train_dataset, test_dataset) in enumerate(dataset.split(n_splits)):
        print(f"--- Split: {i+1}/{n_splits} ---")
        model.fit(train_dataset)
        scores = model.predict(test_dataset)                           
        metrics.loc[f"split: {i+1}", "AUC_per_query"] = AUC_per_query(test_dataset.queries, scores, test_dataset.labels)               
    mean = pd.DataFrame(index=["avg"], data={"AUC_per_query": metrics.AUC_per_query.mean()})          
    return pd.concat([metrics, mean], axis=0)



"""
MODELS
"""    
class CatBoostModel:
    
    def __init__(self, 
                 loss_function="YetiRank", 
                 iterations=150, 
                 task_type="CPU", 
                 random_state=42):
        
        self._random_state = random_state
        self._model = CatBoostRanker(loss_function=loss_function, 
                                     iterations=iterations,
                                     task_type=task_type, 
                                     random_state=random_state)
    
    
    def to_pool(self, dataset):
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
    
    
    def fit(self, dataset):
        pool = self.to_pool(dataset)
        self._model.fit(pool)

    
    def predict(self, dataset):
        pool = self.to_pool(dataset)
        pred = self._model.predict(pool)
        return pred
    
    
    
class EmbeddingModel:
    
    def __init__(self, 
                 embedding_dim=100, 
                 random_state=42):
        
        self._random_state = random_state
            
        self._embedding_dim = embedding_dim
        self._user_emb = None
        self._word2vec = None
        
        self._user_encoder = LabelEncoder()
        self._users = set()

        
    def has_item(self, item):
        return item in self._word2vec.wv

    
    def has_user(self, user):
        return user in self._users
    
    
    def get_sentences(self, df):
        sessions = dict(df.groupby("msno").song_id.apply(list))
        return [values for values in sessions.values() if len(values) > 0]
    
    
    def get_positives(self, df):
        return dict(df[df.target == 1].groupby("msno").song_id.apply(list))
        
    
    def get_mask(self, users, items):
        return np.array([self.has_user(user) and self.has_item(item) for user, item in zip(users, items)])
        
        
    def get_item_emb(self, items):
        return self._word2vec.wv[items]

    
    def get_user_emb(self, users):
        users_encoded = self._user_encoder.transform(users)
        return self._user_emb[users_encoded]
    
    
    def fit_items(self, df):
        print("Fit items...")        
        self._word2vec = Word2Vec(vector_size=self._embedding_dim, 
                                  window=5,
#                                   min_count=5,
                                  min_count=1,
                                  seed=self._random_state)
        
        sentences = self.get_sentences(df)
        self._word2vec.build_vocab(sentences)
        self._word2vec.train(sentences, total_examples=self._word2vec.corpus_count, epochs=10)

        
    def fit_users(self, df):

        positives = self.get_positives(df)

        self._user_encoder.fit(list(positives.keys()))
        self._users = set(self._user_encoder.classes_)
        self._user_emb = np.zeros((len(self._users), self._embedding_dim))

        for user, user_pos in tqdm(positives.items(), 
                                   total=len(positives),
                                   disable=False,  
                                   desc="Fit users"):
            
            user_pos = [pos for pos in user_pos if self.has_item(pos)]
            
            if len(user_pos) > 0:
                user_encoded = self._user_encoder.transform([user])[0]
                self._user_emb[user_encoded] = self.get_item_emb(user_pos).mean(axis=0)


    def fit(self, dataset):
        self.fit_items(dataset._df)
        self.fit_users(dataset._df)
        
    
    def predict(self, dataset):
        
        users = dataset._df["msno"].to_numpy()
        items = dataset._df["song_id"].to_numpy()

        mask = self.get_mask(users, items)

        user_emb_s = self.get_user_emb(users[mask])
        item_emb_s = self.get_item_emb(items[mask])
        
        scores = np.zeros(len(dataset))
        scores[mask] = np.sum(user_emb_s * item_emb_s, axis=1)

        return scores

    
    
class StackModel:
    
    def __init__(self, 
                 loss_function="YetiRank", 
                 iterations=150, 
                 task_type="CPU", 
                 random_state=42,
                 embedding_dim=100):
        
        self._cat_model = CatBoostModel(loss_function=loss_function, 
                                         iterations=iterations,
                                         task_type=task_type, 
                                         random_state=random_state)
                                             
        self._emb_model = EmbeddingModel(embedding_dim=embedding_dim,
                                           random_state=random_state)
                    
            
    def fit(self, dataset):
        self._emb_model.fit(dataset)
        scores = self._emb_model.predict(dataset)
        dataset._add("emb_score", scores)
        self._cat_model.fit(dataset)
        dataset._drop("emb_score")

    
    def predict(self, dataset):
        scores = self._emb_model.predict(dataset)
        dataset._add("emb_score", scores)
        pred = self._cat_model.predict(dataset)
        dataset._drop("emb_score")
        return pred
    
    
"""
SHAPLEY VALUES
"""
def prepare_data(train_dataset_sm):

    train_dataset_sm_sort = train_dataset_sm.sort_by("msno")
    indices = np.random.permutation(len(train_dataset_sm_sort))
    
#     train_indices, test_indices = train_test_split(indices, test_size=1_000)
    train_indices, test_indices = train_test_split(indices, test_size=0.2)
    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)

    X_train = TrainDataset(train_dataset_sm_sort._df.iloc[train_indices])
    X_test = TrainDataset(train_dataset_sm_sort._df.iloc[test_indices])

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    return X_train, X_test


def to_pool(df): 
    
    cat_features = df.select_dtypes(include=["category"]).columns.to_numpy()
    data = df.drop("target", axis=1)
    label = df.target.to_numpy()
    group_id = df.msno.cat.codes.to_numpy()
    
    pool = Pool(data=data, label=label, group_id=group_id,
                cat_features=cat_features, has_header=True)
    return pool



class ShapValues:
    
    def __init__(self,      
                 loss_function="YetiRank", 
                 iterations=150, 
                 task_type="CPU", 
                 random_state=42,
                 embedding_dim=100):       
        
        self.cat_model =  CatBoostRanker(loss_function=loss_function, 
                                        iterations=iterations,
                                        task_type=task_type, 
                                        random_state=random_state)
                                             
        self.emb_model = EmbeddingModel(embedding_dim=embedding_dim,
                                        random_state=random_state)        
                    

    def emb_similarity(self, X_train, X_test):
        
        self.emb_model.fit(X_train)
        train_scores = self.emb_model.predict(X_train)
        test_scores = self.emb_model.predict(X_test)
        
        X_train_df = X_train._df.reset_index(drop=True)
        X_test_df = X_test._df.reset_index(drop=True)

        X_train_df["sim"] = train_scores 
        X_test_df["sim"] = test_scores
        
        return X_train_df, X_test_df
        
    
    def cat_shap(self, pooled_X_train, pooled_X_test):
        
        self.cat_model.fit(pooled_X_train)
        shap_values = self.cat_model.get_feature_importance(pooled_X_test, 
                                                           type="ShapValues", 
                                                           shap_calc_type="Exact")    
        return shap_values[:, :-1]
    
    
    def fit(self, train_dataset_sm_sort):
        
        X_train, X_test = prepare_data(train_dataset_sm_sort)
        X_train_df, X_test_df = self.emb_similarity(X_train, X_test)
        pooled_X_train, pooled_X_test = to_pool(X_train_df), to_pool(X_test_df)
    
        self.shap_values = self.cat_shap(pooled_X_train, pooled_X_test)
        self.X_test_df = X_test_df.drop("target", axis=1)
    
    
    def show_shap_values(self):
        shap.summary_plot(self.shap_values, self.X_test_df)
    
    
    
    