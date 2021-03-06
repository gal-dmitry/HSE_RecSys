import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

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
        cat_features = dataset._df.select_dtypes(include=["category"]).columns.to_numpy()
        data = dataset._df.drop("target", axis=1)
        label = dataset._df.target.to_numpy()
        group_id = dataset._df.msno.cat.codes.to_numpy()
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
    

    
class EmbeddingModel_v1:
    
    def __init__(self, 
                 embedding_dim=100, 
                 random_state=42,
                 min_count=5):
        
        self.min_count = min_count
        self._random_state = random_state
            
        self._embedding_dim = embedding_dim
        self._user_emb = None
        self._word2vec = None
        
        self._user_encoder = LabelEncoder()
        self._users = set()

        
    def has_item(self, item):
        return item in self._word2vec.wv
#         return item in self._word2vec.wv.index_to_key
    
    def has_user(self, user):
        return user in self._users
    
    
    def get_mask(self, users, items):
        return np.array([self.has_user(user) and self.has_item(item) for user, item in zip(users, items)])
        
        
    def get_sentences(self, df):
        sessions = {}
        for msno in df.msno.unique():
            song = df[df.msno == msno].song_id.to_list()
            sessions[msno] = song
            
        return [values for values in sessions.values() if len(values) > 0]
    
    
    def get_positives(self, df):
        positives = {}
    
        pos_df = df[df.target == 1]
        for msno in pos_df.msno.unique():
            song = pos_df[pos_df.msno == msno].song_id.to_list()
            positives[msno] = song
        
        return positives
        
        
    def get_item_emb(self, items):
        return self._word2vec.wv[items]

    
    def get_user_emb(self, users):
#         print(users)
        users_encoded = self._user_encoder.transform(users)
#         print(users_encoded)
        return self._user_emb[users_encoded]

    
    def get_user_emb(self, users):
#         print(users)
        users_encoded = self._user_encoder.transform(users)
#         print(users_encoded)
        return self._user_emb[users_encoded]
    
    
    
    def fit_items(self, df):
        print("Fit items...")        
        self._word2vec = Word2Vec(vector_size=self._embedding_dim, 
                                  window=5,
                                  min_count=self.min_count,
                                  seed=self._random_state)
        
        sentences = self.get_sentences(df)
#         print(sentences)
        self._word2vec.build_vocab(sentences)
        self._word2vec.train(sentences, total_examples=self._word2vec.corpus_count, epochs=10)

        
    def fit_users(self, df):

        positives = self.get_positives(df)
#         print(f"positives: {positives}")
        
        self._user_encoder.fit(list(positives.keys()))
        self._users = set(self._user_encoder.classes_)
        self._user_emb = np.zeros((len(self._users), self._embedding_dim))

        for user, user_pos in tqdm(positives.items(), 
                                   total=len(positives),
                                   disable=False,  
                                   desc="Fit users"):
            
            user_pos = [pos for pos in user_pos if self.has_item(pos)]
            
            if len(user_pos) > 0:
#                 print(f"len(user_pos)")
                user_encoded = self._user_encoder.transform([user])[0]
                self._user_emb[user_encoded] = self.get_item_emb(user_pos).mean(axis=0)

    
    def fit(self, dataset):
        self.fit_items(dataset._df)
        self.fit_users(dataset._df)
        return self
        
        
    def predict(self, dataset):
        
        users = dataset._df["msno"].to_numpy()
#         print(f"users: {users}")
        items = dataset._df["song_id"].to_numpy()
#         print(f"items: {items}")
        
        mask = self.get_mask(users, items)
        not_mask = [not a for a in mask]
        
        if True in mask:
            user_emb_s = self.get_user_emb(users[mask])
            item_emb_s = self.get_item_emb(items[mask])
        else:
            return np.zeros(len(dataset))
        
        scores = np.zeros(len(dataset))
        scores[mask] = np.sum(user_emb_s * item_emb_s, axis=1)        
        return scores


    
class EmbeddingModel_v2:
    
    def __init__(self):        
        self._word2vec = Word2Vec(min_count=2,
                                  window=10,
                                  vector_size=30,
                                  negative=18,
                                  sg=1)

    
    def fit(self, dataset):
        self.userembed = {}
        self.song_sentences = defaultdict(list)
        
        df = dataset._df
        
        total=df[df['target'] == 1].shape[0]
        for i, row in tqdm(df[df['target'] == 1].iterrows(), total=total):
            key = f"{row['msno']}"
            value = f"{row['artist_name']}"
            self.song_sentences[key].append(value)
            
        sentences = list(self.song_sentences.values())
        
        self._word2vec.build_vocab(sentences)
        self._word2vec.train(sentences, total_examples=self._word2vec.corpus_count, epochs=10)
                
        
    def get_similars(self, item):
        self._word2vec.init_sims(replace=True)
        return self._word2vec.wv.most_similar(positive=item)

    
    def get_user_emb(self, row):
        user_name = f"{row['msno']}"
        if user_name in self.userembed:
            return self.userembed[user_name]

        embedding = np.zeros(30,)
        for song in self.song_sentences[user_name]:
            if song in self._word2vec.wv.key_to_index:
                embedding += self._word2vec.wv.get_vector(song)
        self.userembed[user_name] = embedding
        return embedding

    
    def get_item_emb(self, row):
        song_name = f"{row['artist_name']}"
        if song_name in self._word2vec.wv.key_to_index:
            return self._word2vec.wv.get_vector(song_name)
        else:
            return np.zeros(30,)
    
    
    def predict(self, dataset):
        df = dataset._df
        user_emb = np.vstack(df.apply(self.get_user_emb, axis=1))
        item_emb = np.vstack(df.apply(self.get_item_emb, axis=1))
        scores = (item_emb * user_emb).mean(axis=1)
        return scores
    
    
    
class StackModel:
    
    def __init__(self, 
                 loss_function="YetiRank", 
                 iterations=150, 
                 task_type="CPU", 
                 random_state=42,
                 embedding_dim=100,
                 min_count=5):
        
        self._cat_model = CatBoostModel(loss_function=loss_function, 
                                         iterations=iterations,
                                         task_type=task_type, 
                                         random_state=random_state)
                                             
        self._emb_model = EmbeddingModel_v1(embedding_dim=embedding_dim,
                                            random_state=random_state,
                                            min_count=min_count)
                    
            
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
def prepare_data(train_dataset_sm, split=1000):

    train_dataset_sm_sort = train_dataset_sm.sort_by("msno")
    indices = np.random.permutation(len(train_dataset_sm_sort))
    
    train_indices, test_indices = train_test_split(indices, test_size=split)
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



class ShapValuesCatBoost:
    
    def __init__(self,      
                 loss_function="YetiRank", 
                 iterations=150, 
                 task_type="CPU", 
                 random_state=42,
                 min_count=5):       
        
        self.cat_model =  CatBoostRanker(loss_function=loss_function, 
                                        iterations=iterations,
                                        task_type=task_type, 
                                        random_state=random_state)
                            
    
    def cat_shap(self, pooled_X_train, pooled_X_test):
        self.cat_model.fit(pooled_X_train)
        shap_values = self.cat_model.get_feature_importance(pooled_X_test, 
                                                            type="ShapValues", 
                                                            shap_calc_type="Exact")    
        return shap_values[:, :-1]
    
    
    def fit(self, train_dataset_sm_sort, split=1000):
        X_train, X_test = prepare_data(train_dataset_sm_sort, split=split) 
        X_train_df = X_train._df.reset_index(drop=True)
        X_test_df = X_test._df.reset_index(drop=True)
        pooled_X_train, pooled_X_test = to_pool(X_train_df), to_pool(X_test_df)
        self.shap_values = self.cat_shap(pooled_X_train, pooled_X_test)
        self.X_test_df = X_test_df.drop("target", axis=1)
    
    
    def show_shap_values(self):
        shap.summary_plot(self.shap_values, self.X_test_df)


        
class ShapValuesStackModel:
    
    def __init__(self,      
                 loss_function="YetiRank", 
                 iterations=150, 
                 task_type="CPU", 
                 random_state=42,
                 embedding_dim=100,
                 min_count=5):       
        
        self.cat_model =  CatBoostRanker(loss_function=loss_function, 
                                        iterations=iterations,
                                        task_type=task_type, 
                                        random_state=random_state)
                                             
        self.emb_model = EmbeddingModel_v1(embedding_dim=embedding_dim,
                                           random_state=random_state,
                                           min_count=min_count)        
                    

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
    
    
    def fit(self, train_dataset_sm_sort, split=1000):
        
        X_train, X_test = prepare_data(train_dataset_sm_sort)
        X_train_df, X_test_df = self.emb_similarity(X_train, X_test)
        
        pooled_X_train, pooled_X_test = to_pool(X_train_df), to_pool(X_test_df)
    
        self.shap_values = self.cat_shap(pooled_X_train, pooled_X_test)
        self.X_test_df = X_test_df.drop("target", axis=1)
    
    
    def show_shap_values(self):
        shap.summary_plot(self.shap_values, self.X_test_df)

    