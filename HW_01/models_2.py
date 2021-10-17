import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import namedtuple


def create_positives_dataset(user_item_csr, user_ids):
    PositiveSample = namedtuple('PositiveSample', ('user_id', 'movie_id'))
    D = []
    for user_id in user_ids:
        for movie_id in user_item_csr[user_id, :].indices:
            D.append(PositiveSample(user_id, movie_id))
    print(f'Dataset: {len(D)}')
    return D   

                
class MatrixFactorization:
    
    def __init__(self, rank=64, max_epochs=100, eps=1e-5, lambd=1e-5, 
                 lr=None, confidence=None, M=None, max_negatives=None, seed=42):
        self.rank = rank
        self.max_epochs = max_epochs
        self.eps = eps
        self.lambd = lambd
        self.lr = lr
        self.confidence = confidence
        self.M = M
        self.max_negatives = max_negatives
        np.random.seed(seed)
        
        
    def initialize_weights(self, ratings=None, user_item_csr=None):     
        
        if ratings is not None:
            self.users = ratings['user_id'].unique() 
            self.movies = ratings['movie_id'].unique()
            print(f'Users: {len(self.users)} | Movies: {len(self.movies)} | Ratings: {len(ratings)}')
        
        elif user_item_csr is not None:
            self.cx = sp.coo_matrix(user_item_csr)
            self.users = np.unique(self.cx.row)
            self.movies = np.unique(self.cx.col)
            print(f'Users: {len(self.users)} | Movies: {len(self.movies)} | Ratings: {user_item_csr.getnnz()}')
        
        else:
            raise Error
            
        user_ids_size = np.max(self.users) + 1
        movie_ids_size = np.max(self.movies) + 1
        init_limit = 1 / np.sqrt(self.rank)
        
        self.user_embeddings = np.random.uniform(0, init_limit, (user_ids_size, self.rank))
        self.movie_embeddings = np.random.uniform(0, init_limit, (movie_ids_size, self.rank))
        
        if ratings is not None:
            self.user_biases = np.zeros(user_ids_size)
            self.movie_biases = np.zeros(movie_ids_size)
            self.gen_bias = np.mean(ratings['rating'])
    

"""
Models
"""
class SGD_bias(MatrixFactorization):
    
    def __init__(self, rank=64, lr=1e-3, max_epochs=100, eps=1e-5, lambd=1e-5):
        super().__init__(rank=rank, lr=lr, max_epochs=max_epochs, eps=eps, lambd=lambd)
        
    
    def fit(self, ratings):
        ratings_index = np.array(ratings.index)      
        super().initialize_weights(ratings=ratings)
                
        for epoch in range(self.max_epochs):

            np.random.shuffle(ratings_index)
            mse_sum = 0
            mse_count = 0
            
            for rating_index in tqdm(ratings_index):
                rating_row = ratings.iloc[rating_index]
                user_id, movie_id, rating  = rating_row['user_id'], rating_row['movie_id'], rating_row['rating']
                
                user_embedding = self.user_embeddings[user_id]
                movie_embedding = self.movie_embeddings[movie_id]
                user_bias = self.user_biases[user_id]
                movie_bias = self.movie_biases[movie_id]
                
                error = (np.dot(user_embedding, movie_embedding) + user_bias + movie_bias + self.gen_bias) - rating
                mse_sum += error ** 2
                mse_count += 1
                
                user_embedding_delta = self.lr * (error * movie_embedding + self.lambd * user_embedding)
                movie_embedding_delta = self.lr * (error * user_embedding + self.lambd * movie_embedding)
                user_bias_delta = self.lr * (error + self.lambd * user_bias)
                movie_bias_delta = self.lr * (error + self.lambd * movie_bias)
                
                self.user_embeddings[user_id] -= user_embedding_delta
                self.movie_embeddings[movie_id] -= movie_embedding_delta
                self.user_biases[user_id] -= user_bias_delta
                self.movie_biases[movie_id] -= movie_bias_delta
            
            mse = mse_sum / mse_count
            print(f'Epoch: {epoch} | MSE_mean={mse}')
            if mse < self.eps:
                break     

        
class ALS(MatrixFactorization):
    
    def __init__(self, rank=64, max_epochs=100, eps=1e-5, lambd=1e-5, confidence=10):
        super().__init__(rank=rank, max_epochs=max_epochs, eps=eps, lambd=lambd, confidence=confidence)
        
    
    def update_user_embs(self, user_item_csr):  
        movie_dot_movie = (self.movie_embeddings * self.movie_embeddings).sum(axis=1, keepdims=True)       
        for user_id in tqdm(self.users):
            x_i = np.array(user_item_csr[user_id, :].todense()).reshape((-1, 1))
            c = 1 + self.confidence * x_i
            left_side = (c * movie_dot_movie + self.lambd).sum(axis=0)
            right_side = (c * x_i * self.movie_embeddings).sum(axis=0)
            self.user_embeddings[user_id] = right_side / left_side
        
        
    def update_movie_embs(self, user_item_csr):  
        user_dot_user = (self.user_embeddings * self.user_embeddings).sum(axis=1, keepdims=True)             
        for movie_id in tqdm(self.movies):
            x_j = np.array(user_item_csr[:, movie_id].todense()).reshape((-1, 1))
            c = 1 + self.confidence * x_j
            left_side = (c * user_dot_user + self.lambd).sum(axis=0)
            right_side = (c * x_j * self.user_embeddings).sum(axis=0)
            self.movie_embeddings[movie_id] = right_side / left_side
                
    
    def compute_loss(self):
        mse_sum = 0
        mse_count = 0

        for user_id, movie_id, rating in zip(self.cx.row, self.cx.col, self.cx.data):
            error = 1 - np.dot(self.user_embeddings[user_id], self.movie_embeddings[movie_id])
            mse_sum += error**2
            mse_count += 1

        return mse_sum / mse_count
            
            
    def fit(self, user_item_csr):     
        super().initialize_weights(user_item_csr=user_item_csr)
        for epoch in range(self.max_epochs):            
            self.update_user_embs(user_item_csr) if epoch % 2 == 0 else self.update_movie_embs(user_item_csr)
            mse = self.compute_loss()
            print(f'Epoch: {epoch + 1} | MSE={mse}')
            if mse < self.eps:
                break
                       


class BPR(MatrixFactorization):
    
    def __init__(self, rank=64, lr=1e-3, max_epochs=100, eps=1e-5, lambd=1e-5):
        super().__init__(rank=rank, lr=lr, max_epochs=max_epochs, eps=eps, lambd=lambd)

        
    def update_weights(self, user_id, prefers_id, over_id):
        ### product
        x_ui = np.dot(self.user_embeddings[user_id], self.movie_embeddings[prefers_id])
        x_uj = np.dot(self.user_embeddings[user_id], self.movie_embeddings[over_id])
        x_uij = x_ui - x_uj

        ### grad
        user_grad = self.movie_embeddings[prefers_id] - self.movie_embeddings[over_id]
        prefers_grad = self.user_embeddings[user_id]
        over_grad = -self.user_embeddings[user_id]

        ### delt
        sigm = np.exp(-x_uij) / (1 + np.exp(-x_uij))

        user_delt = self.lr * (-sigm * user_grad - self.lambd * self.user_embeddings[user_id])
        prefers_delt = self.lr * (-sigm * prefers_grad - self.lambd * self.movie_embeddings[prefers_id])
        over_delt = self.lr * (-sigm * over_grad - self.lambd * self.movie_embeddings[over_id])

        ### update
        self.user_embeddings[user_id] -= user_delt
        self.movie_embeddings[prefers_id] -= prefers_delt
        self.movie_embeddings[over_id] -= over_delt
        
        return sigm
        
        
    def fit(self, user_item_csr):
        super().initialize_weights(user_item_csr=user_item_csr)
        D = create_positives_dataset(user_item_csr, self.users)
        prev_logloss = None
                
        for epoch in range(self.max_epochs):
            np.random.shuffle(D)
            logloss_sum = 0
            logloss_count = 0
            
            for user_id, prefers_id in tqdm(D):
                while True:
                    over_id = np.random.choice(self.movies, 1)[0]
                    if user_item_csr[user_id, over_id] == 0:
                        break
                
                sigm = self.update_weights(user_id, prefers_id, over_id)

                logloss_sum += np.log(sigm)
                logloss_count += 1
            
            logloss = logloss_sum / logloss_count
            print(f'Epoch: {epoch + 1} | Logloss={logloss}')
        
            if prev_logloss is not None and np.abs(logloss - prev_logloss) / np.abs(prev_logloss) < self.eps:
                break
            else:
                prev_logloss = logloss

                
class WARP(MatrixFactorization):
    
    def __init__(self, rank=64, lr=1e-3, max_epochs=100, eps=1e-5, lambd=1e-5, M=1, max_negatives=10):
        super().__init__(rank=rank, lr=lr, max_epochs=max_epochs, eps=eps, lambd=lambd, M=M, max_negatives=max_negatives)
            
    
    def update_weights(self, n, score_prefers, score_over, user_id, prefers_id, over_id):
        
        rank_approx = np.floor(self.max_negatives / n)
        rank_loss = np.log(rank_approx)

        loss = rank_loss * (self.M + score_over - score_prefers)

        user_grad = self.movie_embeddings[over_id] - self.movie_embeddings[prefers_id]
        prefers_grad = -self.user_embeddings[user_id]
        over_grad = self.user_embeddings[user_id]

        user_delt = self.lr * (rank_loss * user_grad + self.lambd * self.user_embeddings[user_id])
        prefers_delt = self.lr * (rank_loss * prefers_grad + self.lambd * self.movie_embeddings[prefers_id])
        over_delt = self.lr * (rank_loss * over_grad + self.lambd * self.movie_embeddings[over_id])

        self.user_embeddings[user_id] -= user_delt
        self.movie_embeddings[prefers_id] -= prefers_delt
        self.movie_embeddings[over_id] -= over_delt
        
        return loss
    
        
    def fit(self, user_item_csr):
        super().initialize_weights(user_item_csr=user_item_csr)
        D = create_positives_dataset(user_item_csr, self.users)
        prev_loss = None
        
        for epoch in range(self.max_epochs):
            np.random.shuffle(D)
            loss_sum = 0
            loss_count = 0
            
            for user_id, prefers_id in tqdm(D):
                score_prefers = np.dot(self.user_embeddings[user_id], self.movie_embeddings[prefers_id])
                
                for n in range(1, self.max_negatives + 1):
                    while True:
                        over_id = np.random.choice(self.movies, 1)[0]
                        if user_item_csr[user_id, over_id] == 0:
                            break
                            
                    score_over = np.dot(self.user_embeddings[user_id], self.movie_embeddings[over_id])
                    
                    loss = 0.0
                    if self.M + score_over - score_prefers > 0:
                        loss = self.update_weights(n, score_prefers, score_over, user_id, prefers_id, over_id)
                    loss_sum += loss                        
                    loss_count += 1
                    if loss:
                        break  
                        
            loss = loss_sum / loss_count if loss_count else 0.0
            print(f'Epoch: {epoch + 1} | Logloss={loss}')
            
            if prev_loss is not None and np.abs(loss - prev_loss) / np.abs(prev_loss) < self.eps:
                break
            else:
                prev_loss = loss       

        
        