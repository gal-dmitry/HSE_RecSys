import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import namedtuple


def form_pos_data(user_item_csr, user_ids):
    pos_sample = namedtuple('pos_sample', ('user_id', 'movie_id'))
    data = []
    for user_id in user_ids:
        for movie_id in user_item_csr[user_id, :].indices:
            data.append(pos_sample(user_id, movie_id))
    return data   

                
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
            
        user_size = np.max(self.users) + 1
        movie_size = np.max(self.movies) + 1
        uni_border = 1 / np.sqrt(self.rank)
        
        self.user_embeddings = np.random.uniform(0, uni_border, (user_size, self.rank))
        self.movie_embeddings = np.random.uniform(0, uni_border, (movie_size, self.rank))
        
        if ratings is not None:
            self.user_biases = np.zeros(user_size)
            self.movie_biases = np.zeros(movie_size)
            self.mu_bias = np.mean(ratings['rating'])
    

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
                
                user_emb = self.user_embeddings[user_id]
                movie_emb = self.movie_embeddings[movie_id]
                user_bias = self.user_biases[user_id]
                movie_bias = self.movie_biases[movie_id]
                
                error = (np.dot(user_emb, movie_emb) + user_bias + movie_bias + self.mu_bias) - rating
                mse_sum += error ** 2
                mse_count += 1
                
                user_emb_delta = self.lr * (error * movie_emb + self.lambd * user_emb)
                movie_emb_delta = self.lr * (error * user_emb + self.lambd * movie_emb)
                user_bias_delta = self.lr * (error + self.lambd * user_bias)
                movie_bias_delta = self.lr * (error + self.lambd * movie_bias)
                
                self.user_embeddings[user_id] -= user_emb_delta
                self.movie_embeddings[movie_id] -= movie_emb_delta
                self.user_biases[user_id] -= user_bias_delta
                self.movie_biases[movie_id] -= movie_bias_delta
            
            mse = mse_sum / mse_count
            print(f'Epoch: {epoch} | MSE={mse}')
            if mse < self.eps:
                break     

        
class ALS(MatrixFactorization):
    
    def __init__(self, rank=64, max_epochs=100, eps=1e-5, lambd=1e-5, confidence=10):
        super().__init__(rank=rank, max_epochs=max_epochs, eps=eps, lambd=lambd, confidence=confidence)
        
    
    def update_embs(self, user_item_csr, mode='movie'):         
        lst = self.movies if mode=='movie' else self.users
        embs = self.user_embeddings if mode=='movie' else self.movie_embeddings
        dot = (embs * embs).sum(axis=1, keepdims=True)  
        
        for _id in tqdm(lst):
            _x = user_item_csr[:, _id] if mode=='movie' else user_item_csr[_id, :]
            x = np.array(_x.todense()).reshape((-1, 1))
            c = 1 + self.confidence * x
            nom = (c * x * embs).sum(axis=0)
            denom = (c * dot + self.lambd).sum(axis=0)
            
            if mode=='movie':
                self.movie_embeddings[_id] = nom / denom
            else:
                self.user_embeddings[_id] = nom / denom
       
    
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
            self.update_embs(user_item_csr, mode='user')
            self.update_embs(user_item_csr, mode='movie')
            mse = self.compute_loss()
            print(f'Epoch: {epoch + 1} | MSE={mse}')
            if mse < self.eps:
                break
                       


class BPR(MatrixFactorization):
    
    def __init__(self, rank=64, lr=1e-3, max_epochs=100, eps=1e-5, lambd=1e-5):
        super().__init__(rank=rank, lr=lr, max_epochs=max_epochs, eps=eps, lambd=lambd)

        
    def update_weights(self, user_id, prefs_id, blank_id):
        x_ui = np.dot(self.user_embeddings[user_id], self.movie_embeddings[prefs_id])
        x_uj = np.dot(self.user_embeddings[user_id], self.movie_embeddings[blank_id])
        x_uij = x_ui - x_uj

        user_grad = self.movie_embeddings[prefs_id] - self.movie_embeddings[blank_id]
        prefs_grad = self.user_embeddings[user_id]
        blank_grad = -self.user_embeddings[user_id]

        sigm = np.exp(-x_uij + 1e-8) / (1 + np.exp(-x_uij + 1e-8))

        user_delt = self.lr * (-sigm * user_grad - self.lambd * self.user_embeddings[user_id])
        prefs_delt = self.lr * (-sigm * prefs_grad - self.lambd * self.movie_embeddings[prefs_id])
        blank_delt = self.lr * (-sigm * blank_grad - self.lambd * self.movie_embeddings[blank_id])

        self.user_embeddings[user_id] -= user_delt
        self.movie_embeddings[prefs_id] -= prefs_delt
        self.movie_embeddings[blank_id] -= blank_delt
        
        return np.log(sigm)
        
        
    def fit(self, user_item_csr):
        super().initialize_weights(user_item_csr=user_item_csr)
        data = form_pos_data(user_item_csr, self.users)
        prev_loss = None
                
        for epoch in range(self.max_epochs):
            np.random.shuffle(data)
            loss_sum = 0
            loss_count = 0
            
            for user_id, prefs_id in tqdm(data):
                while True:
                    blank_id = np.random.choice(self.movies, 1)[0]
                    if user_item_csr[user_id, blank_id] == 0:
                        break
                
                loss = self.update_weights(user_id, prefs_id, blank_id)
                loss_sum += loss
                loss_count += 1
            
            loss = loss_sum / loss_count
            print(f'Epoch: {epoch + 1} | Loss={loss}')
        
            if prev_loss is not None and np.abs(loss - prev_loss) / np.abs(prev_loss) < self.eps:
                break
            else:
                prev_loss = loss

                
class WARP(MatrixFactorization):
    
    def __init__(self, rank=64, lr=1e-3, max_epochs=100, eps=1e-5, lambd=1e-5, M=1, max_negatives=10):
        super().__init__(rank=rank, lr=lr, max_epochs=max_epochs, eps=eps, lambd=lambd, M=M, max_negatives=max_negatives)
            
    
    def update_weights(self, n, score_prefs, score_blank, user_id, prefs_id, blank_id):
        
        rank_approx = np.floor(self.max_negatives / n)
        rank_loss = np.log(rank_approx)

        loss = rank_loss * (self.M + score_blank - score_prefs)

        user_grad = self.movie_embeddings[blank_id] - self.movie_embeddings[prefs_id]
        prefs_grad = -self.user_embeddings[user_id]
        blank_grad = self.user_embeddings[user_id]

        user_delt = self.lr * (rank_loss * user_grad + self.lambd * self.user_embeddings[user_id])
        prefs_delt = self.lr * (rank_loss * prefs_grad + self.lambd * self.movie_embeddings[prefs_id])
        blank_delt = self.lr * (rank_loss * blank_grad + self.lambd * self.movie_embeddings[blank_id])

        self.user_embeddings[user_id] -= user_delt
        self.movie_embeddings[prefs_id] -= prefs_delt
        self.movie_embeddings[blank_id] -= blank_delt
        
        return loss
    
        
    def fit(self, user_item_csr):
        super().initialize_weights(user_item_csr=user_item_csr)
        data = form_pos_data(user_item_csr, self.users)
        prev_loss = None
        
        for epoch in range(self.max_epochs):
            np.random.shuffle(data)
            loss_sum = 0
            loss_count = 0
            
            for user_id, prefs_id in tqdm(data):
                score_prefs = np.dot(self.user_embeddings[user_id], self.movie_embeddings[prefs_id])
                
                for n in range(1, self.max_negatives + 1):
                    while True:
                        blank_id = np.random.choice(self.movies, 1)[0]
                        if user_item_csr[user_id, blank_id] == 0:
                            break
                            
                    score_blank = np.dot(self.user_embeddings[user_id], self.movie_embeddings[blank_id])
                    if self.M + score_blank - score_prefs > 0:
                        loss = self.update_weights(n, score_prefs, score_blank, user_id, prefs_id, blank_id)
                        loss_sum += loss
                        loss_count += 1
                        break  
                        
                else:
                    loss_count += 1        
                        
            loss = loss_sum / loss_count
            print(f'Epoch: {epoch + 1} | Loss={loss}')
            
            if prev_loss is not None and np.abs(loss - prev_loss) / np.abs(prev_loss) < self.eps:
                break
            else:
                prev_loss = loss       

        
        