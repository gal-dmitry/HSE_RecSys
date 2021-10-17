import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


class RecSys:
    
    def __init__(self, users, movies, user_embeddings, movie_embeddings, 
                 movie_biases, user_biases, gen_bias):   
        
        self.users = users
        self.movies = movies       
        
        self.user_embeddings = user_embeddings
        self.movie_embeddings = movie_embeddings 
        
        self.movie_biases = movie_biases
        self.user_biases = user_biases
        self.gen_bias = gen_bias  
    
    
    def similar_items(self, movie_id, n=10):
     
        query_vector = self.movie_embeddings[movie_id].reshape(-1, 1)
        similars = []

        for _movie_id in self.movies:
            if _movie_id == movie_id:
                continue

            key_vector = self.movie_embeddings[_movie_id].reshape(-1, 1) 
            cos_dist = cosine_distance(query_vector, key_vector)
            similars.append((_movie_id, cos_dist))

        similars.sort(key=lambda pair: pair[1], reverse=False)

        return similars[:n]    

    
    def recommend(self, user_id, user_item_csr, n=10):
        predicted_ratings = []
        liked_set = [movie_id for movie_id in user_item_csr[user_id].indices]

        for movie_id in self.movies:
            if movie_id in liked_set:
                continue
                
            rating = np.dot(self.user_embeddings[user_id], self.movie_embeddings[movie_id])
            
            if self.user_biases is not None:
                rating += self.user_biases[user_id]
            if self.movie_biases is not None:
                rating += self.movie_biases[movie_id]
            if self.gen_bias is not None:
                rating += self.gen_bias
                
            predicted_ratings.append((movie_id, rating))
            
        predicted_ratings.sort(key=lambda movie_rate: movie_rate[1], reverse=True)
        
        return predicted_ratings[:n]
    
    

def build_recsys(ratings, trained_model):
    
    users = ratings['user_id'].unique() 
    movies = ratings['movie_id'].unique()   
        
    user_embeddings = trained_model.user_embeddings
    movie_embeddings = trained_model.movie_embeddings
    
    try:
        movie_biases = trained_model.movie_biases
        user_biases = trained_model.user_biases
        gen_bias = trained_model.gen_bias
    except:
        movie_biases = None
        user_biases = None
        gen_bias = None
    
    return RecSys(users, movies, user_embeddings, movie_embeddings, movie_biases, user_biases, gen_bias)