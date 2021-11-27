import csv
import numpy as np
import pandas as pd
from tqdm import tqdm


"""
DATA LOADING
"""        
def load_members(path):
    return pd.read_csv(path, 
                       dtype={
                        "msno": "category",
                        "city": "category",
                        "bd": np.uint8,
                        "gender": "category",
                        "registered_via": "category"
                       }, 
                       parse_dates=["registration_init_time", "expiration_date"])



def load_song_extra_info(path):
    return pd.read_csv(path, 
                       dtype={
                        "song_id": "category",
                        "name": "category"
                       })


def load_songs(path):
    return pd.read_csv(path, 
                       dtype={
                        "song_id": "category",
                        "song_length": np.int32,
                        "genre_ids": "category",
                        "artist_name": "category",
                        "composer": "category",
                        "lyricist": "category",
                        "language": "category"
                       })


def load_train(path):
    return pd.read_csv(path, 
                       dtype={
                        "msno": "category",
                        "song_id": "category",
                        "source_system_tab": "category",
                        "source_screen_name": "category",
                        "source_type": "category",
                        "target": np.uint8
                       })


"""
DATA ANALYSIS
"""
def show_unique_values(df):
    print("Unique values:")
    print()
    for name in df.columns:
        print(f"{name}: {df[name].unique().shape}")

        
"""
DATA PROCESSING
"""

"""Age"""
def age2group(age):
    if age == 0 or age > 100:
        return "FILL_NAN"
    elif 0 < age <= 4:
        return "FILL_NAN"
    elif 4 < age <= 12:
        return "child"
    elif 12 < age <= 18:
        return "teen"
    elif 18 < age <= 30:
        return "young"
    elif 30 < age <= 45:
        return "normal"
    elif 45 < age <= 60:
        return "middle"
    elif 60 < age <= 80:
        return "elder"
    elif 80 < age <= 100:
        return "top"


def age2cat(df):
    df = df.copy()
    df["age_group"] = df["bd"].apply(age2group).astype("category")
    return df.drop(columns="bd")


"""FILL_NAN: Categorical"""
def fill_nan(df, name):
    assert df[name].dtype == 'category', "TypeError"
    df = df.copy()
    if name == "language":    
        df[name] = df[name].fillna(value="-1.0")
    else:    
        df[name] = df[name].cat.add_categories("FILL_NAN").fillna(value="FILL_NAN")
    return df


def fill_nan_list(df, name_list):
    df = df.copy()
    for name in tqdm(name_list):
        df = fill_nan(df, name)
    return df

        
def fill_nan_all(df):
    df = df.copy()
    for name in df.columns:
        if df[name].isnull().values.any():  
            df = fill_nan(df, name)
    return df

        
"""COUNT featurings"""
def count_pipe(_str):
    if _str == "FILL_NAN":
        return 0
    return sum(map(_str.count, ["|", "/", "\\", ";", "&", "and"])) + 1


def count_union(df, name):
    df = df.copy()
    df[f"{name}_count"] = df[name].apply(count_pipe).astype("int")
    return df
    

def count_union_list(df, name_list):
    df = df.copy()
    for name in name_list:
        df = count_union(df, name)
    return df


"""ISRC"""
def get_countries_set(path='wikipedia-iso-country-codes.csv'):
    countries = {}
    with open(path, 'r') as csvfile:
        codes = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in codes:
            countries[row[1]] = row[0]
    return countries


def to_country_once(df_isrc):
    countries = get_countries_set()
#     print(countries)
    
    lst = []
    for isrc in df_isrc:
        if str(isrc) == "nan":
            lst.append(-1)
            
        else:
            country_code = str(isrc[0:2])
            try:
                lst.append(countries[country_code])
            except:
                lst.append(-1)
            
    return lst


def to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return -1

    
def process_isrc(df):
    df = df.copy()
    df["isrc_country"] = to_country_once(df.isrc)
    df["isrc_country"] = df["isrc_country"].astype("category")
    df["isrc_year"] = df["isrc"].apply(to_year).astype("category")
    return df.drop(columns="isrc")


"""Convert"""
def to_category(df, columns):
    df = df.copy()
    for column in columns:
        df[column] = df[column].astype("category")
    return df



"""
DATASETS
"""
def process_members(members_df):
    post_members_df = fill_nan(members_df, "gender")
    post_members_df = age2cat(post_members_df)
    return post_members_df


def process_songs(song_df):
    post_song_df = fill_nan_all(song_df)
    post_song_df = count_union_list(post_song_df, ["genre_ids", "artist_name", "composer", "lyricist"])
    return post_song_df 
    

def process_song_extra_info(song_extra_info_df):    
    post_song_extra_info_df = process_isrc(song_extra_info_df)
    return post_song_extra_info_df.drop(columns="name")
    
    
def process_train(train_df):   
    post_train_df = fill_nan_list(train_df, ["source_system_tab", "source_screen_name", "source_type"])
    return post_train_df
    
    
def merge_songs(post_song_df, post_song_extra_info_df): 
    extended_song_df = post_song_df.merge(post_song_extra_info_df, on="song_id", how="left")
    for col in extended_song_df.columns:
        if extended_song_df[col].dtype == "category" and extended_song_df[col].isnull().any():
            if "FILL_NAN" in extended_song_df[col].cat.categories:
                extended_song_df[col] = extended_song_df[col].fillna(value="FILL_NAN")
            else:
                extended_song_df[col] = extended_song_df[col].cat.add_categories("FILL_NAN").fillna(value="FILL_NAN")
                
    return extended_song_df
    

def merge_train(post_train_df, post_members_df, extended_song_df):    
    '''merge'''
    extended_train_df = post_train_df.merge(post_members_df, on="msno", how="left")
    extended_train_df = extended_train_df.merge(extended_song_df, on="song_id", how="left")
    '''nan, cat'''
    extended_train_df = to_category(extended_train_df, ["msno", "song_id"])
    
    return extended_train_df.sort_values(by="msno")


