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
    df["age_group"] = df["bd"].apply(age2group).astype("category")
    return df.drop(columns="bd")


def datetime2int(df):
    df["registration_init_year"] = df["registration_init_time"].apply(lambda x: x.year).astype("int")
    df = df.drop(columns="registration_init_time")
        
    df["expiration_date_year"] = df["expiration_date"].apply(lambda x: x.year).astype("int")
    df = df.drop(columns="expiration_date")
        
    return df


"""FILL_NAN: Categorical"""
def fill_nan(df, name):
    assert df[name].dtype == 'category', "TypeError"
    if name == "language":    
        df[name] = df[name].fillna(value="-1.0")
    else:    
        df[name] = df[name].cat.add_categories("FILL_NAN").fillna(value="FILL_NAN")
    return df


def fill_nan_list(df, name_list):
    for name in tqdm(name_list):
        df = fill_nan(df, name)
    return df

        
def fill_nan_all(df):
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
    df[f"{name}_count"] = df[name].apply(count_pipe).astype("int")
    return df
    

def count_union_list(df, name_list):
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
    df["isrc_country"] = to_country_once(df.isrc)
    df["isrc_country"] = df["isrc_country"].astype("category")
    df["isrc_year"] = df["isrc"].apply(to_year).astype("int")
    return df.drop(columns="isrc")


"""Convert"""
def to_category(df, columns):
    for column in columns:
        df[column] = df[column].astype("category")
    return df



"""
DATASETS
"""
def process_members(members_df):
    post_members_df = fill_nan(members_df, "gender")
    post_members_df = age2cat(post_members_df)
    post_members_df = datetime2int(post_members_df)
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
    extended_song_df["isrc_year"] = extended_song_df["isrc_year"].fillna(value=-1)
    extended_song_df["isrc_country"] = extended_song_df["isrc_country"].cat.add_categories("FILL_NAN").fillna(value="FILL_NAN")
    assert not extended_song_df.isnull().values.any(), "NaN is in dataframe"
    return extended_song_df


def merge_train(post_train_df, post_members_df, extended_song_df):    
    '''merge'''
    extended_train_df = post_train_df.merge(post_members_df, on="msno", how="left")
    extended_train_df = extended_train_df.merge(extended_song_df, on="song_id", how="left")
    '''nan, cat'''
    extended_train_df = to_category(extended_train_df, ["msno", "song_id"])
    
    return extended_train_df.sort_values(by="msno")


def get_final_df(df):
    final_df = df[~df.song_length.isnull()]
    
    for col in ["genre_ids_count", "artist_name_count", "composer_count", "lyricist_count", "isrc_year", "song_length"]:
        _series = final_df[col].copy().astype("int")
        final_df.loc[:, col] = _series
        
    assert not final_df.isnull().values.any(), "NaN in dataframe"

    return final_df


"""
DATA restoring
"""
def restore_csv(main_path, dct_path):
    
    dct_df = pd.read_csv(dct_path)
    dct = {}
    for i, row in dct_df.iterrows():
        dct[row["index"]] = row["dtypes"]
    
    restored = pd.read_csv(main_path, dtype=dct)
    return restored



"""
PIPELINE
"""
def get_preprocessed_dataset(csv_folder_path="kkbox-music-recommendation-challenge/csv_folder",
                             members_path="members.csv",
                             song_extra_info_path="song_extra_info.csv",
                             songs_path="songs.csv",
                             train_path="train.csv"):
    
    '''members'''
    print(f"process members...")
    members_df = load_members(f"{csv_folder_path}/{members_path}")
    post_members_df = process_members(members_df)
    del members_df

    '''songs'''
    print(f"process songs...")
    song_df = load_songs(f"{csv_folder_path}/{songs_path}")
    post_song_df = process_songs(song_df)
    del song_df

    '''songs extra info'''
    print(f"process songs extra info...")
    song_extra_info_df = load_song_extra_info(f"{csv_folder_path}/{song_extra_info_path}")
    post_song_extra_info_df = process_song_extra_info(song_extra_info_df)
    del song_extra_info_df
    
    '''merged songs'''
    print(f"merge songs...")
    extended_song_df = merge_songs(post_song_df, post_song_extra_info_df)
    del post_song_df
    del post_song_extra_info_df

    '''train'''
    print(f"process train...")
    train_df = load_train(f"{csv_folder_path}/{train_path}")
    post_train_df = process_train(train_df)
    del train_df

    '''extended train'''
    print(f"merge train...")
    extended_train_df = merge_train(post_train_df, post_members_df, extended_song_df)
    del post_train_df
    del post_members_df
    del extended_song_df
    
    '''final'''
    print(f"get final dataset...")
    full_extended_train_df = get_final_df(extended_train_df)
    del extended_train_df

    return full_extended_train_df





