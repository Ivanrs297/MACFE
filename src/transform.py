from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from metafeatures import get_metafeatures_dataset, get_histogram
from tensorflow.keras.models import load_model
from unary_transformations import Transformations as Unary_transformations
from binary_transformations import Transformations as Binary_transformations
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pandas as pd

trans_symbols = ['log', 'sqrt', 'sqr', 'round', 'freq', 'tanh', 'sig', 'z', 'reci', 'minmax', 'std_scaler']

operations = ['+','-','*','/','%']

def preprocess_dataset(df):
    # Preprocess data
    df = df.replace('?', np.nan)

    # df = df.fillna(df.mean())
    
    X_raw = df.iloc[:,0:-1]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X_raw = X_raw.select_dtypes(include=numerics)
    X_raw = X_raw.fillna(X_raw.mean())
    X_raw = X_raw.astype('float32')

    le = LabelEncoder()
    y_raw = df.iloc[:,-1].values
    y_raw = le.fit_transform(y_raw)
    
    return X_raw, y_raw

def get_dataset_unary_encoding(X, y):

    ds = {
        'X': X,
        'y': y,
        'dataset_name': 'test'
    }

    mf, hist = get_metafeatures_dataset(ds), get_histogram(ds)
    
    ds_encodings = []
    
    for f_i in range(len(ds['X'].T)):
        encoding = np.concatenate( (mf, hist[f_i]),axis = 0)
        ds_encodings.append(encoding)

    ds_encodings = np.array(ds_encodings)
    ds_encodings = StandardScaler().fit_transform(ds_encodings)
    return ds_encodings

def transform_unary(X, y, column_names, TRM_dataset):

    new_column_names = []
    new_features = []

    # Get the encodings of all the features in dataset
    ds_encodings = get_dataset_unary_encoding(X, y)

    # check for NaN values
    if ( np.isnan(ds_encodings).any() or np.isinf(ds_encodings).any()):
        ds_encodings = np.nan_to_num(ds_encodings)

    # Get 1 neighbor for each feature
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric='euclidean', n_jobs = -1).fit(TRM_dataset[:, :-1])
    indices = nbrs.kneighbors(ds_encodings, return_distance = False)

    # Iterate in t indices
    for neigh_index, f_index in zip(indices, range(len(X.T))):
        
        t_index = int(TRM_dataset[neigh_index,-1])
        
        if t_index == len(Unary_transformations):
            # No suitable transformation found
            continue
        
        # The feature is already there
        new_column_name = f'{trans_symbols[t_index]}({column_names[f_index]})'
        if (new_column_name in new_column_names or new_column_name in column_names):
            continue
        
        success, t_values = Unary_transformations[t_index](X[:,f_index])

        if success and not np.isnan(t_values).any():
            new_features.append(t_values)
            new_column_names.append(new_column_name)

    if len(new_features) > 0:
        new_features = np.array(new_features)
        return new_features.T, new_column_names
    else:    
        return None, None

def transform_binary(X, y, column_names, TRM_binary_dataset):
    ds = {
        'X': X,
        'y': y,
        'dataset_name': 'test'
    }
    
    new_column_names = []
    new_features = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric='euclidean', n_jobs = -1).fit(TRM_binary_dataset[:, :-1])

    # Get the encodings of all the features in dataset
    mf, hist = get_metafeatures_dataset(ds), get_histogram(ds)

    for f_i in range(len(X.T)):
        for f_j in range(f_i + 1, len(X.T)):

            encoding = np.concatenate( (mf, hist[f_i], hist[f_j] ), axis = 0)

            # Get 1 neighbor for each feature
            neigh_index = nbrs.kneighbors([encoding], return_distance = False)
                
            t_index = int(TRM_binary_dataset[neigh_index, -1])
            
            if t_index == len(Binary_transformations):
                # No suitable transformation found
                continue

            # Test if the candidate feature is already there
            new_col_name = f"{operations[t_index]}({column_names[f_i]},{column_names[f_j]})"
            if (new_col_name in new_column_names or new_col_name in column_names):
                continue
            
            t_f = Binary_transformations[t_index](X[:,f_i], X[:,f_j])

            # if we made the transformation, then we check if there is any NaN or Inf value
            if ( np.isnan(t_f).any() or np.isinf(t_f).any()):
                continue

            new_features.append(t_f)
            new_column_names.append(new_col_name)
            break

    if len(new_features) > 0:
        new_features = np.array(new_features)
        return new_features.T, new_column_names
    else:    
        return None, None

def transform_scaler(X, y, column_names, TRM_scaler):

    ds = {
        'X': X,
        'y': y,
        'dataset_name': 'test'
    }

    scalers = [RobustScaler, StandardScaler, MinMaxScaler]
  
    # Get the encoding of current dataset
    ds_encoding = get_metafeatures_dataset(ds)

    # check for NaN values
    if ( np.isnan(ds_encoding).any() or np.isinf(ds_encoding).any()):
        ds_encoding = np.nan_to_num(ds_encoding)

    # get similar dataset
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric='euclidean', n_jobs = -1).fit(TRM_scaler[:, :-1])
    neigh_index = nbrs.kneighbors([ds_encoding], return_distance = False)[0]

    # get scaler index
    t_index = int(TRM_scaler[neigh_index,-1])

    # apply scaling
    scaler = scalers[t_index]()
    X_scaled = scaler.fit_transform(X)

    # rename columns
    scaler_prefix = ['robust_s', 'std_s', 'minmax_s']
    scaled_column_names = []
    for col_name in column_names:
        scaled_column_names.append(f"{scaler_prefix[t_index]}({col_name})")

    # create new scaled DF
    df_scaled = pd.DataFrame(X_scaled, columns = scaled_column_names)
    df_scaled['class'] = y
   
    return df_scaled
