from unary_transformations import Transformations as Unary_transformations
from binary_transformations import Transformations as Binary_transformations
import ppscore as pps
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import shapiro

warnings.filterwarnings("ignore")

def evaluation_performance(index_f, t_f, X, y, task = 'classification' ):    
    df = pd.DataFrame()
    df['x'] = X[:, index_f]
    df['x_t'] = t_f
    
    df['y'] = y
    df['y'] = df['y'].astype('category')
        
    original = pps.score(df, 'x', "y")
    transformed = pps.score(df, 'x_t', "y")
        
    return transformed['ppscore'] - original['ppscore']

def fit_unary_transformations(train_dataset_set, metafeature_set_datasets, histogram_dataset):

    TRM_set = []

    for ds_i, ds in enumerate(train_dataset_set):
        print("Working on Dataset: ", ds['dataset_name'])

        current_dataset_metafeatures = metafeature_set_datasets[ds_i]

        # Get Features and targets from dataset
        X, y = ds['X'], ds['y']

        # Iterate through features
        for f_i in range(len(X.T)):

            # list to save scores
            scores = list()

            # get the histogram of current feature
            current_f_histogram = histogram_dataset[ds_i][f_i]

            # Encoding: Dataset MFs + Feature Histogram
            encoding = np.concatenate(
                (current_dataset_metafeatures, current_f_histogram),
                axis = 0
            )

            # Execute each of the transformations on T set
            for i_t, t in enumerate(Unary_transformations):

                # t_f = transformed feature
                success, t_f = t(X[:,f_i])

                # if we made the transformation, then we evaluate the resulting feature
                if success:
                    scores.append( evaluation_performance(f_i, t_f, X, y, task = 'classification'))
                else:
                    scores.append(0)

            # Save the top transformation index for dataset/feature
            top_t_index = np.argmax(scores)

            if scores[top_t_index] <= 0: 
                # Then, the transformation is useless
                TRM_set.append({
                    'encoding': encoding,
                    'top_t_index': len(Unary_transformations)
                })

            else:
                TRM_set.append({
                    'encoding': encoding,
                    'top_t_index': top_t_index
                })
    
    return TRM_set

def fit_binary_transformations(train_dataset_set, metafeature_set_datasets, histogram_dataset):
    TRM_binary_set = []

    for ds_i, ds in enumerate(train_dataset_set):
        print("Working on Dataset: ", ds['dataset_name'])

        current_dataset_metafeatures = metafeature_set_datasets[ds_i]

        # Get Features and targets from dataset
        X, y = ds['X'], ds['y']

        # Iterate through features
        for f_i in range(len(X.T)):

            # get the histogram of current feature
            current_f_i_histogram = histogram_dataset[ds_i][f_i]
                            
            for f_j in range(f_i + 1, len(X.T)):    # To reduce iterations
                    
                # list to save scores
                scores = list()
                    
                # get the histogram of current feature
                current_f_j_histogram = histogram_dataset[ds_i][f_j]

                # Encoding: Dataset MFs + Feature_i Histogram + Feature_j Histogram
                encoding = np.concatenate(
                    (current_dataset_metafeatures, current_f_i_histogram, current_f_j_histogram),
                    axis = 0)

                # Execute each of the transformations on T set
                for i_t, t in enumerate(Binary_transformations):

                    # t_f = transformed feature
                    t_f = t(X[:,f_i], X[:,f_j])

                    # if we made the transformation, then we check if there is any NaN or Inf value
                    if ( np.isnan(t_f).any() or np.isinf(t_f).any()):
                        continue
                    
                    # Score of feature_i  vs transformed feature (t_f)
                    s1 =  evaluation_performance(f_i, t_f, X, y, task = 'classification')

                    # Score of feature_j  vs transformed feature (t_f)
                    s2 =  evaluation_performance(f_j, t_f, X, y, task = 'classification')

                    # We save the max score between s1 and s2 if they are more informative than the original one
                    if (s1 > 0 and s2 > 0):
                        scores.append(max(s1, s2))
                    
                    # Else, the resulting feature is non-informative
                    else:
                        scores.append(0)
                        
                # Save the top transformation index for dataset/feature
                top_t_index = np.argmax(scores)

                if scores[top_t_index] <= 0: 
                    # Then, the transformation is useless
                    TRM_binary_set.append({
                        'encoding': encoding,
                        'top_t_index': len(Binary_transformations)
                    })

                else:
                    TRM_binary_set.append({
                        'encoding': encoding,
                        'top_t_index': top_t_index
                    })

    return TRM_binary_set

def _test_normal_distribution(X, threshold = 0.5):
    # Count of normal distributions in data
    count = 0
    cols = len(X[0])
    
    # Iterate colums
    for i in range(cols):
        # If p-value > 0.05 then is normal
        if shapiro(X[:,i])[1] > 0.05:
            count += 1
            
    if count/cols > threshold:
        return True
    else: 
        return False

def _test_outliers_for_robust_scaler(X, threshold = 0.11):
    iso = IsolationForest(random_state = 42, n_jobs = 4, contamination = 0.1)
    outliers = iso.fit_predict(X)
    count_outliers = np.count_nonzero(outliers == -1)
    if (count_outliers / len(X)) >= threshold:
        return True
    else:
        return False

def scale_features(X):
    if _test_normal_distribution(X):
        return StandardScaler().fit_transform(X)
    else:
        return MinMaxScaler().fit_transform(X)

def fit_scaler_transformations(train_dataset_set, metafeature_set_datasets):
    # Scaler Recommendation Matrix
    TRM_scalers = []
        
    # for each dataset
    for ds_i, ds in enumerate(train_dataset_set):
        current_dataset_metafeatures = metafeature_set_datasets[ds_i]
        print("Working on Dataset: ", ds['dataset_name'])      

        # Get Features and targets from dataset
        X, y = ds['X'], ds['y']
        
        if (_test_outliers_for_robust_scaler(X)):
            scaler_index = 0
            
        elif (_test_normal_distribution(X)):
            scaler_index = 1
        else:
            scaler_index = 2

        TRM_scalers.append({
            'encoding': current_dataset_metafeatures,
            'top_t_index': scaler_index
        })
            
    # SRM.to_csv(f'./Exported_data/Datasets_scores_scalers/dataset_scalers.csv', sep=",", index = False)
    return TRM_scalers


if __name__ == "__main__":

    with open('data/train_dataset_set.pkl', 'rb') as f:
        train_dataset_set = pickle.load(f)

    with open('data/metafeature_set_datasets.pkl', 'rb') as f:
        metafeature_set_datasets = pickle.load(f)

    with open('data/histogram_dataset.pkl', 'rb') as f:
        histogram_dataset = pickle.load(f)

    # TRM_set = fit_unary_transformations(train_dataset_set, metafeature_set_datasets, histogram_dataset)
    # with open('data/TRM_set.pkl', 'wb') as f:
    #     pickle.dump(TRM_set, f)

    # TRM_binary_set = fit_binary_transformations(train_dataset_set, metafeature_set_datasets, histogram_dataset)
    # with open('data/TRM_binary_set_maxf1f2.pkl', 'wb') as f:
    #     pickle.dump(TRM_binary_set, f)

    TRM_scaler_set = fit_scaler_transformations(train_dataset_set, metafeature_set_datasets)
    with open('data/TRM_scaler_set.pkl', 'wb') as f:
        pickle.dump(TRM_scaler_set, f)



