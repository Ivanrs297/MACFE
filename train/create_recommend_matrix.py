from utils import Unary_transformations as Unary_transformations
from utils import Binary_transformations as Binary_transformations
from utils import get_MIC_score
import numpy as np
import pickle

def evaluation_performance(index_f, t_f, X, y, task = 'classification' ):    
    original = get_MIC_score(X[:, index_f], y)
    transformed = get_MIC_score(t_f, y)
    return transformed- original

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
                TRM_set.append(np.append(encoding.ravel(), len(Unary_transformations)))

            else:
                TRM_set.append(np.append(encoding.ravel(), top_t_index))
    
    return np.array(TRM_set)

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
                    TRM_binary_set.append(np.append(encoding.ravel(), len(Binary_transformations)))

                else:
                    TRM_binary_set.append(np.append(encoding.ravel(), top_t_index))

    return np.array(TRM_binary_set)

if __name__ == "__main__":

    with open('../data/train_dataset_set.pkl', 'rb') as f:
        train_dataset_set = pickle.load(f)

    with open('../data/metafeature_set_datasets.pkl', 'rb') as f:
        metafeature_set_datasets = pickle.load(f)

    with open('../data/histogram_dataset.pkl', 'rb') as f:
        histogram_dataset = pickle.load(f)

    # TRM_set = fit_unary_transformations(train_dataset_set, metafeature_set_datasets, histogram_dataset)
    # with open('../data/TRM_set.npy', 'wb') as f:
    #     # pickle.dump(TRM_set, f)
    #     np.save(f, TRM_set)

    TRM_binary_set = fit_binary_transformations(train_dataset_set, metafeature_set_datasets, histogram_dataset)
    with open('../data/TRM_binary_set.npy', 'wb') as f:
        # pickle.dump(TRM_binary_set, f)
        np.save(f, TRM_binary_set)



