from minepy import MINE
import numpy as np
import warnings
import openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pymfe.mfe import MFE
import scipy
from scipy import stats


warnings.filterwarnings("ignore")

def get_MIC_score(x, y):
    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()

def download_datasets(N = 20):
    openml_df = openml.datasets.list_datasets(output_format="dataframe")

    # Filter datasets
    filtered_df = openml_df[(openml_df.NumberOfInstances < 2000) &
                            (openml_df.NumberOfInstances > 50) &
                            (openml_df.NumberOfClasses > 0) &
                            (openml_df.NumberOfNumericFeatures > 0) &
                            (openml_df.NumberOfFeatures < 100) &
                            (openml_df.NumberOfSymbolicFeatures > 0) &
                            (openml_df.version == 1) &
                            (openml_df.status == 'active')
                        ]

    # Save the IDs of the filetered ones
    dids = filtered_df.did.values

    # List to save the datasets
    train_dataset_set = []

    for i in range(0, N):
        
        dataset_id = int(dids[i])
        
        try:
        
            # Get dataset from OpenML API
            dataset = openml.datasets.get_dataset(dataset_id)

            print("Retrieving dataset #:", i, " ID: ", dataset_id, ", \t Name:", dataset.name)

            # Format dataset to Numpy Arrays
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute)

            # Replace NaNs with feature mean
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            X = imp_mean.fit_transform(X)

            dataset = {
                'X': X,
                'y': y,
                'dataset_id': dataset_id,
                'dataset_name': dataset.name,
                'feature_names': attribute_names
            }

            train_dataset_set.append(dataset)
            
        except:
            print("Error dataset: \t\t", dataset_id, " - \t", dataset.name)
            continue

    return train_dataset_set

def get_metafeatures_dataset(ds):
    mfe = MFE()
    # print(f"Extracting: {ds['dataset_name']}")

    # Convert  sparse matrix if needed
    if (type(ds['X']) == scipy.sparse.csr.csr_matrix):
        ds['X'] = ds['X'].A

    # Create and fit the meta-feature extractor
    mfe.fit(ds['X'], ds['y'])
    metafeatures = mfe.extract()

    # Get just the value of the meta-feature
    metafeatures = np.array(metafeatures[1])

    # Replace NaNs with Zero
    metafeatures = np.nan_to_num(metafeatures) 

    if np.iscomplexobj(metafeatures):
        metafeatures = abs(metafeatures)
    
    # filter datasets with issues (< 111 metafeatures)
    if (len(metafeatures) == 111):
        return np.array(metafeatures).flatten()

def get_histogram(ds, bins = 20):

    histogram = []

    for x in ds['X'].T:
        h, _ = np.histogram(x, bins = bins)
        h = MinMaxScaler().fit_transform(h.reshape(-1, 1))
        histogram.append(h.flatten())
        
    return histogram

def _log_t(X):
    """ log transformation """
    
    if X.ndim == 1:
        # Check if the value can be evaluated with the log function
        if all(i > 0 for i in X):
            return True, np.log(X)
        
        else: return False, []
    
    else:
        X_aux = list()
        
        for x in X.T:
            # Check if the value can be evaluated with the log function
            if all(i > 0 for i in x):
                X_aux.append(np.log(x))
                
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def _square_root_t(X):
    """ Square-root Transformation """
    if X.ndim == 1:
        
        # Check if the value can be evaluated with the sqrt function
        if all(i > 0 for i in X):
            return True, np.sqrt(X)
        
        else: return False, []
    
    else:
        X_aux = list()
        
        for x in X.T:
            # Check if the value can be evaluated with the log function
            if all(i > 0 for i in x):
                X_aux.append(np.sqrt(x))
            
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def _square_t(X):
    """ Square Transformation """

    if X.ndim == 1:
        return True, np.square(X)
    
    else:
        X_aux = list()
        
        for x in X.T:
            X_aux.append(np.square(x))
            
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def _round_t(X):
    """ Round Transformation """ 

    if X.ndim == 1:
        return True, np.around(X)
    
    else:
        X_aux = list()
        
        for x in X.T:
            X_aux.append(np.around(x))
            
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def _frequency_t(X):
    """ frequency of a feature """

    if X.ndim == 1:
        F = list() # frequency list for each value
        
        unique_elements, counts_elements = np.unique(X, return_counts = True)

        for i in X:
            F.append(counts_elements[np.where(unique_elements == i)][0])
        return True, F
    
    else:
        X_aux = list()
        
        for x in X.T:
            F = list() # frequency list for each value
            unique_elements, counts_elements = np.unique(x, return_counts = True)
            for i in x:
                F.append(counts_elements[np.where(unique_elements == i)][0])

            X_aux.append(F)
            
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def _tanh_t(X):
    """ Hyperbolic Tanget """

    if X.ndim == 1:
        return True, np.tanh(X)
    
    else:
        X_aux = list()
        
        for x in X.T:
            X_aux.append(np.tanh(x))
            
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def __sigmoid(z):
    return 1/(1 + np.exp(-z))

def _sigmoid_t(X):
    """ Sigmoid function """
    
    if X.ndim == 1:
        return True, __sigmoid(X)
    
    else:
        X_aux = list()
        
        for x in X.T:
            X_aux.append(__sigmoid(x))
            
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def _zscore_t(X):
    """ ZScore function """
    if X.ndim == 1:
        
        if all(i > 0 for i in X):
            return True, stats.zscore(X)
        
        else: return False, []
            
    else:
        X_aux = list()
        
        for x in X.T:
            # Check if the value can be evaluated with the log function
            if all(i > 0 for i in x):
                X_aux.append(stats.zscore(x))
                            
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def _reciprocal_t(X):
    """ Recriprocal function """
    if X.ndim == 1:
        
        if all(i != 0 for i in X):
            return True, np.reciprocal(X)
        
        else: return False, []
            
    else:
        X_aux = list()
        
        for x in X.T:
            # Check if the value can be evaluated with the log function
            if all(i != 0 for i in x):
                X_aux.append(np.reciprocal(x))
                            
        if len(X_aux):
            return True, np.array(X_aux).T
        
        else: return False, []

def _exp_t(X):
    """ Exponential function """
    if X.ndim == 1:
        res = np.exp(X) 
        if np.any(np.isinf(res)):
            return False, []
        else:
            return True, np.nan_to_num(res)
                    
    else:
        X_aux = list()
        
        for x in X.T:
            res = np.exp(x) 
            if not np.any(np.isinf(res)):
                X_aux.append(np.nan_to_num(res))    
                  
        if len(X_aux):    
            return True, np.array(X_aux).T
        
def _minmax_t(X):
    if X.ndim == 1:
        X_ = MinMaxScaler().fit_transform(X.reshape(-1, 1))
        return True, X_.ravel()
    
    else:
        X_ = MinMaxScaler().fit_transform(X)
        return True, X_.ravel()
        
    return False, []

def _standard_scaler_t(X):
    if X.ndim == 1:
        X_ = StandardScaler().fit_transform(X.reshape(-1, 1))
        return True, X_.ravel()
    
    else:
        X_ = StandardScaler().fit_transform(X)
        return True, X_.ravel()
        
    return False, []
    
def _add(a, b):
    """ Addition """
    return a + b

def _sub(a, b):
    """ Subtract """
    return a - b

def _mul(a, b):
    """ Multiplitcation """
    return a * b

def _div(a, b):
    """ Divison """
    return a / b

def _mod(a, b):
    """ Modulo """
    return a % b

Unary_transformations = [_log_t, _square_root_t, _square_t, _round_t, _frequency_t, _tanh_t, _sigmoid_t, _zscore_t, _reciprocal_t, _minmax_t, _standard_scaler_t]

Binary_transformations = [_add, _sub, _mul, _div, _mod]