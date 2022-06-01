import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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

from scipy import stats


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
    
        
Transformations = [_log_t, _square_root_t, _square_t, _round_t, _frequency_t, _tanh_t, _sigmoid_t, _zscore_t, _reciprocal_t, _minmax_t, _standard_scaler_t]