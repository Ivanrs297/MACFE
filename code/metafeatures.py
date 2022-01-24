import openml
import numpy as np
from sklearn.impute import SimpleImputer
from pymfe.mfe import MFE
import scipy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
import pickle

warnings.filterwarnings("ignore")


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

if __name__ == "__main__":

    train_dataset_set = download_datasets(100)

    metafeature_set_datasets = [get_metafeatures_dataset(ds) for ds in train_dataset_set]
    metafeature_set_datasets = [ds for ds in metafeature_set_datasets if ds is not None]

    histogram_dataset = [get_histogram(ds) for ds in train_dataset_set]

    with open('data/train_dataset_set.pkl', 'wb') as f:
        pickle.dump(train_dataset_set, f)

    with open('data/metafeature_set_datasets.pkl', 'wb') as f:
        pickle.dump(metafeature_set_datasets, f)

    with open('data/histogram_dataset.pkl', 'wb') as f:
        pickle.dump(histogram_dataset, f)
