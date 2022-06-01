from utils import download_datasets, get_metafeatures_dataset, get_histogram
import warnings
import pickle

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    train_dataset_set = download_datasets(10)

    metafeature_set_datasets = [get_metafeatures_dataset(ds) for ds in train_dataset_set]
    metafeature_set_datasets = [ds for ds in metafeature_set_datasets if ds is not None]

    histogram_dataset = [get_histogram(ds) for ds in train_dataset_set]

    with open('../data/train_dataset_set.pkl', 'wb') as f:
        pickle.dump(train_dataset_set, f)

    with open('../data/metafeature_set_datasets.pkl', 'wb') as f:
        pickle.dump(metafeature_set_datasets, f)

    with open('../data/histogram_dataset.pkl', 'wb') as f:
        pickle.dump(histogram_dataset, f)