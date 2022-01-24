import warnings
from IPython.display import Image
import pandas as pd
from causalnex.structure import DAGClassifier
# from keras.models import load_model
import pickle
import numpy as np
import pandas as pd
import argparse
import os
import sys
import time
from datetime import datetime
from transform import transform_unary, transform_binary, transform_scaler
warnings.filterwarnings("ignore")
 
from transform import preprocess_dataset
from evaluations import run_models

def get_TRMs():
    with open('data/TRM_set.pkl', 'rb') as f:
        TRM_set = pickle.load(f)
    TRM_dataset = list()
    for i in range(len(TRM_set)):
        TRM_dataset.append(
            np.append( TRM_set[i]['encoding'].ravel(), TRM_set[i]['top_t_index']))
    TRM_dataset = np.array(TRM_dataset)

    # with open('data/TRM_binary_set.pkl', 'rb') as f:
    with open('data/TRM_binary_set_maxf1f2.pkl', 'rb') as f:
        TRM_binary_set = pickle.load(f)
    TRM_binary_dataset = list()
    for i in range(len(TRM_binary_set)):
        TRM_binary_dataset.append(
            np.append( TRM_binary_set[i]['encoding'].ravel(), TRM_binary_set[i]['top_t_index']))
    TRM_binary_dataset = np.array(TRM_binary_dataset)

    with open('data/TRM_scaler_set.pkl', 'rb') as f:
        TRM_scaler_set = pickle.load(f)
    TRM_scaler_dataset = list()
    for i in range(len(TRM_scaler_set)):
        TRM_scaler_dataset.append(
            np.append( TRM_scaler_set[i]['encoding'].ravel(), TRM_scaler_set[i]['top_t_index']))
    TRM_scaler_dataset = np.array(TRM_scaler_dataset)

    return TRM_dataset, TRM_binary_dataset, TRM_scaler_dataset

def save_results_to_file(results, f1, acc, auc):
    scores =[
            f1.loc[[0]].values[0][0],
            f1.loc[[0]].values[0][1],
            f1.loc[[0]].values[0][2],
            f1.loc[[0]].values[0][3],
            f1.loc[[0]].values[0][4],
            f1.loc[[0]].values[0][5],
            f1.loc[[0]].values[0][6],
            f1.loc[[0]].values[0][7],
            acc.loc[[0]].values[0][0],
            acc.loc[[0]].values[0][1],
            acc.loc[[0]].values[0][2],
            acc.loc[[0]].values[0][3],
            acc.loc[[0]].values[0][4],
            acc.loc[[0]].values[0][5],
            acc.loc[[0]].values[0][6],
            acc.loc[[0]].values[0][7],
            auc.loc[[0]].values[0][0],
            auc.loc[[0]].values[0][1],
            auc.loc[[0]].values[0][2],
            auc.loc[[0]].values[0][3],
            auc.loc[[0]].values[0][4],
            auc.loc[[0]].values[0][5],
            auc.loc[[0]].values[0][6],
            auc.loc[[0]].values[0][7],
    ]
    results = results + scores + [datetime.now()]
    a = np.asarray(results)

    # Open file and append experiment results
    with open(f"results.csv", "ab+") as f:
        np.savetxt(f, a.reshape(1, a.shape[0]), delimiter=',', fmt='%s', newline = "\n")

def _feature_construction_step(df, TRM_dataset, TRM_binary_dataset):
    X, y = preprocess_dataset(df)
    _column_names = df.columns.tolist()[:-1].copy()

    X_new_unary, _column_names_unary = transform_unary(X.values, y, _column_names, TRM_dataset)
    X_new_binary, _column_names_binary = transform_binary(X.values, y, _column_names, TRM_binary_dataset)
    
    # New DF with novel features
    df_e = df.copy()
    df_e = df_e.drop(['class'], axis = 1)

    if (X_new_unary is not None):
        df_unary = pd.DataFrame(X_new_unary, columns = _column_names_unary)
        df_e = pd.concat([df_e, df_unary], axis=1)

    if (X_new_binary is not None):
        df_binary = pd.DataFrame(X_new_binary, columns = _column_names_binary)
        df_e = pd.concat([df_e, df_binary], axis=1)

    df_e['class'] = y

    return df_e 

def feature_constuction(df_original, d_list):
    df_engineered_list = []
    print("Construction...")
    df_engineered = df_original.copy(deep=True)

    max_d = max(d_list)

    for d_i in range(1, max_d + 1):
        df_engineered = _feature_construction_step(df_engineered, TRM_dataset, TRM_binary_dataset)
        # Drop Original Features from engineered ones
        df_engineered_d = df_engineered.drop(df_original.columns, axis = 1)

        if (d_i in d_list):
            print(f"d:{d_i} done.")
            # Add df to the testing list if current d in in d_list
            df_engineered_list.append(df_engineered_d)

    return df_engineered_list

def feature_selection(df, s_list):
    print("Selection...")
    X, y = preprocess_dataset(df)
    y = pd.Series(y, name="class")

    dag = DAGClassifier(
        alpha=0.01,
        beta=0.5,
        hidden_layer_units=[5],
        fit_intercept=True,
        standardize=True
    )
    X = X.astype(float)
    y = y.astype(int)
    dag.fit(X.values, y)

    # List to save features for each "s"
    df_selected_list = []
    
    for threshold in s_list:
        # Select top (1 - threshold)% 
        _threshold = np.quantile(dag.feature_importances_[0], (1.0 - threshold))
        selection_idx = np.where(dag.feature_importances_[0] >= _threshold)[0]
        X_selected = X.iloc[:, selection_idx]

        _column_names = np.array(df.columns.tolist()[:-1])
        _column_names = _column_names[selection_idx]

        # Create new selected DataFrame
        df_e = pd.DataFrame(X_selected)
        df_e.columns = _column_names
        df_e['class'] = y
        df_selected_list.append(df_e)
        print(f's:{threshold}, Dim:{df_e.shape[1] - 1}')

    return df_selected_list

def feature_scaler(df):
    X = df.drop(['class'], axis = 1)
    y = df['class']

    column_names = X.columns.tolist().copy()

    df_scaled = transform_scaler(
        X.values,
        y.values,
        column_names,
        TRM_scaler)
    return df_scaled


def model_evaluation(df, method, dataset_file, s, d):
    # Run models
    X, y = df.drop(['class'], axis = 1).values, df['class'].values
    f1_score, acc_score, auc_score = run_models(X, y)
    # print(f"Mean F1 score: {np.round(np.mean(f1_score.values), 2)}")
    save_results_to_file(
        [dataset_file,method,s,d, df.shape[1] - 1],
        f1_score, acc_score, auc_score
    )

    # Save Transformed Dataset to file
    df.to_csv(f'datasets_output/{dataset_file}_MACFE_s{s}_d{d}.csv', index = False)

if __name__ == "__main__":
    start = time.time()

    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d_l','--Depth_list',
        nargs='+', help='<"d" list',
        type=int,
        required=True
    )

    parser.add_argument('-s_l','--Selection_list',
        nargs='+', help='<"s" list',
        type=float,
        required=True
    )

    args = parser.parse_args()
    s_list, d_list = args.Selection_list, args.Depth_list

    # Read TRMs (Unary and Binary)
    TRM_dataset, TRM_binary_dataset, TRM_scaler = get_TRMs()

    # Iterate though datasets
    datasets_directory = 'datasets_input'
    for dataset_index, dataset_file in enumerate(os.listdir(datasets_directory)):

        if dataset_file.endswith(".gitignore"):
            continue

        df_original = pd.read_csv(f'datasets_input/{dataset_file}', header= 0)

        # START - Automated Feature Engineering Process
        print(f"\n*** Starting MACFE, d:{d_list}, s:{s_list} ***\n")
        print(f"Working on {dataset_file}")

        # Baseline
        print("Original Dim: ", df_original.shape[1] - 1)
        X, y = preprocess_dataset(df_original)
        f1_score, acc_score, auc_score = run_models(X, y)
        save_results_to_file(
            [ dataset_file,'original',1,0,df_original.shape[1] - 1],
            f1_score, acc_score, auc_score
        )


        # Selection
        df_selected_list = feature_selection(df_original, s_list)

        for s, df_selected in zip(s_list, df_selected_list):
            # Construction
            df_engineered_list = feature_constuction(df_selected, d_list)

            # Evaluation
            print("Evaluation...")
            for d, df_engineered in zip(d_list, df_engineered_list):
                # Concatenate Selected and Engineered features
                df = pd.concat([df_selected, df_engineered], axis = 1)

                # Scale Features
                df = feature_scaler(df)

                # Run models
                model_evaluation(df, 'MACFE_sel+engin(sel)', dataset_file, s, d)

                print(f"DONE!! MACFE with s:{s}, d: {d} - Dim: {df.shape[1] - 1}")


    end = time.time()
    print("\MACFE Elapsed time: ", end - start)
