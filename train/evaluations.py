from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import numpy as np
import pandas as pd

def run_models(X, y, random_state = 42):
    model_names = [
        "Nearest Neighbors",
        "Logistic Regression",
        "Linear SVM",
        "Poly SVM",
        "Random Forest",
        "AdaBoost",
        "NN",
        "Decision Tree"
        ]
    
    classifiers = [
        KNeighborsClassifier(),
        LogisticRegression(random_state = random_state),
        SVC(kernel="linear", max_iter=3000, probability=True, random_state = random_state),
        SVC(kernel='poly', max_iter=3000, probability=True, random_state = random_state),
        RandomForestClassifier(random_state = random_state),
        AdaBoostClassifier(),
        MLPClassifier(),
        tree.DecisionTreeClassifier()
    ]
    
    f1_scores = []
    accuracy_scores = []
    auc_scores = []

    # if multi-label
    if (len(set(y)) > 2):
        f1_method = 'f1_weighted'
        auc_method = 'roc_auc_ovo_weighted'
    else:
        f1_method = 'f1'
        auc_method = 'roc_auc'

    scoring = {
        'acc': 'accuracy',
        # 'f1_score': f1_method,
        # 'auc': auc_method
        }

    # Iterate classifiers
    for name, clf in zip(model_names, classifiers):
        strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        score = cross_validate(clf, X, y, cv = strat_k_fold, scoring = scoring)
        # f1_scores.append(np.round(np.mean(score['test_f1_score']) * 100, 2))
        accuracy_scores.append(np.round(np.mean(score['test_acc']) * 100, 2))
        # auc_scores.append(np.round(np.mean(score['test_auc']) * 100, 2))

    # df_f1 = pd.DataFrame(f1_scores)
    # df_f1 = df_f1.T
    # df_f1.columns = model_names

    df_acc = pd.DataFrame(accuracy_scores)
    df_acc = df_acc.T
    df_acc.columns = model_names

    # df_auc = pd.DataFrame(auc_scores)
    # df_auc = df_auc.T
    # df_auc.columns = model_names
        
    # return df_acc, df_f1, df_auc
    return df_acc, df_acc, df_acc
