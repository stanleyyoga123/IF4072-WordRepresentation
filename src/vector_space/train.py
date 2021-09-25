import os
import pandas as pd
import numpy as np
import datetime

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from src.vector_space import vectorize_data

def get_model(
    model='lgbm',
    random_state=5816, # all
    n_estimators=500, # lgbm, xgb
    learning_rate=0.1, # xgb, lgbm
    num_leaves=20, # lgbm
    min_child_samples=10, # lgbm
    max_depth=24, # lgbm, xgb
    subsample=0.67, # xgb
    colsample_level=0.06, # xgb,
    C=9.302426503798305, # svm
    gamma=0.01335571976108241, # svm
    verbose=1
):
    if model == 'lgbm':
        return LGBMClassifier(
            n_estimators=n_estimators, verbose=verbose, random_state=random_state,
            learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, max_depth=max_depth
        ), True
    elif model == 'xgb':
        return XGBClassifier(
            n_estimators=n_estimators, tree_method='hist',
            subsample=subsample, colsample_level=colsample_level,
            verbose=verbose, random_state=random_state,
            eta=learning_rate, max_depth=max_depth
        ), True
    else:
        return SVC(
            C=C, break_ties=False, cache_size=200, class_weight=None,
            coef0=0.0, decision_function_shape='ovr', degree=3,
            gamma=gamma, kernel='rbf', max_iter=-1, probability=False,
            random_state=random_state, shrinking=True, tol=0.001, verbose=verbose
        ), False


def train_vector_space(
    x_train,
    y_train,
    x_test,
    y_test,
    validation_split=0.2,
    vectorizer='tfidf',
    max_features=5000,
    ngram_range=(1,1),
    model='lgbm',
    random_state=5816, # all
    n_estimators=1000, # lgbm, xgb
    learning_rate=0.1, # xgb, lgbm
    num_leaves=20, # lgbm
    min_child_samples=10, # lgbm
    max_depth=20, # lgbm, xgb
    subsample=0.67, # xgb
    colsample_level=0.06, # xgb,
    C=9.302426503798305, # svm
    gamma=0.01335571976108241, # svm
    verbose=1,
    save_pred=True,
    model_path=os.path.join("bin", "vspace"),
):
    x_train_v, x_test_v = vectorize_data(x_train, x_test,
        vectorizer=vectorizer, max_features=max_features, ngram_range=ngram_range)

    x_train_split, x_valid, y_train_split, y_valid = \
        train_test_split(x_train_v, y_train, test_size=validation_split, random_state=random_state)

    model, lgbm_or_xgb = get_model(
        model, n_estimators, random_state, learning_rate, num_leaves, min_child_samples, max_depth,
        subsample, colsample_level, C, gamma, verbose
    )
    
    if lgbm_or_xgb:
        model.fit(x_train_split, y_train_split, eval_set = [(x_valid, y_valid)], early_stopping_rounds=15, verbose=verbose)
    else:
        model.fit(x_train_split, y_train_split)

    y_pred = model.predict(x_test_v) 
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy: {}".format(accuracy))

    if save_pred:

        test_result_df = pd.DataFrame(
            {
                "text": x_test,
                "label": y_test,
                "pred": y_pred,
            }
        )
        # Type II error
        FP = test_result_df.loc[
            (test_result_df["label"] == 0) & (test_result_df["pred"] == 1)
        ]

        # Type I error
        FN = test_result_df.loc[
            (test_result_df["label"] == 1) & (test_result_df["pred"] == 0)
        ]

        print(
            f"False Positive (Type II Error) : {len(FP)} / {len(test_result_df)}"
        )
        FP.to_csv(os.path.join(model_path, "FP.csv"), index=False)

        print(
            f"False Negative (Type I Error) : {len(FN)} / {len(test_result_df)}"
        )
        FN.to_csv(os.path.join(model_path, "FN.csv"), index=False)