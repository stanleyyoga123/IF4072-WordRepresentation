import os
import pandas as pd
import numpy as np
import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from src.vector_space import vectorize_data

def train_vector_space(
    x_train,
    y_train,
    x_test,
    y_test,
    validation_split=0.2,
    max_features=5000,
    ngram_range=(1,1),
    n_estimators=500,
    feature_fraction=0.06,
    bagging_fraction=0.67,
    bagging_freq=1,
    random_state=5816,
    learning_rate=0.1,
    num_leaves=20,
    min_child_samples=10,
    max_depth=24,
    verbose=1,
    save_pred=True,
    model_path=os.path.join("bin", "vspace"),
):
    # TFIDF Vectorize
    x_train_v, x_test_v = vectorize_data(x_train, x_test, max_features=max_features, ngram_range=ngram_range)
    x_train_split, x_valid, y_train_split, y_valid = \
        train_test_split(x_train_v, y_train, test_size=validation_split, random_state=random_state)
    lgbm = LGBMClassifier(
        n_estimators=n_estimators,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        verbose=verbose,
        random_state=random_state,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        max_depth=max_depth
    )
    lgbm.fit(x_train_split, y_train_split, eval_set = [(x_valid, y_valid)], early_stopping_rounds=15, verbose=verbose)
    y_pred = lgbm.predict(x_test_v) 
    accuracy = np.mean(y_pred == y_test)
    print(accuracy)

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